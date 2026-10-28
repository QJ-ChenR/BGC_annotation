"""Local frozen ModernBERT DNA features with full or windowed context.

Full mode uses one backbone forward per BGC, without imposing the saved
max_position_embeddings as an input-length cap. Windowed mode retains the
original conservative baseline. Neither mode changes weights or RoPE settings.
"""

from hashlib import sha256
import json
from pathlib import Path
import warnings

import torch


class DNASequenceTooLongError(ValueError):
    """An explicit full-sequence resource limit was exceeded; never truncate."""

    def __init__(self, input_tokens, limit):
        self.input_tokens = input_tokens
        self.limit = limit
        super().__init__(
            f"Full DNA input requires {input_tokens} tokens including CLS/SEP, "
            f"exceeding the explicit max_tokens={limit}. No truncation or "
            "windowed fallback was performed."
        )


def _sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dtype(value):
    choices = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if isinstance(value, torch.dtype) and value in choices.values():
        return value
    if isinstance(value, str) and value in choices:
        return choices[value]
    raise ValueError("dtype must be float32, float16, or bfloat16")


def _weight_files(directory):
    """Identify the local HF weight set, including all indexed shards."""
    for name in (
        "model.safetensors", "model.safetensors.index.json",
        "pytorch_model.bin", "pytorch_model.bin.index.json",
    ):
        candidate = directory / name
        if not candidate.is_file():
            continue
        if not name.endswith(".index.json"):
            return [candidate]
        index = json.loads(candidate.read_text())
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid checkpoint shard index: {candidate}")
        files = [candidate]
        for shard in sorted(set(weight_map.values())):
            path = (directory / shard).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("Checkpoint shard must be inside checkpoint directory")
            if not path.is_file():
                raise FileNotFoundError(f"Missing checkpoint shard: {path}")
            files.append(path)
        return files
    raise FileNotFoundError(
        f"No trained HF model weights found in {directory}; config alone is insufficient"
    )


class FrozenDNA:
    """Return one unit-normalized CPU float32 vector per entire DNA sequence.

    The BPE tokenizer processes the full sequence without truncation. Its content
    tokens are either passed together in one forward (``sequence_mode='full'``)
    or divided into disjoint windows (the backwards-compatible default).
    CLS/SEP are added per forward sequence. Final backbone hidden states are
    averaged over *content tokens*, including UNK, then L2-normalized once.
    Neither special tokens nor padding participate in the average. BPE tokens
    have variable nucleotide lengths; this is intentionally token-weighted.
    """

    def __init__(
        self, model, tokenizer, *, device="cpu", dtype="float32",
        attention_backend="sdpa", max_tokens=None, batch_size=1,
        pooling="token_mean", provenance=None, sequence_mode="windowed",
    ):
        if pooling != "token_mean":
            raise ValueError("DNA pooling must be 'token_mean'")
        if attention_backend not in ("sdpa", "eager", "flash_attention_2"):
            raise ValueError("Unsupported attention backend")
        if sequence_mode not in ("windowed", "full"):
            raise ValueError("sequence_mode must be 'windowed' or 'full'")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if sequence_mode == "full" and batch_size != 1:
            raise ValueError("Full sequence mode requires batch_size=1 (one BGC per forward)")
        if attention_backend == "flash_attention_2" and _dtype(dtype) == torch.float32:
            raise ValueError("flash_attention_2 requires float16 or bfloat16; use sdpa for float32")
        config = model.config
        if getattr(config, "model_type", None) != "modernbert":
            raise ValueError("Expected a ModernBERT backbone")
        configured_limit = int(config.max_position_embeddings)
        if sequence_mode == "windowed" and max_tokens is None:
            max_tokens = configured_limit
        if max_tokens is not None and (
            not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 3
        ):
            raise ValueError("max_tokens must be an integer greater than 2, including CLS/SEP")
        if sequence_mode == "windowed" and max_tokens > configured_limit:
            raise ValueError(
                "Windowed max_tokens must not exceed "
                f"config.max_position_embeddings ({configured_limit}); "
                "use sequence_mode='full' for a full BGC forward"
            )
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("A fast tokenizer with nucleotide offsets is required")
        vocab_size = int(config.vocab_size)
        if len(tokenizer) != vocab_size:
            raise ValueError("Tokenizer vocabulary size does not match checkpoint")
        if model.get_input_embeddings().weight.shape[0] != vocab_size:
            raise ValueError("Input embedding rows do not match checkpoint vocabulary")
        token_ids = {}
        for name in ("cls", "sep", "pad", "unk"):
            value = getattr(tokenizer, f"{name}_token_id", None)
            if not isinstance(value, int) or not 0 <= value < vocab_size:
                raise ValueError(f"Invalid tokenizer {name}_token_id: {value}")
            token_ids[name] = value
        if len(set(token_ids.values())) != len(token_ids):
            raise ValueError("CLS, SEP, PAD, and UNK must have distinct token IDs")
        if getattr(config, "pad_token_id", None) != token_ids["pad"]:
            raise ValueError("Tokenizer PAD ID does not match checkpoint pad_token_id")
        embedding_pad = getattr(model.get_input_embeddings(), "padding_idx", None)
        if embedding_pad is not None and embedding_pad != token_ids["pad"]:
            raise ValueError("Tokenizer PAD ID does not match input embedding padding_idx")
        # These are metadata corrections, never vocabulary/embedding resizing.
        corrections = {}
        for name, actual in {
            "cls_token_id": token_ids["cls"], "bos_token_id": token_ids["cls"],
            "sep_token_id": token_ids["sep"], "eos_token_id": token_ids["sep"],
        }.items():
            original = getattr(config, name, None)
            if original != actual:
                corrections[name] = {"config_value": original, "tokenizer_value": actual}
                setattr(config, name, actual)
        self.device = torch.device(device)
        self.dtype = _dtype(dtype)
        self.model = model.to(device=self.device, dtype=self.dtype)
        self.model.requires_grad_(False)
        self.model.eval()
        self.tokenizer = tokenizer
        self.special_ids = token_ids
        self.embedding_dim = int(config.hidden_size)
        self.sequence_mode = sequence_mode
        self.max_tokens = max_tokens
        self.max_content_tokens = None if max_tokens is None else max_tokens - 2
        self.batch_size = batch_size
        self.attention_backend = attention_backend
        self.pooling = pooling
        self.provenance = dict(provenance or {})
        self.special_id_corrections = corrections

    @classmethod
    def from_checkpoint(
        cls, path, *, tokenizer_path=None, config_path=None, device="cpu",
        dtype="float32", attention_backend="sdpa", max_tokens=None,
        batch_size=1, pooling="token_mean", sequence_mode="windowed",
    ):
        """Load an entire trained MLM checkpoint strictly and retain its backbone.

        Everything is local-only. Missing, unexpected, or mismatched parameters
        are fatal: a newly initialized encoder is never a fallback.
        """
        path = Path(path).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Local checkpoint directory not found: {path}")
        # Reject unsupported FA2 precision before allocating checkpoint tensors.
        if attention_backend == "flash_attention_2" and _dtype(dtype) == torch.float32:
            raise ValueError("flash_attention_2 requires float16 or bfloat16; use sdpa for float32")
        files = _weight_files(path)
        tokenizer_path = Path(tokenizer_path or path).resolve()
        config_path = Path(config_path or path).resolve()
        from transformers import AutoConfig, AutoTokenizer, ModernBertForMaskedLM
        import transformers

        config = AutoConfig.from_pretrained(
            str(config_path), local_files_only=True, trust_remote_code=False,
        )
        if config.model_type != "modernbert":
            raise ValueError("Checkpoint model_type must be modernbert")
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), local_files_only=True, trust_remote_code=False,
            use_fast=True,
        )
        # Validating the PAD setting before loading avoids treating a different
        # vocabulary as compatible just because its embedding matrix fits.
        if config.pad_token_id != tokenizer.pad_token_id:
            raise ValueError("Tokenizer PAD ID does not match checkpoint pad_token_id")
        loaded, info = ModernBertForMaskedLM.from_pretrained(
            str(path), config=config, local_files_only=True,
            dtype=_dtype(dtype), attn_implementation=attention_backend,
            output_loading_info=True, ignore_mismatched_sizes=False,
        )
        problems = {
            key: info[key] for key in (
                "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs",
            ) if info.get(key)
        }
        if problems:
            raise ValueError(f"Checkpoint weights do not strictly match model: {problems}")
        for name, parameter in loaded.named_parameters():
            if not torch.isfinite(parameter).all():
                raise ValueError(f"Non-finite checkpoint parameter: {name}")
        loaded.requires_grad_(False)
        loaded.eval()
        config_file = config_path / "config.json" if config_path.is_dir() else config_path
        tokenizer_files = [
            file for file in (
                tokenizer_path / "tokenizer.json", tokenizer_path / "tokenizer_config.json",
                tokenizer_path / "special_tokens_map.json",
            ) if file.is_file()
        ]
        provenance = {
            "checkpoint_path": str(path),
            "checkpoint_files": [
                {"name": file.name, "size_bytes": file.stat().st_size, "sha256": _sha256(file)}
                for file in files
            ],
            "checkpoint_loading": "strict_local_ModernBertForMaskedLM",
            "config_path": str(config_file), "config_sha256": _sha256(config_file),
            "tokenizer_path": str(tokenizer_path),
            "tokenizer_files": [
                {"name": file.name, "sha256": _sha256(file)} for file in tokenizer_files
            ],
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
        }
        return cls(
            loaded.model, tokenizer, device=device, dtype=dtype,
            attention_backend=attention_backend, max_tokens=max_tokens,
            batch_size=batch_size, pooling=pooling, provenance=provenance,
            sequence_mode=sequence_mode,
        )

    @property
    def metadata(self):
        config = self.model.config
        return {
            **self.provenance,
            "encoder": "ModernBERT_DNA",
            "representation": "backbone_last_hidden_state",
            "embedding_dim": self.embedding_dim,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "vocab_size": config.vocab_size,
            "config_max_position_embeddings": config.max_position_embeddings,
            "rope_parameters": getattr(config, "rope_parameters", None),
            "pooling": self.pooling,
            "normalization": (
                "L2_after_full_sequence_content_mean" if self.sequence_mode == "full"
                else "L2_after_all_windows"
            ),
            "sequence_mode": self.sequence_mode,
            "context_scope": "entire_bgc" if self.sequence_mode == "full" else "per_window",
            "max_tokens": self.max_tokens,
            "max_content_tokens": self.max_content_tokens,
            "max_tokens_policy": (
                "optional_full_input_limit_fail_without_truncation"
                if self.sequence_mode == "full" else "per_window_limit_from_config"
            ),
            "special_token_ids": dict(self.special_ids),
            "special_token_id_corrections": dict(self.special_id_corrections),
            "window_policy": (
                "full_sequence_single_forward" if self.sequence_mode == "full"
                else "full_sequence_BPE_then_disjoint_token_windows"
            ),
            "window_aggregation": (
                "not_applicable_single_forward" if self.sequence_mode == "full"
                else "content_token_count_weighted_mean"
            ),
            "cross_window_attention": False,
            "position_policy": (
                "backbone_positions_across_entire_bgc" if self.sequence_mode == "full"
                else "backbone_positions_restart_in_each_window"
            ),
            "config_length_enforced": self.sequence_mode == "windowed",
            "automatic_window_fallback": False,
            "training_long_sequence_policy": "unknown",
            "unknown_token_policy": "retain_in_pooling_and_report_tokens_and_bases",
            "reverse_complement_policy": "input_orientation_only",
            "frozen": True,
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
            "attention_backend": self.attention_backend,
            "batch_size": self.batch_size,
        }

    def _tokenize(self, sequence):
        if not isinstance(sequence, str) or not sequence:
            raise ValueError("DNA must be a nonempty sequence string")
        invalid = set(sequence) - set("ACGTRYSWKMBDHVN")
        if invalid:
            raise ValueError(f"Unsupported DNA characters: {sorted(invalid)}")
        encoded = self.tokenizer(
            sequence, add_special_tokens=False, truncation=False, padding=False,
            return_attention_mask=False, return_token_type_ids=False,
            return_offsets_mapping=True,
        )
        ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        if not ids or len(ids) != len(offsets):
            raise ValueError("Tokenizer returned empty or inconsistent content tokens")
        prohibited = {self.special_ids[key] for key in ("cls", "sep", "pad")}
        mask_id = getattr(self.tokenizer, "mask_token_id", None)
        if mask_id is not None:
            prohibited.add(mask_id)
        end = 0
        for token_id, (start, stop) in zip(ids, offsets):
            if not isinstance(token_id, int) or not 0 <= token_id < len(self.tokenizer):
                raise ValueError("Tokenizer returned an out-of-vocabulary token ID")
            if token_id in prohibited:
                raise ValueError("Tokenizer inserted a structural special token into content")
            if start != end or not start < stop <= len(sequence):
                raise ValueError("Token offsets do not cover the DNA exactly once")
            end = stop
        if end != len(sequence):
            raise ValueError("Tokenizer dropped the end of the DNA sequence")
        return ids, offsets

    def encode(self, sequence):
        """Return ``(unit_vector[d], sequence_metadata)`` for one entire BGC."""
        ids, offsets = self._tokenize(sequence)
        if self.sequence_mode == "full":
            if self.max_tokens is not None and len(ids) + 2 > self.max_tokens:
                raise DNASequenceTooLongError(len(ids) + 2, self.max_tokens)
            windows = [(0, len(ids))]
        else:
            windows = [
                (start, min(start + self.max_content_tokens, len(ids)))
                for start in range(0, len(ids), self.max_content_tokens)
            ]
        unknown_positions = [i for i, token in enumerate(ids) if token == self.special_ids["unk"]]
        unknown_bases = sum(offsets[i][1] - offsets[i][0] for i in unknown_positions)
        if unknown_positions:
            warnings.warn(
                f"DNA contains {len(unknown_positions)} UNK tokens covering "
                f"{unknown_bases} bases; retained in the representation",
                RuntimeWarning, stacklevel=2,
            )
        total = torch.zeros(self.embedding_dim, dtype=torch.float32)
        forward_shapes = []
        self.model.eval()
        with torch.inference_mode():
            for offset in range(0, len(windows), self.batch_size):
                batch = windows[offset:offset + self.batch_size]
                width = max(end - start + 2 for start, end in batch)
                inputs = torch.full(
                    (len(batch), width), self.special_ids["pad"], dtype=torch.long,
                    device=self.device,
                )
                attention = torch.zeros_like(inputs)
                content = torch.zeros_like(inputs, dtype=torch.bool)
                for row, (start, end) in enumerate(batch):
                    length = end - start
                    inputs[row, :length + 2] = torch.tensor(
                        [self.special_ids["cls"], *ids[start:end], self.special_ids["sep"]],
                        dtype=torch.long, device=self.device,
                    )
                    attention[row, :length + 2] = 1
                    content[row, 1:length + 1] = True
                output = self.model(input_ids=inputs, attention_mask=attention, return_dict=True)
                forward_shapes.append(list(inputs.shape))
                hidden = output.last_hidden_state
                if hidden.shape != (*inputs.shape, self.embedding_dim):
                    raise ValueError("Unexpected ModernBERT hidden-state shape")
                # Boolean selection ensures even NaNs in excluded padding cannot
                # contaminate the content average through multiplication by zero.
                selected = hidden[content].float()
                if not torch.isfinite(selected).all():
                    raise ValueError("Non-finite ModernBERT content hidden states")
                total += selected.sum(dim=0).to(device="cpu", dtype=torch.float32)
        vector = total / len(ids)
        norm = torch.linalg.vector_norm(vector)
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError("DNA mean representation is zero or non-finite")
        vector = vector / norm
        metadata = {
            "sequence_mode": self.sequence_mode,
            "sequence_length": len(sequence),
            "sequence_sha256": sha256(sequence.encode("ascii")).hexdigest(),
            "token_count": len(ids),
            "content_token_count": len(ids),
            "unknown_token_count": len(unknown_positions),
            "unknown_base_count": unknown_bases,
            "ambiguous_base_count": sum(base not in "ACGT" for base in sequence),
            "window_count": len(windows),
            "forward_pass_count": len(forward_shapes),
            "forward_input_shapes": forward_shapes,
            "max_forward_tokens": max(shape[1] for shape in forward_shapes),
            "total_input_tokens": len(ids) + 2 * len(windows),
            "exceeds_config_max_position_embeddings": any(
                shape[1] > int(self.model.config.max_position_embeddings) for shape in forward_shapes
            ),
            "window_content_token_counts": [end - start for start, end in windows],
            "windows": [
                {"token_start": start, "token_end": end,
                 "base_start": offsets[start][0], "base_end": offsets[end - 1][1]}
                for start, end in windows
            ],
        }
        return vector, metadata
