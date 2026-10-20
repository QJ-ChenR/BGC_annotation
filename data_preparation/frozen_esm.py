"""Frozen, local-only ESM2 encoding with complete long-sequence coverage."""

from hashlib import sha256
from pathlib import Path

import torch


def _file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FrozenESM2:
    """Return one CPU float32 vector per original domain/protein.

    Long inputs are split into disjoint windows. Window vectors are averaged
    using residue counts, before any BGC-level pooling, so a long protein does
    not gain extra BGC tokens. BOS is compatible with the author's short-input
    representation; ``mean`` averages residue positions, excluding BOS/EOS.
    """

    def __init__(
        self, model, alphabet, *, device="cpu", pooling="bos",
        max_residues=1021, batch_size=8, checkpoint_path=None,
        checkpoint_sha256=None,
    ):
        if pooling not in ("bos", "mean"):
            raise ValueError("ESM pooling must be 'bos' or 'mean'")
        if not isinstance(max_residues, int) or not 1 <= max_residues <= 1021:
            raise ValueError("max_residues must be an integer between 1 and 1021")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not getattr(alphabet, "prepend_bos", False) or not getattr(
            alphabet, "append_eos", False
        ):
            raise ValueError("Expected an ESM2 alphabet with BOS and EOS")
        self.model = model.to(torch.device(device))
        self.model.requires_grad_(False)
        self.model.eval()
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()
        self.device = torch.device(device)
        self.pooling = pooling
        self.max_residues = max_residues
        self.batch_size = batch_size
        self.layer = int(model.num_layers)
        self.embedding_dim = int(model.embed_dim)
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = checkpoint_sha256

    @classmethod
    def from_checkpoint(
        cls, path, *, device="cpu", pooling="bos", max_residues=1021,
        batch_size=8, trusted=False,
    ):
        """Load a local ESM2 checkpoint; never fetch weights or regression data.

        Official ESM checkpoints often contain configuration objects unsupported
        by PyTorch's restricted loader. Such files need explicit ``trusted=True``
        after the caller has verified their source.
        """
        import esm

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Local ESM2 checkpoint not found: {path}")
        try:
            model_data = torch.load(path, map_location="cpu", weights_only=not trusted)
        except Exception as exc:
            if not trusted:
                raise ValueError(
                    "Restricted ESM checkpoint loading failed. Only for a verified "
                    "checkpoint, pass --trusted-input to allow its configuration "
                    f"objects. Original error: {exc}"
                ) from exc
            raise
        if not isinstance(model_data, dict) or not {"cfg", "model"}.issubset(model_data):
            raise ValueError("Expected an ESM2 checkpoint containing 'cfg' and 'model'")
        if not isinstance(model_data["model"], dict):
            raise ValueError("ESM2 checkpoint 'model' must be a state dictionary")
        for key, value in model_data["model"].items():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError(f"Non-finite ESM2 checkpoint tensor: {key}")
        # The core loader has no network access. Contact-regression weights are
        # unnecessary for representations and deliberately not loaded.
        model, alphabet = esm.pretrained.load_model_and_alphabet_core(
            "esm2_local", model_data, regression_data=None
        )
        return cls(
            model, alphabet, device=device, pooling=pooling,
            max_residues=max_residues, batch_size=batch_size,
            checkpoint_path=str(path.resolve()),
            checkpoint_sha256=_file_sha256(path),
        )

    @property
    def metadata(self):
        return {
            "encoder": "ESM2",
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "layer": self.layer,
            "pooling": self.pooling,
            "embedding_dim": self.embedding_dim,
            "max_residues": self.max_residues,
            "window_policy": "disjoint_full_coverage",
            "window_aggregation": "residue_count_weighted_mean_per_original_token",
            "special_tokens": "BOS_and_EOS_preserved",
            "frozen": True,
            "device": str(self.device),
        }

    def _windows(self, sequence):
        if not isinstance(sequence, str) or not sequence:
            raise ValueError("Every token must contain a nonempty sequence string")
        invalid = set(sequence) - set("ACDEFGHIKLMNPQRSTVWYBXZUO")
        if invalid:
            raise ValueError(f"Unsupported sequence characters: {sorted(invalid)}")
        return [
            (start, min(start + self.max_residues, len(sequence)))
            for start in range(0, len(sequence), self.max_residues)
        ]

    def token_metadata(self, tokens):
        result = []
        for token in tokens:
            sequence = token["sequence"]
            windows = self._windows(sequence)
            result.append({
                **{key: value for key, value in token.items() if key != "sequence"},
                "length": len(sequence),
                "sequence_sha256": sha256(sequence.encode("ascii")).hexdigest(),
                "window_count": len(windows),
                "window_lengths": [end - start for start, end in windows],
                "windows": [[start, end] for start, end in windows],
            })
        return result

    def encode(self, tokens):
        """Encode token dictionaries, returning [N, embedding_dim] on CPU."""
        tokens = list(tokens)
        if not tokens:
            raise ValueError("Cannot encode an empty BGC")
        windows = []
        lengths = []
        for token_index, token in enumerate(tokens):
            sequence = token["sequence"]
            lengths.append(len(sequence))
            for start, end in self._windows(sequence):
                windows.append((token_index, sequence[start:end]))
        result = torch.zeros(len(tokens), self.embedding_dim, dtype=torch.float32)
        self.model.eval()
        with torch.inference_mode():
            for offset in range(0, len(windows), self.batch_size):
                batch = windows[offset:offset + self.batch_size]
                _, strings, batch_tokens = self.batch_converter([
                    (str(index), sequence) for index, sequence in batch
                ])
                expected_lengths = torch.tensor([len(s) + 2 for s in strings])
                actual_lengths = (
                    batch_tokens != self.alphabet.padding_idx
                ).sum(dim=1).cpu()
                if not torch.equal(expected_lengths, actual_lengths):
                    raise ValueError("ESM tokenization changed sequence lengths")
                representations = self.model(
                    batch_tokens.to(self.device), repr_layers=[self.layer],
                    return_contacts=False,
                )["representations"][self.layer]
                if representations.shape[:2] != batch_tokens.shape or (
                    representations.shape[-1] != self.embedding_dim
                ):
                    raise ValueError("Unexpected ESM representation shape")
                for batch_index, (token_index, sequence) in enumerate(batch):
                    if self.pooling == "bos":
                        vector = representations[batch_index, 0]
                    else:
                        vector = representations[
                            batch_index, 1:len(sequence) + 1
                        ].float().mean(dim=0)
                    vector = vector.detach().to(device="cpu", dtype=torch.float32)
                    if not torch.isfinite(vector).all():
                        raise ValueError(f"Non-finite ESM vector for token {token_index}")
                    result[token_index] += vector * (len(sequence) / lengths[token_index])
        if not torch.isfinite(result).all():
            raise ValueError("Non-finite aggregated ESM vectors")
        return result
