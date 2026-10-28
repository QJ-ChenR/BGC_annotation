import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from data_preparation.frozen_dna import DNASequenceTooLongError, FrozenDNA, _weight_files


def fake_config(**updates):
    values = dict(
        model_type="modernbert", hidden_size=2, vocab_size=16,
        max_position_embeddings=8, pad_token_id=3, cls_token_id=50281,
        bos_token_id=50281, sep_token_id=50282, eos_token_id=50282,
        num_hidden_layers=2, num_attention_heads=1,
    )
    return SimpleNamespace(**{**values, **updates})


class FakeTokenizer:
    is_fast = True
    cls_token_id = 1
    sep_token_id = 2
    pad_token_id = 3
    mask_token_id = 4
    unk_token_id = 0

    def __len__(self):
        return 16

    def __call__(self, sequence, **kwargs):
        assert kwargs["add_special_tokens"] is False
        assert kwargs["truncation"] is False
        assert kwargs["padding"] is False
        assert kwargs["return_offsets_mapping"] is True
        ids, offsets = [], []
        pos = 0
        while pos < len(sequence):
            if sequence[pos:pos + 2] == "AA":
                ids.append(9)
                end = pos + 2
            else:
                ids.append({"A": 5, "C": 6, "G": 7, "T": 8}.get(sequence[pos], 0))
                end = pos + 1
            offsets.append((pos, end))
            pos = end
        return {"input_ids": ids, "offset_mapping": offsets}


class FakeBackbone(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or fake_config()
        self.embeddings = torch.nn.Embedding(self.config.vocab_size, 2, padding_idx=3)
        self.seen = []

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, input_ids, attention_mask, return_dict):
        assert torch.is_inference_mode_enabled()
        assert not torch.is_grad_enabled()
        assert not self.training
        assert return_dict is True
        assert torch.equal(attention_mask.bool(), input_ids != 3)
        self.seen.append((input_ids.detach().cpu(), attention_mask.detach().cpu()))
        hidden = torch.stack((input_ids.float(), torch.ones_like(input_ids).float()), dim=-1)
        # Poison specials and padding so accidental inclusion is observable.
        hidden[(input_ids == 1) | (input_ids == 2)] = 10000
        hidden[input_ids == 3] = float("nan")
        return SimpleNamespace(last_hidden_state=hidden)


class FakeMLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeBackbone()
        self.head = torch.nn.Linear(2, 16)


class FrozenDNATests(unittest.TestCase):
    def encoder(self, **kwargs):
        self.model = FakeBackbone()
        return FrozenDNA(self.model, FakeTokenizer(), **kwargs)

    @staticmethod
    def expected(values):
        vector = torch.tensor([sum(values) / len(values), 1.0])
        return vector / torch.linalg.vector_norm(vector)

    def test_content_mean_excludes_cls_sep_and_returns_frozen_unit_vector(self):
        encoder = self.encoder()
        vector, metadata = encoder.encode("ACGT")
        torch.testing.assert_close(vector, self.expected([5, 6, 7, 8]))
        self.assertEqual(vector.shape, (2,))
        self.assertEqual(vector.dtype, torch.float32)
        self.assertEqual(vector.device.type, "cpu")
        self.assertFalse(vector.requires_grad)
        self.assertTrue(all(not p.requires_grad for p in self.model.parameters()))
        self.assertEqual(metadata["content_token_count"], 4)
        self.assertEqual(metadata["sequence_length"], 4)
        self.assertEqual(metadata["window_content_token_counts"], [4])

    def test_long_tail_coverage_weighting_and_padding(self):
        encoder = self.encoder(max_tokens=5, batch_size=3)
        vector, metadata = encoder.encode("ACGTACG")
        torch.testing.assert_close(vector, self.expected([5, 6, 7, 8, 5, 6, 7]))
        self.assertEqual(metadata["window_content_token_counts"], [3, 3, 1])
        self.assertEqual(metadata["windows"][-1], {
            "token_start": 6, "token_end": 7, "base_start": 6, "base_end": 7,
        })
        ids, mask = self.model.seen[0]
        self.assertEqual(ids.tolist(), [[1, 5, 6, 7, 2], [1, 8, 5, 6, 2], [1, 7, 2, 3, 3]])
        self.assertEqual(mask[-1].tolist(), [1, 1, 1, 0, 0])

    def test_full_sequence_ignores_config_length_without_changing_it(self):
        encoder = self.encoder(sequence_mode="full")
        before = {name: value.detach().clone() for name, value in self.model.named_parameters()}
        vector, metadata = encoder.encode("ACGTACGTACG")
        torch.testing.assert_close(vector, self.expected([5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7]))
        self.assertEqual(len(self.model.seen), 1)
        self.assertEqual(self.model.seen[0][0].tolist(), [[1, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 2]])
        self.assertEqual(metadata["window_count"], 1)
        self.assertEqual(metadata["forward_pass_count"], 1)
        self.assertEqual(metadata["forward_input_shapes"], [[1, 13]])
        self.assertEqual(metadata["total_input_tokens"], 13)
        self.assertEqual(metadata["max_forward_tokens"], 13)
        self.assertTrue(metadata["exceeds_config_max_position_embeddings"])
        self.assertEqual(metadata["windows"], [{
            "token_start": 0, "token_end": 11, "base_start": 0, "base_end": 11,
        }])
        self.assertEqual(self.model.config.max_position_embeddings, 8)
        self.assertIsNone(encoder.metadata["max_tokens"])
        self.assertEqual(encoder.metadata["context_scope"], "entire_bgc")
        self.assertFalse(encoder.metadata["config_length_enforced"])
        self.assertFalse(encoder.metadata["automatic_window_fallback"])
        for name, parameter in self.model.named_parameters():
            torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)

    def test_full_explicit_limit_counts_specials_and_fails_before_forward(self):
        encoder = self.encoder(sequence_mode="full", max_tokens=10)
        encoder.encode("ACGTACGT")
        self.assertEqual(len(self.model.seen), 1)
        with self.assertRaises(DNASequenceTooLongError) as context:
            encoder.encode("ACGTACGTA")
        self.assertEqual(context.exception.input_tokens, 11)
        self.assertEqual(context.exception.limit, 10)
        self.assertEqual(len(self.model.seen), 1)

    def test_full_oom_propagates_without_chunk_fallback(self):
        encoder = self.encoder(sequence_mode="full")
        with patch.object(self.model, "forward", side_effect=torch.cuda.OutOfMemoryError("test OOM")) as forward:
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                encoder.encode("ACGTACGTACG")
        self.assertEqual(forward.call_count, 1)

    def test_bpe_pooling_weights_tokens_not_nucleotides(self):
        encoder = self.encoder(max_tokens=3)
        vector, metadata = encoder.encode("AAC")
        torch.testing.assert_close(vector, self.expected([9, 6]))
        self.assertEqual(metadata["content_token_count"], 2)
        self.assertEqual(metadata["windows"][0]["base_end"], 2)
        self.assertEqual(metadata["windows"][1]["base_start"], 2)

    def test_unknown_tokens_are_retained_and_coverage_reported(self):
        encoder = self.encoder()
        with self.assertWarnsRegex(RuntimeWarning, "2 UNK tokens covering 2 bases"):
            vector, metadata = encoder.encode("ANRC")
        torch.testing.assert_close(vector, self.expected([5, 0, 0, 6]))
        self.assertEqual(metadata["unknown_token_count"], 2)
        self.assertEqual(metadata["unknown_base_count"], 2)
        self.assertEqual(metadata["ambiguous_base_count"], 2)

    def test_batch_size_does_not_change_representation(self):
        sequence = "ACGTACGTACG"
        a, _ = self.encoder(max_tokens=5, batch_size=1).encode(sequence)
        b, _ = self.encoder(max_tokens=5, batch_size=4).encode(sequence)
        torch.testing.assert_close(a, b)

    def test_eval_is_restored_before_each_encoding(self):
        encoder = self.encoder()
        self.model.train()
        encoder.encode("AC")
        self.assertFalse(self.model.training)

    def test_special_ids_corrected_without_resizing_and_recorded(self):
        encoder = self.encoder()
        self.assertEqual(self.model.get_input_embeddings().weight.shape, (16, 2))
        self.assertEqual(self.model.config.cls_token_id, 1)
        self.assertEqual(self.model.config.sep_token_id, 2)
        metadata = encoder.metadata
        self.assertEqual(metadata["special_token_id_corrections"]["bos_token_id"], {
            "config_value": 50281, "tokenizer_value": 1,
        })
        self.assertFalse(metadata["cross_window_attention"])
        self.assertEqual(metadata["training_long_sequence_policy"], "unknown")

    def test_padding_or_vocabulary_mismatch_is_fatal(self):
        with self.assertRaisesRegex(ValueError, "PAD ID"):
            FrozenDNA(FakeBackbone(fake_config(pad_token_id=0)), FakeTokenizer())
        with self.assertRaisesRegex(ValueError, "vocabulary size"):
            FrozenDNA(FakeBackbone(fake_config(vocab_size=17)), FakeTokenizer())

    def test_invalid_parameters_and_sequences_fail(self):
        for kwargs in (
            {"pooling": "cls"}, {"max_tokens": 2}, {"max_tokens": 9},
            {"batch_size": 0}, {"dtype": "auto"}, {"attention_backend": "invalid"},
            {"sequence_mode": "auto"}, {"sequence_mode": "full", "batch_size": 2},
            {"sequence_mode": "full", "max_tokens": True},
            {"attention_backend": "flash_attention_2", "dtype": "float32"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.encoder(**kwargs)
        encoder = self.encoder()
        for sequence in ("", "acgt", "A C", "ACU", "A*", None):
            with self.subTest(sequence=sequence), self.assertRaises(ValueError):
                encoder.encode(sequence)

    def test_truncated_or_overlapping_token_offsets_fail(self):
        encoder = self.encoder()
        for encoded in (
            {"input_ids": [5], "offset_mapping": [(0, 1)]},
            {"input_ids": [5, 6], "offset_mapping": [(0, 2), (1, 2)]},
            {"input_ids": [1, 6], "offset_mapping": [(0, 1), (1, 2)]},
        ):
            with self.subTest(encoded=encoded), patch.object(
                FakeTokenizer, "__call__", return_value=encoded,
            ), self.assertRaises(ValueError):
                encoder.encode("AC")

    def test_nonfinite_or_zero_representation_fails(self):
        encoder = self.encoder()
        for value in (float("nan"), 0.0):
            with patch.object(self.model, "forward", return_value=SimpleNamespace(
                last_hidden_state=torch.full((1, 3, 2), value),
            )), self.assertRaisesRegex(ValueError, "[Nn]on-finite|zero"):
                encoder.encode("A")

    def make_checkpoint(self, directory):
        path = Path(directory)
        (path / "model.safetensors").write_bytes(b"fake: loader is mocked")
        (path / "config.json").write_text("{}")
        (path / "tokenizer.json").write_text("{}")
        return path

    def fake_transformers(self, info=None):
        self.mlm = FakeMLM()
        self.loader = Mock(return_value=(self.mlm, info or {}))
        return SimpleNamespace(
            __version__="5.1.0",
            AutoConfig=SimpleNamespace(from_pretrained=Mock(return_value=self.mlm.model.config)),
            AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=FakeTokenizer())),
            ModernBertForMaskedLM=SimpleNamespace(from_pretrained=self.loader),
        )

    def test_loader_is_local_strict_and_keeps_backbone_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_checkpoint(temporary)
            module = self.fake_transformers()
            with patch.dict("sys.modules", {"transformers": module}):
                encoder = FrozenDNA.from_checkpoint(path)
            self.assertIs(encoder.model, self.mlm.model)
            self.assertTrue(self.loader.call_args.kwargs["local_files_only"])
            self.assertFalse(self.loader.call_args.kwargs["ignore_mismatched_sizes"])
            self.assertTrue(self.loader.call_args.kwargs["output_loading_info"])
            self.assertEqual(self.loader.call_args.kwargs["attn_implementation"], "sdpa")
            self.assertEqual(len(encoder.metadata["checkpoint_files"][0]["sha256"]), 64)
            self.assertFalse(self.mlm.head.weight.requires_grad)

    def test_missing_unexpected_mismatched_and_error_keys_are_fatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_checkpoint(temporary)
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
                module = self.fake_transformers({key: ["model.layer.weight"]})
                with self.subTest(key=key), patch.dict(
                    "sys.modules", {"transformers": module},
                ), self.assertRaisesRegex(ValueError, "strictly match"):
                    FrozenDNA.from_checkpoint(path)

    def test_nonfinite_checkpoint_is_fatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_checkpoint(temporary)
            module = self.fake_transformers()
            with torch.no_grad():
                self.mlm.head.weight[0, 0] = float("nan")
            with patch.dict("sys.modules", {"transformers": module}), self.assertRaisesRegex(
                ValueError, "Non-finite checkpoint",
            ):
                FrozenDNA.from_checkpoint(path)

    def test_config_only_checkpoint_cannot_create_random_encoder(self):
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "config.json").write_text("{}")
            with self.assertRaisesRegex(FileNotFoundError, "config alone is insufficient"):
                FrozenDNA.from_checkpoint(temporary)

    def test_sharded_checkpoint_requires_all_local_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            index = path / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"weight": "part.safetensors"}}))
            with self.assertRaisesRegex(FileNotFoundError, "Missing checkpoint shard"):
                _weight_files(path)
            (path / "part.safetensors").write_bytes(b"fake")
            self.assertEqual(len(_weight_files(path)), 2)
            index.write_text(json.dumps({"weight_map": {"weight": "../outside.safetensors"}}))
            with self.assertRaisesRegex(ValueError, "inside checkpoint directory"):
                _weight_files(path)


if __name__ == "__main__":
    unittest.main()
