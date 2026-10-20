import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from data_preparation.frozen_esm import FrozenESM2


class FakeAlphabet:
    padding_idx = 0
    prepend_bos = True
    append_eos = True

    def get_batch_converter(self):
        def convert(data):
            labels, strings = zip(*data)
            tokens = torch.zeros(len(data), max(map(len, strings)) + 2, dtype=torch.long)
            for row, sequence in enumerate(strings):
                tokens[row, 0] = 100
                tokens[row, 1:len(sequence) + 1] = torch.tensor(
                    [ord(letter) - ord("A") + 1 for letter in sequence]
                )
                tokens[row, len(sequence) + 1] = 101
            return labels, strings, tokens
        return convert


class FakeESM(torch.nn.Module):
    num_layers = 33
    embed_dim = 4

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.seen = []

    def forward(self, tokens, repr_layers, return_contacts):
        assert not torch.is_grad_enabled()
        assert not self.training
        assert repr_layers == [33]
        assert not return_contacts
        self.seen.extend(tokens.detach().cpu().tolist())
        vectors = tokens.float().unsqueeze(-1).expand(-1, -1, 4).clone()
        # Context-dependent BOS vector makes window weighting observable.
        for row in range(len(tokens)):
            residues = tokens[row][(tokens[row] > 0) & (tokens[row] < 100)].float()
            vectors[row, 0] = torch.tensor([
                residues.mean(), residues.sum(), len(residues), residues[0]
            ], device=tokens.device)
        return {"representations": {33: vectors}}


class FrozenESMTests(unittest.TestCase):
    def encoder(self, **kwargs):
        self.model = FakeESM()
        return FrozenESM2(self.model, FakeAlphabet(), **kwargs)

    def test_short_bos_matches_direct_final_layer_and_is_frozen(self):
        encoder = self.encoder()
        result = encoder.encode([{"sequence": "ACD"}])
        torch.testing.assert_close(result, torch.tensor([[8 / 3, 8, 3, 1]]))
        self.assertEqual(result.device.type, "cpu")
        self.assertEqual(result.dtype, torch.float32)
        self.assertFalse(result.requires_grad)
        self.assertTrue(all(not p.requires_grad for p in self.model.parameters()))

    def test_long_sequence_complete_coverage_and_returns_one_token(self):
        encoder = self.encoder(max_residues=3, batch_size=2)
        tokens = [{"sequence": "ACDEFGH", "token_id": "long"}]
        result = encoder.encode(tokens)
        first = torch.tensor([8 / 3, 8, 3, 1])
        second = torch.tensor([6, 18, 3, 5])
        third = torch.tensor([8, 8, 1, 8])
        expected = ((first * 3 + second * 3 + third) / 7).unsqueeze(0)
        torch.testing.assert_close(result, expected)
        self.assertEqual(result.shape, (1, 4))
        seen = [
            token for row in self.model.seen for token in row
            if token not in (0, 100, 101)
        ]
        self.assertEqual(seen, [1, 3, 4, 5, 6, 7, 8])
        metadata = encoder.token_metadata(tokens)[0]
        self.assertNotIn("sequence", metadata)
        self.assertEqual(metadata["window_lengths"], [3, 3, 1])
        self.assertEqual(metadata["windows"], [[0, 3], [3, 6], [6, 7]])

    def test_residue_mean_excludes_special_tokens_and_padding(self):
        encoder = self.encoder(pooling="mean", max_residues=3)
        result = encoder.encode([{"sequence": "ACDEFGH"}, {"sequence": "A"}])
        torch.testing.assert_close(result[0], torch.full((4,), 34 / 7))
        torch.testing.assert_close(result[1], torch.ones(4))

    def test_batch_size_does_not_change_vectors(self):
        tokens = [{"sequence": "ACDEFGH"}, {"sequence": "WY"}, {"sequence": "A"}]
        first = self.encoder(batch_size=1, max_residues=3).encode(tokens)
        second = self.encoder(batch_size=7, max_residues=3).encode(tokens)
        torch.testing.assert_close(first, second)

    def test_eval_restored_before_encoding(self):
        encoder = self.encoder()
        self.model.train()
        encoder.encode([{"sequence": "AC"}])
        self.assertFalse(self.model.training)

    def test_nonfinite_vectors_fail(self):
        encoder = self.encoder()
        with patch.object(
            self.model, "forward",
            return_value={"representations": {33: torch.full((1, 3, 4), float("nan"))}},
        ):
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                encoder.encode([{"sequence": "A"}])

    def test_invalid_inputs_and_parameters_fail(self):
        for kwargs in ({"max_residues": 1022}, {"batch_size": 0}, {"pooling": "cls"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.encoder(**kwargs)
        encoder = self.encoder()
        for tokens in ([], [{"sequence": ""}], [{"sequence": "A*"}], [{"sequence": "AJ"}]):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                encoder.encode(tokens)

    def test_checkpoint_loader_uses_restricted_loading_and_core_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            torch.save({"cfg": {}, "model": {}}, path)
            fake_esm = SimpleNamespace(pretrained=SimpleNamespace(
                load_model_and_alphabet_core=lambda *args, **kwargs: (FakeESM(), FakeAlphabet())
            ))
            original_load = torch.load
            with patch.dict("sys.modules", {"esm": fake_esm}), patch(
                "data_preparation.frozen_esm.torch.load", wraps=original_load
            ) as load:
                encoder = FrozenESM2.from_checkpoint(path)
                self.assertTrue(load.call_args.kwargs["weights_only"])
                self.assertEqual(len(encoder.metadata["checkpoint_sha256"]), 64)
            with patch.dict("sys.modules", {"esm": fake_esm}), patch(
                "data_preparation.frozen_esm.torch.load", wraps=original_load
            ) as load:
                FrozenESM2.from_checkpoint(path, trusted=True)
                self.assertFalse(load.call_args.kwargs["weights_only"])

    def test_real_esm2_local_checkpoint_namespace_and_direct_bos(self):
        import esm

        alphabet = esm.Alphabet.from_architecture("ESM-1b")
        model = esm.model.esm2.ESM2(
            num_layers=2, embed_dim=32, attention_heads=4,
            alphabet=alphabet, token_dropout=False,
        ).eval()
        checkpoint = {
            "cfg": {"model": Namespace(
                encoder_layers=2, encoder_embed_dim=32,
                encoder_attention_heads=4, token_dropout=False,
            )},
            "model": model.state_dict(),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny_esm2.pt"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "--trusted-input"):
                FrozenESM2.from_checkpoint(path)
            encoder = FrozenESM2.from_checkpoint(path, trusted=True)
            result = encoder.encode([{"sequence": "ACDEFGHIK"}])
            _, _, tokens = alphabet.get_batch_converter()([("test", "ACDEFGHIK")])
            with torch.inference_mode():
                expected = model(tokens, repr_layers=[2])["representations"][2][:, 0]
            torch.testing.assert_close(result, expected)
            self.assertEqual(result.shape, (1, 32))
            self.assertEqual(encoder.metadata["layer"], 2)
            checkpoint["model"].pop("embed_tokens.weight")
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                FrozenESM2.from_checkpoint(path, trusted=True)
            checkpoint["model"] = model.state_dict()
            checkpoint["model"]["embed_tokens.weight"][0, 0] = float("nan")
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "Non-finite ESM2 checkpoint tensor"):
                FrozenESM2.from_checkpoint(path, trusted=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_device_independent_windowing(self):
        tokens = [{"sequence": "ACDEFGH"}]
        cpu = self.encoder(device="cpu", max_residues=3).encode(tokens)
        gpu = self.encoder(device="cuda", max_residues=3).encode(tokens)
        torch.testing.assert_close(cpu, gpu)


if __name__ == "__main__":
    unittest.main()
