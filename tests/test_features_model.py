import tempfile
import unittest
from pathlib import Path

import torch

from model.encoders import BGCEncoder
from model.features import BGCFeatureExtractor


class FrozenFeatureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "model.pt"
        self.parameters = {
            "esm_size": 4, "gearnet_size": 6, "attention_dim": 8,
            "num_heads": 2, "dropout": 0.8,
        }

    def checkpoint(self, task="classification", **overrides):
        encoder = BGCEncoder(**self.parameters)
        state = {"BGCEncoder." + key: value for key, value in encoder.state_dict().items()}
        # Deliberately not a usable classifier/product head: these are not loaded.
        state["unused_task_head.weight"] = torch.ones(3, 5)
        config = {
            "task": task,
            "data": {"use_structure": 0, "structure_only": 0},
            "model_parameters": self.parameters.copy(),
        }
        checkpoint = {"config": config, "state_dict": [state]}
        checkpoint.update(overrides)
        torch.save(checkpoint, self.path)
        return checkpoint

    def test_baseline_exact_pooling_valid_zero_coordinate_and_nonfinite_padding(self):
        model = BGCFeatureExtractor(esm_size=4)
        values = torch.tensor([[
            [1., 0., 2., 3.], [3., 2., 0., 1.],
            [float("nan"), float("inf"), -99., 17.],
        ]], requires_grad=True)
        vector = model(values, torch.tensor([2]))
        expected = torch.tensor([[2., 1., 1., 2.]])
        expected /= torch.linalg.vector_norm(expected, dim=-1, keepdim=True)
        torch.testing.assert_close(vector, expected)
        self.assertFalse(vector.requires_grad)
        self.assertEqual(model.input_dim, 4)
        self.assertEqual(model.output_dim, 4)
        self.assertEqual(model.metadata["route"], "esm")

    def test_mac_and_map_match_direct_pretrained_encoder(self):
        for route, task in (("mac", "classification"), ("map", "product_matching")):
            with self.subTest(route=route):
                checkpoint = self.checkpoint(task)
                model = BGCFeatureExtractor.from_checkpoint(self.path, route=route)
                reference = BGCEncoder(**self.parameters).eval()
                reference.load_state_dict({
                    key.removeprefix("BGCEncoder."): value
                    for key, value in checkpoint["state_dict"][0].items()
                    if key.startswith("BGCEncoder.")
                })
                values = torch.randn(2, 3, 4)
                lengths = torch.tensor([2, 3])
                mask = torch.arange(3)[None, :] >= lengths[:, None]
                with torch.inference_mode():
                    hidden = reference(values.masked_fill(mask[..., None], 0), mask)
                    expected = hidden.sum(1) / lengths[:, None]
                    expected /= torch.linalg.vector_norm(expected, dim=-1, keepdim=True)
                torch.testing.assert_close(model(values, lengths), expected)
                self.assertEqual(model.metadata["checkpoint_task"], task)
                self.assertEqual(model.output_dim, 8)
                self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))

    def test_encoder_is_deterministic_and_padding_batch_invariant(self):
        self.checkpoint()
        model = BGCFeatureExtractor.from_checkpoint(self.path, route="mac")
        model.train(True)
        self.assertFalse(any(module.training for module in model.modules()))
        short = torch.randn(1, 2, 4, requires_grad=True)
        alone = model(short, torch.tensor([2]))
        batch = torch.randn(2, 7, 4)
        batch[0, :2] = short.detach()[0]
        batch[0, 2:] = float("nan")
        together = model(batch, torch.tensor([2, 7]))
        torch.testing.assert_close(alone[0], together[0], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(together, model(batch, torch.tensor([2, 7])))
        torch.testing.assert_close(together.norm(dim=-1), torch.ones(2))
        self.assertFalse(alone.requires_grad)

    def test_member_selection_does_not_average_weights_or_vectors(self):
        checkpoint = self.checkpoint()
        first = checkpoint["state_dict"][0]
        second = {key: value.clone() for key, value in first.items()}
        second["BGCEncoder.norm2.bias"] += torch.arange(8).float()
        checkpoint["state_dict"].append(second)
        torch.save(checkpoint, self.path)
        one = BGCFeatureExtractor.from_checkpoint(self.path, route="mac", member=0)
        two = BGCFeatureExtractor.from_checkpoint(self.path, route="mac", member=1)
        values = torch.randn(1, 2, 4)
        self.assertFalse(torch.allclose(one(values, torch.tensor([2])), two(values, torch.tensor([2]))))
        self.assertEqual(two.metadata["checkpoint_member"], 1)
        self.assertEqual(two.metadata["checkpoint_members"], 2)
        with self.assertRaisesRegex(ValueError, "out of range"):
            BGCFeatureExtractor.from_checkpoint(self.path, route="mac", member=2)

    def test_rejects_missing_wrong_task_and_structure_checkpoints(self):
        with self.assertRaises(FileNotFoundError):
            BGCFeatureExtractor.from_checkpoint(self.path, route="mac")
        checkpoint = self.checkpoint()
        with self.assertRaisesRegex(ValueError, "requires task"):
            BGCFeatureExtractor.from_checkpoint(self.path, route="map")
        for flag in ("use_structure", "structure_only"):
            checkpoint["config"]["data"] = {flag: True}
            torch.save(checkpoint, self.path)
            with self.assertRaisesRegex(ValueError, flag):
                BGCFeatureExtractor.from_checkpoint(self.path, route="mac")

    def test_strict_loading_rejects_missing_unexpected_or_nonfinite_weights(self):
        for mutation, message in (
            ("missing", "strictly match"),
            ("extra", "strictly match"),
            ("nonfinite", "nonfinite"),
        ):
            with self.subTest(mutation=mutation):
                checkpoint = self.checkpoint()
                state = checkpoint["state_dict"][0]
                if mutation == "missing":
                    del state["BGCEncoder.esm_projection.bias"]
                elif mutation == "extra":
                    state["BGCEncoder.unknown"] = torch.ones(1)
                else:
                    state["BGCEncoder.esm_projection.bias"][0] = float("nan")
                torch.save(checkpoint, self.path)
                with self.assertRaisesRegex(ValueError, message):
                    BGCFeatureExtractor.from_checkpoint(self.path, route="mac")

    def test_rejects_missing_model_parameters(self):
        checkpoint = self.checkpoint()
        del checkpoint["config"]["model_parameters"]["num_heads"]
        torch.save(checkpoint, self.path)
        with self.assertRaisesRegex(ValueError, "Missing checkpoint model"):
            BGCFeatureExtractor.from_checkpoint(self.path, route="mac")

    def test_rejects_invalid_embeddings_and_lengths(self):
        model = BGCFeatureExtractor(esm_size=4)
        values = torch.ones(2, 3, 4)
        for lengths in (
            torch.tensor([0, 3]), torch.tensor([2, 4]),
            torch.tensor([2]), torch.tensor([2., 3.]), torch.tensor([True, True]),
        ):
            with self.subTest(lengths=lengths):
                with self.assertRaises((ValueError, TypeError)):
                    model(values, lengths)
        with self.assertRaisesRegex(ValueError, "last dimension"):
            model(torch.ones(2, 3, 5), torch.tensor([2, 3]))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            model(torch.ones(0, 3, 4), torch.empty(0, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "zero norm"):
            model(torch.zeros(2, 3, 4), torch.tensor([2, 3]))
        values[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            model(values, torch.tensor([2, 3]))

    def test_default_checkpoint_loader_rejects_pickle_objects(self):
        checkpoint = self.checkpoint()
        checkpoint["legacy_extra"] = Path("legacy-object")
        torch.save(checkpoint, self.path)
        with self.assertRaisesRegex(ValueError, "trusted=True"):
            BGCFeatureExtractor.from_checkpoint(self.path, route="mac")
        model = BGCFeatureExtractor.from_checkpoint(
            self.path, route="mac", trusted=True
        )
        self.assertEqual(model.metadata["checkpoint_loading"], "trusted_pickle")


if __name__ == "__main__":
    unittest.main()
