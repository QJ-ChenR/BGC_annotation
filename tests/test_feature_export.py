"""Behavioral checks of cache input and the executable export workflow."""

import csv
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from data_preparation.feature_cache import CACHE_FORMAT, load_feature_cache
from experiment.extract_features import build_parser, run


class FeatureExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def args(self, cache, name="features", *extra):
        return build_parser().parse_args([
            "--esm-cache", str(cache), "--output", str(self.root / name), *extra,
        ])

    def cache(self, mapping, name="cache.pth"):
        path = self.root / name
        torch.save(mapping, path)
        return path

    def test_legacy_baseline_values_id_alignment_and_unknown_provenance(self):
        cache = self.cache({
            "BGC_B": [torch.tensor([0., 3., 0.])],
            "BGC_A": [torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.])],
        })
        args = self.args(cache)
        metadata = run(args)
        actual = np.load(args.output / "embeddings.npy", allow_pickle=False)
        expected = np.array([[1 / np.sqrt(2), 1 / np.sqrt(2), 0], [0, 1, 0]])
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        with (args.output / "bgc_ids.tsv").open() as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual([row["bgc_id"] for row in rows], ["BGC_A", "BGC_B"])
        self.assertEqual([row["token_count"] for row in rows], ["2", "1"])
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(metadata["status"], "complete")
        self.assertEqual(metadata["input"]["esm_pooling"], "unknown")
        self.assertEqual(len(metadata["input"]["cache_sha256"]), 64)

    def test_batch_size_does_not_change_features(self):
        cache = self.cache({"a": torch.randn(5, 7), "b": torch.randn(1, 7)})
        one = self.args(cache, "one", "--batch-size", "1")
        two = self.args(cache, "two", "--batch-size", "2")
        run(one)
        run(two)
        np.testing.assert_allclose(np.load(one.output / "embeddings.npy"),
                                   np.load(two.output / "embeddings.npy"), atol=1e-7)

    def test_invalid_entries_stop_and_leave_reviewable_report(self):
        cache = self.cache({"bad": torch.empty(0, 3), "good": torch.ones(2, 3)})
        args = self.args(cache)
        with self.assertRaisesRegex(ValueError, "Invalid inputs"):
            run(args)
        self.assertFalse((args.output / "embeddings.npy").exists())
        self.assertIn("bad", (args.output / "failures.tsv").read_text())
        self.assertEqual(json.loads((args.output / "metadata.json").read_text())["status"], "failed")

    def test_skip_invalid_is_explicit_and_identifies_zero_norm(self):
        cache = self.cache({
            "cancelled": torch.tensor([[1., 0.], [-1., 0.]]),
            "empty": torch.empty(0, 2),
            "valid": torch.tensor([[0., 1.]]),
        })
        args = self.args(cache, "features", "--skip-invalid")
        result = run(args)
        self.assertEqual(result["excluded_count"], 2)
        self.assertEqual(result["count"], 1)
        np.testing.assert_array_equal(np.load(args.output / "embeddings.npy"), [[0, 1]])
        failures = (args.output / "failures.tsv").read_text()
        self.assertIn("cancelled", failures)
        self.assertIn("empty", failures)

    def test_versioned_cache_retains_metadata(self):
        cache = self.cache({
            "format": CACHE_FORMAT,
            "embeddings": {"x": torch.ones(2, 3)},
            "records": {"x": {"source_sha256": "abc", "tokens": [{"source": "domain"}]}},
            "provenance": {"esm": {"pooling": "bos", "max_residues": 1021}},
        })
        result = run(self.args(cache))
        self.assertEqual(result["records"]["x"]["source_sha256"], "abc")
        self.assertEqual(result["input"]["esm"]["pooling"], "bos")

    def test_cache_reuse_retains_upstream_exclusions(self):
        cache = self.cache({
            "format": CACHE_FORMAT,
            "embeddings": {"good": torch.ones(2, 3)},
            "records": {},
            "provenance": {"input_count": 2, "exclusions": [{
                "bgc_id": "bad", "source": "bad.gbk",
                "stage": "gbk_to_esm", "error": "Missing translation",
            }]},
        })
        args = self.args(cache)
        result = run(args)
        self.assertEqual(result["excluded_count"], 0)
        self.assertEqual(result["upstream_excluded_count"], 1)
        self.assertEqual(result["status"], "complete_with_exclusions")
        report = (args.output / "failures.tsv").read_text()
        self.assertIn("bad.gbk", report)
        self.assertIn("upstream/gbk_to_esm", report)

    def test_save_cache_rejects_pickle_suffix_before_work(self):
        args = build_parser().parse_args([
            "--gbk", str(self.root / "example.gbk"),
            "--esm-checkpoint", str(self.root / "esm.pt"),
            "--save-cache", str(self.root / "bad.pkl"),
            "--output", str(self.root / "features"),
        ])
        with self.assertRaisesRegex(ValueError, "must use .pt or .pth"):
            run(args)
        self.assertFalse(args.output.exists())

    def test_pickle_requires_explicit_trust(self):
        path = self.root / "legacy.pkl"
        with path.open("wb") as handle:
            pickle.dump({"a": [torch.ones(2)]}, handle)
        with self.assertRaisesRegex(ValueError, "trusted-input"):
            load_feature_cache(path)
        values, _, _ = load_feature_cache(path, trusted=True)
        self.assertEqual(list(values), ["a"])

    def test_output_cannot_overwrite_existing_results(self):
        cache = self.cache({"a": torch.ones(2, 3)})
        args = self.args(cache)
        run(args)
        before = (args.output / "embeddings.npy").read_bytes()
        with self.assertRaisesRegex(ValueError, "already exist"):
            run(args)
        self.assertEqual((args.output / "embeddings.npy").read_bytes(), before)

    def test_checkpoint_required_and_cached_pooling_cannot_be_relabelled(self):
        cache = self.cache({"a": torch.ones(2, 3)})
        with self.assertRaisesRegex(ValueError, "requires --checkpoint"):
            run(self.args(cache, "mac", "--route", "mac"))
        with self.assertRaisesRegex(ValueError, "cannot be changed"):
            run(self.args(cache, "mean", "--esm-pooling", "mean"))

    def test_mac_map_exports_load_encoder_only_and_select_member(self):
        from model.BGC_models import BGCClassifier, ProductMatching

        parameters = {"esm_size": 6, "gearnet_size": 4, "attention_dim": 8,
                      "num_heads": 2, "dropout": 0.8}
        for route, cls, task in (
            ("mac", BGCClassifier, "classification"),
            ("map", ProductMatching, "product_matching"),
        ):
            with self.subTest(route=route):
                models = [cls(**parameters), cls(**parameters)]
                checkpoint = self.cache({
                    "state_dict": [model.state_dict() for model in models],
                    "config": {"task": task, "model_parameters": parameters,
                               "data": {"use_structure": 0, "structure_only": 0}},
                }, f"{route}.ckpt")
                cache = self.cache({"a": torch.randn(3, 6), "b": torch.randn(1, 6)},
                                   f"{route}_inputs.pth")
                args = self.args(cache, route, "--route", route, "--checkpoint",
                                 str(checkpoint), "--member", "1")
                metadata = run(args)
                actual = np.load(args.output / "embeddings.npy")
                self.assertEqual(actual.shape, (2, 8))
                np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1, atol=1e-6)
                self.assertEqual(metadata["extractor"]["checkpoint_member"], 1)

    def test_raw_input_and_saved_cache_round_trip(self):
        from Bio import SeqIO
        from Bio.Seq import Seq
        from Bio.SeqFeature import FeatureLocation, SeqFeature
        from Bio.SeqRecord import SeqRecord

        gbk = self.root / "example.gbk"
        record = SeqRecord(Seq("ATG" * 3), id="example",
                           annotations={"molecule_type": "DNA"})
        record.features = [SeqFeature(
            FeatureLocation(0, 9, strand=1), type="CDS",
            qualifiers={"locus_tag": ["g1"], "translation": ["MMM"]},
        )]
        SeqIO.write(record, gbk, "genbank")
        cache = self.root / "new.pth"
        args = build_parser().parse_args([
            "--gbk", str(gbk), "--esm-checkpoint", str(self.root / "unused.pt"),
            "--save-cache", str(cache), "--output", str(self.root / "raw"),
        ])

        class StubESM:
            metadata = {"pooling": "bos", "max_residues": 1021}

            def encode(self, tokens):
                return torch.tensor([[1., 2., 3.] for _ in tokens])

            def token_metadata(self, tokens):
                return [{key: value for key, value in token.items() if key != "sequence"}
                        for token in tokens]

        with patch("data_preparation.frozen_esm.FrozenESM2.from_checkpoint",
                   return_value=StubESM()):
            result = run(args)
        cached = run(self.args(cache, "cached"))
        self.assertEqual(result["count"], cached["count"])
        self.assertEqual(cached["input"]["parser"], "gbk_ordered_domains_v1")
        np.testing.assert_array_equal(np.load(args.output / "embeddings.npy"),
                                      np.load(self.root / "cached" / "embeddings.npy"))


if __name__ == "__main__":
    unittest.main()
