"""Validation CLI integration tests using a tiny locally generated checkpoint."""

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiment.validate_dna_encoding import (
    _measure, _select_samples, build_parser, main, parameter_sha256, run,
)


class ValidationHelperTests(unittest.TestCase):
    def test_length_sample_is_deterministic_and_preserves_extremes(self):
        records = [{"bgc_id": str(index), "token_count": 10 - index} for index in range(10)]
        chosen = _select_samples(records, 3, False)
        self.assertEqual([record["token_count"] for record in chosen], [1, 5, 10])
        self.assertEqual(len(_select_samples(records, 3, True)), 10)

    def test_parameter_hash_handles_bfloat16_and_detects_mutation(self):
        model = torch.nn.Linear(2, 2).to(dtype=torch.bfloat16)
        before = parameter_sha256(model)
        self.assertEqual(before, parameter_sha256(model))
        with torch.no_grad():
            model.weight[0, 0] += 1
        self.assertNotEqual(before, parameter_sha256(model))

    def test_oom_is_recorded_without_fallback(self):
        for error in (torch.OutOfMemoryError("synthetic OOM"), MemoryError("synthetic OOM"),
                      RuntimeError("CUDA out of memory: synthetic OOM")):
            with self.subTest(error=type(error).__name__):
                def fail():
                    raise error
                value, result = _measure(torch.device("cpu"), fail)
                self.assertIsNone(value)
                self.assertEqual(result["status"], "oom")
                self.assertIn("synthetic OOM", result["error"])
                self.assertIsNone(result["peak_allocated_bytes"])

    def test_async_cuda_error_invalidates_already_returned_value(self):
        with patch("torch.cuda.synchronize", side_effect=[None, RuntimeError("CUDA out of memory")]), \
             patch("torch.cuda.reset_peak_memory_stats"), \
             patch("torch.cuda.max_memory_allocated", return_value=12), \
             patch("torch.cuda.max_memory_reserved", return_value=24), \
             patch("torch.cuda.empty_cache"):
            value, result = _measure(torch.device("cuda:0"), lambda: "not a valid result")
        self.assertIsNone(value)
        self.assertEqual(result["status"], "oom")
        self.assertEqual(result["peak_allocated_bytes"], 12)


@unittest.skipUnless(importlib.util.find_spec("transformers"), "DNA extra is not installed")
class DNAValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer, models, processors
        from transformers import ModernBertConfig, ModernBertForMaskedLM, PreTrainedTokenizerFast

        cls.checkpoint_temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.checkpoint_temp.cleanup)
        cls.checkpoint = Path(cls.checkpoint_temp.name)
        vocab = {token: index for index, token in enumerate(
            ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "A", "C", "G", "T"])}
        backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="[UNK]"))
        backend.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)])
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="[UNK]", cls_token="[CLS]",
            sep_token="[SEP]", pad_token="[PAD]", mask_token="[MASK]",
        )
        tokenizer.save_pretrained(cls.checkpoint)
        config = ModernBertConfig(
            vocab_size=9, hidden_size=16, intermediate_size=24,
            num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=8,
            local_attention=4, pad_token_id=3, cls_token_id=1, bos_token_id=1,
            sep_token_id=2, eos_token_id=2, attention_dropout=0., mlp_dropout=0.,
        )
        torch.manual_seed(37)
        ModernBertForMaskedLM(config).save_pretrained(cls.checkpoint)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.gbk = self.root / "gbk"
        self.gbk.mkdir()

    def write(self, name, sequence):
        from Bio import SeqIO
        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord
        record = SeqRecord(Seq(sequence), id=name, description="synthetic validation")
        record.annotations["molecule_type"] = "DNA"
        SeqIO.write(record, self.gbk / f"{name}.gbk", "genbank")

    def argv(self, output="validation", *extra):
        return ["--gbk", str(self.gbk), "--checkpoint", str(self.checkpoint),
                "--output", str(self.root / output), *extra]

    def test_real_full_forward_exceeds_config_and_keeps_weights_and_pooling(self):
        self.write("short", "ACGT")
        self.write("long", "ACGTNACGTACGTACGT")
        before = (self.checkpoint / "model.safetensors").read_bytes()
        args = build_parser().parse_args(self.argv())
        report = run(args)
        self.assertEqual(report["status"], "passed", report)
        self.assertEqual(report["selected_count"], 2)
        self.assertTrue(report["parameters_unchanged"])
        self.assertTrue(report["frozen_before"] and report["frozen_after"])
        rows = {row["bgc_id"]: row for row in report["results"]}
        self.assertEqual(rows["long"]["full_forward_calls"], 1)
        self.assertEqual(rows["long"]["full_actual_input_tokens"], 19)
        self.assertEqual(rows["long"]["windowed_metadata"]["window_count"], 3)
        self.assertTrue(rows["long"]["reference_pooling_matches"])
        self.assertTrue(rows["short"]["single_window_matches"])
        self.assertTrue(rows["short"]["repeat_matches"])
        self.assertNotIn("repeat_matches", rows["long"])
        self.assertEqual(before, (self.checkpoint / "model.safetensors").read_bytes())
        matrix = np.load(args.output / "full_embeddings.npy")
        self.assertEqual(matrix.shape, (2, 16))
        np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-6)
        with (args.output / "full_bgc_ids.tsv").open() as handle:
            ids = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual([row["bgc_id"] for row in ids], ["long", "short"])

    def test_resource_cap_fails_without_truncation_and_cli_exit_is_nonzero(self):
        self.write("long", "ACGTACGTACGT")
        with self.assertRaises(SystemExit) as caught:
            main(self.argv("capped", "--max-tokens", "8"))
        self.assertEqual(caught.exception.code, 1)
        output = self.root / "capped"
        report = json.loads((output / "summary.json").read_text())
        self.assertEqual(report["status"], "failed")
        row = report["results"][0]
        self.assertEqual(row["full_status"], "failed")
        self.assertEqual(row["windowed_status"], "passed")
        self.assertFalse(row["validation_passed"])
        self.assertEqual(np.load(output / "full_embeddings.npy").shape, (0, 16))
        self.assertEqual(np.load(output / "windowed_embeddings.npy").shape, (1, 16))
        self.assertIn("long", (output / "failures.tsv").read_text())

    def test_selection_exclusions_and_unparsed_inputs_are_preserved(self):
        self.write("short", "AC")
        self.write("middle", "ACGTAC")
        self.write("long", "ACGTACGTACGT")
        self.write("invalid", "AC-Z")
        args = build_parser().parse_args(self.argv("subset", "--sample-count", "1", "--exclude-id", "middle"))
        report = run(args)
        self.assertEqual(report["status"], "failed")  # Invalid scan is not silently omitted.
        self.assertEqual(report["selected_count"], 1)
        with (args.output / "inventory.tsv").open() as handle:
            rows = {r["bgc_id"]: r for r in csv.DictReader(handle, delimiter="\t")}
        self.assertEqual(rows["middle"]["status"], "excluded")
        self.assertEqual(rows["long"]["status"], "not_sampled")
        self.assertEqual(rows["invalid"]["status"], "failed")
        self.assertEqual(rows["short"]["status"], "passed")

    def test_explicit_cohort_disables_sampling_and_output_cannot_be_overwritten(self):
        self.write("a", "ACGT")
        self.write("b", "ACGTACGTAC")
        self.write("c", "ACGTAC")
        cohort = self.root / "cohort.tsv"
        cohort.write_text("bgc_id\na\nb\n")
        args = build_parser().parse_args(self.argv("explicit", "--ids-from", str(cohort), "--sample-count", "1"))
        report = run(args)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["selected_count"], 2)
        content = (args.output / "summary.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "already exist"):
            run(args)
        self.assertEqual(content, (args.output / "summary.json").read_bytes())

    def test_bp_cutoff_precedes_tokenization_and_validation_sampling(self):
        from data_preparation.frozen_dna import FrozenDNA

        self.write("short", "ACGT")
        self.write("boundary", "ACGTACGT")
        self.write("long", "ACGTACGTN")
        original_tokenize = FrozenDNA._tokenize

        def checked_tokenize(encoder, sequence):
            self.assertLessEqual(len(sequence), 8)
            return original_tokenize(encoder, sequence)

        args = build_parser().parse_args(self.argv("cutoff", "--max-bp", "8"))
        with patch.object(FrozenDNA, "_tokenize", checked_tokenize):
            report = run(args)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["selected_count"], 2)
        self.assertEqual(report["length_excluded_count"], 1)
        self.assertEqual(report["failures"], [])
        with (args.output / "inventory.tsv").open() as handle:
            rows = {r["bgc_id"]: r for r in csv.DictReader(handle, delimiter="\t")}
        self.assertEqual(rows["boundary"]["status"], "passed")
        self.assertEqual(rows["long"]["status"], "length_excluded")
        self.assertEqual(rows["long"]["base_count"], "9")
        self.assertEqual(rows["long"]["token_count"], "")

    def test_no_inputs_below_bp_cutoff_fails_with_inventory(self):
        self.write("long", "ACGTN")
        args = build_parser().parse_args(self.argv("empty", "--max-bp", "4"))
        with patch("data_preparation.frozen_dna.FrozenDNA.encode") as encode:
            report = run(args)
        encode.assert_not_called()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["selected_count"], 0)
        self.assertEqual(report["length_excluded_count"], 1)

    def test_nonpositive_bp_limit_is_rejected(self):
        args = build_parser().parse_args(self.argv("bad_limit", "--max-bp", "0"))
        with self.assertRaisesRegex(ValueError, "max-bp"):
            run(args)
        self.assertFalse(args.output.exists())

    def test_auditor_rejects_a_windowed_fallback_presented_as_full(self):
        from data_preparation.frozen_dna import FrozenDNA

        self.write("long", "ACGTACGTACGT")
        original = FrozenDNA.encode

        def wrong_mode(encoder, sequence):
            if encoder.sequence_mode == "full":
                windowed = FrozenDNA(encoder.model, encoder.tokenizer, sequence_mode="windowed")
                return original(windowed, sequence)
            return original(encoder, sequence)

        with patch.object(FrozenDNA, "encode", wrong_mode):
            report = run(build_parser().parse_args(self.argv("broken")))
        self.assertEqual(report["status"], "failed")
        self.assertIn("single forward", report["results"][0]["full_error"])
        self.assertEqual(report["results"][0]["windowed_status"], "passed")


if __name__ == "__main__":
    unittest.main()
