"""Integration checks with a tiny local MLM checkpoint, never production weights."""

import csv
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from data_preparation.dna_inputs import discover_gbk, read_dna
from experiment.extract_dna_features import build_parser, run


def write_gbk(path, sequence, record_id="different_from_filename"):
    record = SeqRecord(Seq(sequence), id=record_id, description="synthetic DNA test")
    record.annotations["molecule_type"] = "DNA"
    SeqIO.write(record, path, "genbank")


class DNAInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_complete_record_without_cds_and_ambiguity_are_preserved(self):
        path = self.root / "BGC_A.gbk"
        write_gbk(path, "acgtnryacgt")
        sequence, record = read_dna(path)
        self.assertEqual(sequence, "ACGTNRYACGT")
        self.assertEqual(record["bgc_id"], "BGC_A")
        self.assertEqual(record["base_count"], 11)
        self.assertEqual(record["sequence_region"], [0, 11])
        self.assertEqual(record["ambiguous_base_count"], 3)
        self.assertEqual(record["gc_fraction_acgt"], 0.5)
        self.assertEqual(len(record["source_sha256"]), 64)

    def test_multirecord_and_invalid_bases_fail_instead_of_silent_repair(self):
        path = self.root / "BGC_A.gbk"
        write_gbk(path, "ACGT")
        text = path.read_text()
        path.write_text(text + text)
        with self.assertRaises(ValueError):
            read_dna(path)
        write_gbk(path, "ACGT-Z")
        with self.assertRaisesRegex(ValueError, "Unsupported DNA"):
            read_dna(path)

    def test_duplicate_stems_rejected_across_directories(self):
        write_gbk(self.root / "a.gbk", "ACGT")
        (self.root / "sub").mkdir()
        write_gbk(self.root / "sub" / "a.gb", "ACGT")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            discover_gbk(self.root)


@unittest.skipUnless(importlib.util.find_spec("transformers"), "DNA extra is not installed")
class DNAExportTests(unittest.TestCase):
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
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", cls_token="[CLS]",
                                           sep_token="[SEP]", pad_token="[PAD]", mask_token="[MASK]")
        tokenizer.save_pretrained(cls.checkpoint)
        config = ModernBertConfig(vocab_size=9, hidden_size=16, intermediate_size=24,
                                  num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=8,
                                  local_attention=4, pad_token_id=3, cls_token_id=50281, bos_token_id=50281,
                                  sep_token_id=50282, eos_token_id=50282, attention_dropout=0., mlp_dropout=0.)
        torch.manual_seed(17)
        ModernBertForMaskedLM(config).save_pretrained(cls.checkpoint)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / "gbk"
        self.inputs.mkdir()

    def args(self, name="output", *extra):
        return build_parser().parse_args([
            "--gbk", str(self.inputs), "--checkpoint", str(self.checkpoint),
            "--output", str(self.root / name), *extra])

    def test_real_checkpoint_long_inputs_export_aligned_normalized_vectors(self):
        write_gbk(self.inputs / "BGC_B.gbk", "ACGTNACGTACGTACGT")
        write_gbk(self.inputs / "BGC_A.gbk", "TTGCAA")
        first = self.args("one")
        metadata = run(first)
        second = self.args("batch", "--window-batch-size", "3")
        run(second)
        vectors = np.load(first.output / "embeddings.npy")
        self.assertEqual(vectors.shape, (2, 16))
        self.assertEqual(vectors.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
        np.testing.assert_allclose(vectors, np.load(second.output / "embeddings.npy"), atol=2e-6)
        with (first.output / "bgc_ids.tsv").open() as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual([row["bgc_id"] for row in rows], ["BGC_A", "BGC_B"])
        self.assertEqual([int(row["token_count"]) for row in rows], [6, 17])
        self.assertEqual(metadata["route"], "dna")
        self.assertEqual(metadata["token_count_unit"], "dna_bpe_tokens")
        self.assertFalse(metadata["input"]["training_preprocessing_verified"])
        self.assertEqual(metadata["records"]["BGC_B"]["ambiguous_base_count"], 1)
        self.assertEqual(metadata["status"], "complete")
        # Correcting legacy token IDs is in memory only.
        self.assertEqual(json.loads((self.checkpoint / "config.json").read_text())["cls_token_id"], 50281)

    def test_full_mode_exceeds_config_length_in_one_forward_without_checkpoint_changes(self):
        write_gbk(self.inputs / "BGC_B.gbk", "ACGTNACGTACGTACGT")
        write_gbk(self.inputs / "BGC_A.gbk", "TTGCAA")
        original_files = {path.name: path.read_bytes() for path in self.checkpoint.iterdir() if path.is_file()}
        args = self.args("full", "--sequence-mode", "full")
        metadata = run(args)
        vectors = np.load(args.output / "embeddings.npy", allow_pickle=False)
        self.assertEqual(vectors.shape, (2, 16))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
        self.assertEqual(metadata["sequence_mode"], "full")
        self.assertEqual(metadata["pooling"], "content_token_mean_single_full_sequence")
        self.assertFalse(metadata["input"]["training_preprocessing_verified"])
        record = metadata["records"]["BGC_B"]
        self.assertEqual(record["token_count"], 17)
        self.assertEqual(record["window_count"], 1)
        self.assertEqual(record["forward_pass_count"], 1)
        self.assertEqual(record["total_input_tokens"], 19)
        self.assertTrue(record["exceeds_config_max_position_embeddings"])
        self.assertEqual(record["windows"], [{
            "token_start": 0, "token_end": 17, "base_start": 0, "base_end": 17,
        }])
        self.assertEqual(
            {path.name: path.read_bytes() for path in self.checkpoint.iterdir() if path.is_file()},
            original_files,
        )
        # An input already fitting one old window has identical context in both modes.
        windowed = self.args("windowed")
        run(windowed)
        np.testing.assert_allclose(vectors[0], np.load(windowed.output / "embeddings.npy")[0], atol=2e-6)

    def test_full_input_limit_is_fatal_even_with_skip_invalid_and_preserves_audit(self):
        from data_preparation.frozen_dna import DNASequenceTooLongError

        write_gbk(self.inputs / "A_short.gbk", "ACGT")
        write_gbk(self.inputs / "B_long.gbk", "ACGTACGTACGT")
        write_gbk(self.inputs / "C_unattempted.gbk", "TGCA")
        args = self.args("limited", "--sequence-mode", "full", "--max-tokens", "10", "--skip-invalid")
        with self.assertRaises(DNASequenceTooLongError):
            run(args)
        self.assertFalse((args.output / "embeddings.npy").exists())
        metadata = json.loads((args.output / "metadata.json").read_text())
        self.assertEqual(metadata["status"], "failed")
        self.assertEqual(metadata["completed_ids"], ["A_short"])
        self.assertEqual(metadata["attempted_ids"], ["A_short", "B_long"])
        self.assertEqual(metadata["unprocessed_ids"], ["C_unattempted"])
        self.assertEqual(metadata["sequence_mode"], "full")
        with (args.output / "failures.tsv").open() as handle:
            failures = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(failures[0]["bgc_id"], "B_long")
        self.assertEqual(failures[0]["stage"], "resource_token_limit")

    def test_cuda_oom_is_fatal_even_with_skip_invalid_without_window_fallback(self):
        write_gbk(self.inputs / "A.gbk", "ACGTACGT")
        write_gbk(self.inputs / "B.gbk", "ACGT")
        args = self.args("oom", "--sequence-mode", "full", "--skip-invalid")
        with patch("data_preparation.frozen_dna.FrozenDNA.encode", side_effect=torch.OutOfMemoryError("CUDA out of memory")) as encode:
            with self.assertRaisesRegex(RuntimeError, "Out of memory.*A"):
                run(args)
        self.assertEqual(encode.call_count, 1)
        self.assertFalse((args.output / "embeddings.npy").exists())
        metadata = json.loads((args.output / "metadata.json").read_text())
        self.assertEqual(metadata["status"], "failed")
        self.assertEqual(metadata["unprocessed_ids"], ["B"])
        self.assertIn("resource_oom", (args.output / "failures.tsv").read_text())

    def test_full_mode_does_not_accept_window_batching(self):
        write_gbk(self.inputs / "A.gbk", "ACGT")
        args = self.args("bad_batch", "--sequence-mode", "full", "--window-batch-size", "2")
        with self.assertRaisesRegex(ValueError, "window-batch-size 1"):
            run(args)
        self.assertFalse(args.output.exists())

    def test_cohort_selection_explicit_exclusion_and_existing_output_protection(self):
        for key in ("A", "B", "C"):
            write_gbk(self.inputs / f"{key}.gbk", "ACGTACGT")
        cohort = self.root / "bgc_ids.tsv"
        cohort.write_text("row\tbgc_id\ttoken_count\n0\tC\t2\n1\tA\t4\n")
        args = self.args("output", "--ids-from", str(cohort), "--exclude-id", "C")
        metadata = run(args)
        self.assertEqual(list(metadata["records"]), ["A"])
        self.assertEqual(metadata["excluded_count"], 1)
        self.assertEqual(metadata["invalid_count"], 0)
        self.assertIn("user_exclusion", (args.output / "failures.tsv").read_text())
        before = (args.output / "embeddings.npy").read_bytes()
        with self.assertRaisesRegex(ValueError, "already exist"):
            run(args)
        self.assertEqual(before, (args.output / "embeddings.npy").read_bytes())

    def test_invalid_inputs_are_reported_and_skip_is_explicit(self):
        write_gbk(self.inputs / "bad.gbk", "ACGT-Z")
        write_gbk(self.inputs / "good.gbk", "ACGTACGT")
        args = self.args()
        with self.assertRaisesRegex(ValueError, "bad"):
            run(args)
        self.assertFalse((args.output / "embeddings.npy").exists())
        self.assertEqual(json.loads((args.output / "metadata.json").read_text())["status"], "failed")
        valid = self.args("valid", "--skip-invalid")
        metadata = run(valid)
        self.assertEqual(metadata["count"], 1)
        self.assertEqual(metadata["invalid_count"], 1)
        self.assertIn("bad", (valid.output / "failures.tsv").read_text())

    def test_missing_requested_id_and_bad_exclusion_do_not_silently_change_cohort(self):
        write_gbk(self.inputs / "A.gbk", "ACGT")
        cohort = self.root / "ids.tsv"
        cohort.write_text("bgc_id\nA\nmissing\n")
        with self.assertRaisesRegex(ValueError, "Missing GenBank"):
            run(self.args("missing", "--ids-from", str(cohort)))
        with self.assertRaisesRegex(ValueError, "absent"):
            run(self.args("excluded", "--exclude-id", "typo"))

    def test_incomplete_real_checkpoint_is_rejected_before_feature_extraction(self):
        from safetensors.torch import load_file, save_file
        from data_preparation.frozen_dna import FrozenDNA

        damaged = self.root / "damaged_checkpoint"
        shutil.copytree(self.checkpoint, damaged)
        weights = load_file(damaged / "model.safetensors")
        del weights["model.layers.0.attn.Wqkv.weight"]
        save_file(weights, damaged / "model.safetensors", metadata={"format": "pt"})
        with self.assertRaisesRegex(ValueError, "strictly match"):
            FrozenDNA.from_checkpoint(damaged)


if __name__ == "__main__":
    unittest.main()
