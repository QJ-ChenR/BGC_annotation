"""Length filtering uses actual complete DNA and produces portable GenBank input."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from data_preparation.dna_inputs import discover_gbk, read_dna
from experiment import filter_gbk_by_length
from experiment.filter_gbk_by_length import build_parser, main, run


def write_gbk(path, sequence):
    record = SeqRecord(Seq(sequence), id="record_id_not_filename", description="synthetic whole-record DNA")
    record.annotations["molecule_type"] = "DNA"
    SeqIO.write(record, path, "genbank")


def table(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class GenBankLengthFilterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()

    def args(self, *extra, output="filtered"):
        return build_parser().parse_args([
            "--gbk", str(self.inputs), "--output", str(self.root / output), *extra,
        ])

    def test_40000_bp_boundary_nested_discovery_byte_preservation_and_alignment(self):
        nested = self.inputs / "nested"
        nested.mkdir()
        write_gbk(nested / "A.GB", "A" * 39999)
        write_gbk(self.inputs / "B.genbank", "A" * 39996 + "NRYS")
        write_gbk(self.inputs / "C.gbk", "A" * 40001)
        originals = {path.stem: path.read_bytes() for path in discover_gbk(self.inputs)}
        args = self.args()
        metadata = run(args)
        self.assertTrue(metadata["export_complete"])
        self.assertEqual(metadata["max_bp"], 40000)
        self.assertEqual(metadata["selected_ids"], ["A", "B"])
        self.assertEqual(metadata["selected_count"], 2)
        self.assertEqual(metadata["length_excluded_count"], 1)
        self.assertEqual(metadata["invalid_count"], 0)
        selected = discover_gbk(args.output / "gbk")
        self.assertEqual([path.name for path in selected], ["A.GB", "B.genbank"])
        for path in selected:
            self.assertEqual(path.read_bytes(), originals[path.stem])
            self.assertLessEqual(read_dna(path)[1]["base_count"], 40000)
        rows = table(args.output / "bgc_ids.tsv")
        self.assertEqual(rows, [
            {"row": "0", "bgc_id": "A", "base_count": "39999"},
            {"row": "1", "bgc_id": "B", "base_count": "40000"},
        ])
        inventory = table(args.output / "inventory.tsv")
        self.assertEqual([row["status"] for row in inventory], ["selected", "selected", "length_excluded"])
        self.assertEqual(inventory[1]["ambiguous_base_count"], "4")
        self.assertEqual(inventory[2]["base_count"], "40001")
        self.assertEqual(inventory[2]["output"], "")
        self.assertTrue(all(len(row["source_sha256"]) == 64 for row in inventory))
        self.assertEqual(table(args.output / "failures.tsv"), [])

    def test_invalid_input_keeps_complete_audit_without_publishing_partial_subset(self):
        write_gbk(self.inputs / "A_bad.gbk", "ACGT-Z")
        write_gbk(self.inputs / "B_good.gbk", "ACGT")
        write_gbk(self.inputs / "C_long.gbk", "A" * 9)
        args = self.args("--max-bp", "8")
        with self.assertRaisesRegex(ValueError, "invalid GenBank"):
            run(args)
        metadata = json.loads((args.output / "metadata.json").read_text())
        self.assertEqual(metadata["status"], "failed")
        self.assertFalse(metadata["export_complete"])
        self.assertEqual(metadata["eligible_count"], 1)
        self.assertEqual(metadata["selected_count"], 0)
        self.assertFalse((args.output / "gbk").exists())
        self.assertFalse((args.output / "bgc_ids.tsv").exists())
        rows = table(args.output / "inventory.tsv")
        self.assertEqual([row["status"] for row in rows], ["invalid", "not_exported", "length_excluded"])
        self.assertEqual(len(rows[0]["source_sha256"]), 64)
        self.assertEqual(table(args.output / "failures.tsv")[0]["bgc_id"], "A_bad")

    def test_skip_invalid_is_explicit_and_preserves_ambiguous_bases(self):
        write_gbk(self.inputs / "bad.gbk", "ACGT-Z")
        write_gbk(self.inputs / "good.gbk", "ACGNRYTN")
        args = self.args("--max-bp", "8", "--skip-invalid")
        metadata = run(args)
        self.assertEqual(metadata["status"], "complete_with_exclusions")
        self.assertEqual(metadata["selected_ids"], ["good"])
        self.assertEqual(metadata["invalid_count"], 1)
        sequence, record = read_dna(args.output / "gbk" / "good.gbk")
        self.assertEqual(sequence, "ACGNRYTN")
        self.assertEqual(record["ambiguous_base_count"], 4)
        self.assertEqual(record["base_count"], 8)

    def test_multirecord_file_is_invalid_not_summed_or_first_record_only(self):
        path = self.inputs / "multi.gbk"
        write_gbk(path, "ACGT")
        original = path.read_text()
        path.write_text(original + original)
        args = self.args()
        with self.assertRaisesRegex(ValueError, "invalid GenBank"):
            run(args)
        self.assertEqual(table(args.output / "inventory.tsv")[0]["status"], "invalid")

    def test_empty_selection_audited_and_cli_returns_nonzero(self):
        write_gbk(self.inputs / "long.gbk", "ACGT")
        args = self.args("--max-bp", "3")
        with self.assertRaisesRegex(ValueError, "No valid BGCs"):
            run(args)
        self.assertFalse((args.output / "gbk").exists())
        self.assertEqual(table(args.output / "inventory.tsv")[0]["status"], "length_excluded")
        metadata = json.loads((args.output / "metadata.json").read_text())
        self.assertEqual(metadata["length_excluded_count"], 1)
        with self.assertRaises(SystemExit) as raised:
            main(["--gbk", str(self.inputs), "--max-bp", "3", "--output", str(self.root / "cli_empty")])
        self.assertEqual(raised.exception.code, 1)

    def test_duplicate_ids_and_invalid_limits_rejected_before_output_creation(self):
        write_gbk(self.inputs / "A.gbk", "ACGT")
        for limit in ("0", "-1"):
            args = self.args("--max-bp", limit)
            with self.assertRaisesRegex(ValueError, "positive"):
                run(args)
            self.assertFalse(args.output.exists())
        (self.inputs / "nested").mkdir()
        write_gbk(self.inputs / "nested" / "A.gb", "ACGT")
        args = self.args()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            run(args)
        self.assertFalse(args.output.exists())

    def test_output_is_outside_input_and_existing_artifacts_are_never_overwritten(self):
        write_gbk(self.inputs / "A.gbk", "ACGT")
        for output in (self.inputs, self.inputs / "filtered"):
            args = build_parser().parse_args(["--gbk", str(self.inputs), "--output", str(output)])
            with self.assertRaisesRegex(ValueError, "outside"):
                run(args)
        args = self.args()
        args.output.mkdir()
        unrelated = args.output / "user_notes.txt"
        unrelated.write_text("preserve")
        run(args)
        self.assertEqual(unrelated.read_text(), "preserve")
        before = (args.output / "metadata.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "already exist"):
            run(args)
        self.assertEqual((args.output / "metadata.json").read_bytes(), before)

    def test_input_change_during_copy_aborts_without_publishing_gbk_directory(self):
        path = self.inputs / "A.gbk"
        write_gbk(path, "ACGT")
        args = self.args()

        def change_during_copy(source, destination):
            destination.write_bytes(source.read_bytes() + b"\n")

        with patch("experiment.filter_gbk_by_length.shutil.copyfile", side_effect=change_during_copy):
            with self.assertRaisesRegex(ValueError, "changed during filtering"):
                run(args)
        self.assertFalse((args.output / "gbk").exists())
        metadata = json.loads((args.output / "metadata.json").read_text())
        self.assertFalse(metadata["export_complete"])
        self.assertEqual(metadata["selected_count"], 0)
        self.assertEqual(table(args.output / "inventory.tsv")[0]["status"], "not_exported")

    def test_single_file_input_is_supported(self):
        path = self.inputs / "single.gbk"
        write_gbk(path, "ACGT")
        args = self.args("--gbk", str(path))
        metadata = run(args)
        self.assertEqual(metadata["selected_ids"], ["single"])
        self.assertEqual((args.output / "gbk" / path.name).read_bytes(), path.read_bytes())

    def test_single_file_with_nonstandard_suffix_remains_discoverable_after_copy(self):
        path = self.inputs / "single.txt"
        write_gbk(path, "ACGTN")
        args = self.args("--gbk", str(path))
        metadata = run(args)
        copied = discover_gbk(args.output / "gbk")
        self.assertEqual(metadata["selected_ids"], ["single"])
        self.assertEqual([item.name for item in copied], ["single.gbk"])
        self.assertEqual(copied[0].read_bytes(), path.read_bytes())
        self.assertEqual(read_dna(copied[0])[1]["bgc_id"], "single")
        self.assertEqual(table(args.output / "inventory.tsv")[0]["output"], "gbk/single.gbk")

    def test_manifest_or_audit_write_failure_rolls_back_only_newly_published_cohort(self):
        write_gbk(self.inputs / "A.gbk", "ACGT")
        original_table = filter_gbk_by_length._table
        original_audit = filter_gbk_by_length._save_audit
        for failure_stage in ("manifest", "audit"):
            with self.subTest(failure_stage=failure_stage):
                args = self.args(output=failure_stage)
                args.output.mkdir()
                notes = args.output / "notes.txt"
                notes.write_text("user content")

                def fail_manifest(path, fields, rows):
                    if path.name == "bgc_ids.tsv":
                        path.write_text("partial manifest")
                        raise OSError("injected manifest write failure")
                    return original_table(path, fields, rows)

                def fail_audit(output, metadata, inventory):
                    if metadata["export_complete"]:
                        raise OSError("injected audit write failure")
                    return original_audit(output, metadata, inventory)

                target = "_table" if failure_stage == "manifest" else "_save_audit"
                failure = fail_manifest if failure_stage == "manifest" else fail_audit
                with patch.object(filter_gbk_by_length, target, side_effect=failure):
                    with self.assertRaisesRegex(OSError, "injected"):
                        run(args)
                self.assertFalse((args.output / "gbk").exists())
                self.assertFalse((args.output / "bgc_ids.tsv").exists())
                self.assertEqual(notes.read_text(), "user content")
                metadata = json.loads((args.output / "metadata.json").read_text())
                self.assertEqual(metadata["status"], "failed")
                self.assertEqual(metadata["selected_count"], 0)
                self.assertEqual(metadata["selected_ids"], [])
                self.assertFalse(metadata["export_complete"])
                row = table(args.output / "inventory.tsv")[0]
                self.assertEqual(row["status"], "not_exported")
                self.assertEqual(row["output"], "")

    def test_filter_import_does_not_import_torch_or_transformers(self):
        result = subprocess.run([
            sys.executable, "-c",
            "import sys; import experiment.filter_gbk_by_length; "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules",
        ], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
