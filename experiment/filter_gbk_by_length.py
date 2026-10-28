"""Select complete single-record GenBank BGCs by an inclusive DNA length limit.

Copies eligible files unchanged into OUTPUT/gbk for direct use with the DNA
encoder. This preprocessing command needs Biopython, but no model or tokenizer.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile

from data_preparation.dna_inputs import DNA_INPUT_VERSION, discover_gbk, file_sha256, read_dna


ARTIFACTS = ("gbk", "bgc_ids.tsv", "inventory.tsv", "failures.tsv", "metadata.json")
INVENTORY_FIELDS = (
    "bgc_id", "status", "base_count", "source", "source_sha256", "sequence_sha256",
    "record_id", "ambiguous_base_count", "output", "error",
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gbk", type=Path, required=True, help="Single-record GenBank file or directory (searched recursively).")
    parser.add_argument("--max-bp", type=int, default=40000, help="Inclusive whole-record DNA length limit in bp (default: 40000).")
    parser.add_argument("--output", type=Path, required=True, help="Output directory outside the input directory; existing artifacts are never overwritten.")
    parser.add_argument("--skip-invalid", action="store_true", help="Explicitly permit invalid files to be excluded; all are retained in the audit tables.")
    return parser


def _table(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_audit(output, metadata, inventory):
    _table(output / "inventory.tsv", INVENTORY_FIELDS, inventory)
    _table(output / "failures.tsv", ("bgc_id", "source", "stage", "error"), [
        {**row, "stage": "gbk_to_dna"} for row in inventory if row["status"] == "invalid"
    ])
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8",
    )


def _output_name(path):
    # Explicit single-file inputs may have an arbitrary extension, whereas the
    # downstream directory reader discovers only these GenBank extensions.
    return path.name if path.suffix.lower() in {".gb", ".gbk", ".genbank"} else f"{path.stem}.gbk"


def run(args):
    if args.max_bp <= 0:
        raise ValueError("--max-bp must be a positive integer.")
    source, output = args.gbk.resolve(), args.output.resolve()
    if source.is_dir() and (output == source or source in output.parents):
        raise ValueError("--output must be outside the input directory to avoid recursive input contamination.")
    paths = discover_gbk(source)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() or (output / name).is_symlink() for name in ARTIFACTS):
        raise ValueError("Output artifacts already exist; choose a new --output directory.")

    inventory, eligible = [], []
    for path in paths:
        row = {"bgc_id": path.stem, "source": str(path.resolve()), "output": "", "error": ""}
        try:
            _, record = read_dna(path)
            row.update({key: record[key] for key in (
                "base_count", "source_sha256", "sequence_sha256", "record_id", "ambiguous_base_count",
            )})
            if record["base_count"] <= args.max_bp:
                row["status"] = "eligible"
                eligible.append((path, row))
            else:
                row["status"] = "length_excluded"
                row["error"] = f"Whole-record length {record['base_count']} bp exceeds inclusive limit {args.max_bp} bp"
        except (ValueError, OSError, TypeError, KeyError) as exc:
            row.update(status="invalid", error=f"{type(exc).__name__}: {exc}")
            try:
                row["source_sha256"] = file_sha256(path)
            except OSError:
                pass
        inventory.append(row)

    invalid_count = sum(row["status"] == "invalid" for row in inventory)
    metadata = {
        "format": "bgc_length_filter_v1", "status": "failed", "export_complete": False,
        "input": {"path": str(source), "format": DNA_INPUT_VERSION, "input_count": len(paths)},
        "max_bp": args.max_bp, "comparison": "base_count <= max_bp", "inclusive": True,
        "length_unit": "bp", "length_scope": "entire single GenBank record, including ambiguous IUPAC bases",
        "sequence_policy": "no truncation, no region inference, no CDS selection; accepted GenBank files copied byte-for-byte",
        "skip_invalid": bool(args.skip_invalid), "eligible_count": len(eligible),
        "selected_count": 0,
        "length_excluded_count": sum(row["status"] == "length_excluded" for row in inventory),
        "invalid_count": invalid_count, "selected_ids": [],
        "selected_gbk_directory": "gbk", "selected_ids_file": "bgc_ids.tsv",
        "source_sha256": {
            "experiment/filter_gbk_by_length.py": file_sha256(Path(__file__)),
            "data_preparation/dna_inputs.py": file_sha256(Path(__file__).resolve().parents[1] / "data_preparation/dna_inputs.py"),
        },
    }
    error = None
    if invalid_count and not args.skip_invalid:
        error = f"{invalid_count} invalid GenBank inputs; see inventory.tsv and failures.tsv. Use --skip-invalid to explicitly exclude them."
    elif not eligible:
        error = "No valid BGCs satisfy the inclusive length limit; no GenBank subset was exported."
    if error:
        metadata["error"] = error
        for _, row in eligible:
            row["status"] = "not_exported"
        _save_audit(output, metadata, inventory)
        raise ValueError(error)

    published_gbk = False
    started_manifest = False
    try:
        # Publish only after every selected file has been copied and hash-checked.
        with tempfile.TemporaryDirectory(prefix=".gbk-stage-", dir=output) as temporary:
            staged = Path(temporary) / "gbk"
            staged.mkdir()
            for path, row in eligible:
                copied = staged / _output_name(path)
                shutil.copyfile(path, copied)
                if file_sha256(copied) != row["source_sha256"]:
                    raise ValueError(f"GenBank input changed during filtering: {path}")
            staged.rename(output / "gbk")
            published_gbk = True
        for path, row in eligible:
            row.update(status="selected", output=f"gbk/{_output_name(path)}")
        started_manifest = True
        _table(output / "bgc_ids.tsv", ("row", "bgc_id", "base_count"), [
            {"row": index, "bgc_id": row["bgc_id"], "base_count": row["base_count"]}
            for index, (_, row) in enumerate(eligible)
        ])
        metadata.update(
            status="complete_with_exclusions" if len(eligible) != len(paths) else "complete",
            export_complete=True, selected_count=len(eligible),
            selected_ids=[row["bgc_id"] for _, row in eligible],
        )
        _save_audit(output, metadata, inventory)
    except (ValueError, OSError, TypeError) as exc:
        # These artifacts were absent at startup and are owned by this run.
        # A failed manifest/audit write must not leave a usable partial cohort.
        if published_gbk:
            shutil.rmtree(output / "gbk")
        if started_manifest:
            (output / "bgc_ids.tsv").unlink(missing_ok=True)
        metadata.update(
            status="failed", export_complete=False, selected_count=0,
            selected_ids=[], error=f"{type(exc).__name__}: {exc}",
        )
        for _, row in eligible:
            row.update(status="not_exported", output="")
        _save_audit(output, metadata, inventory)
        raise
    return metadata


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        metadata = run(args)
    except (ValueError, RuntimeError, OSError, TypeError, ImportError) as exc:
        parser.exit(1, f"GenBank length filtering failed: {exc}\n")
    print(
        f"Selected {metadata['selected_count']} of {metadata['input']['input_count']} BGCs "
        f"with length <= {metadata['max_bp']} bp into {args.output / 'gbk'}. "
        f"Length exclusions: {metadata['length_excluded_count']}; invalid inputs: {metadata['invalid_count']}."
    )


if __name__ == "__main__":
    main()
