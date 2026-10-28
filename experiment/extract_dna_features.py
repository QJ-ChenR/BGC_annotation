"""Export one frozen ModernBERT DNA vector per BGC for the existing UMAP CLI."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np

from data_preparation.dna_inputs import DNA_INPUT_VERSION, discover_gbk, read_dna
from data_preparation.feature_cache import file_sha256


OUTPUT_FILES = ("embeddings.npy", "bgc_ids.tsv", "metadata.json", "failures.tsv")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gbk", type=Path, required=True, help="One single-record GenBank file or directory.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local trained HF MLM checkpoint directory.")
    parser.add_argument("--tokenizer", type=Path, help="Tokenizer directory if separate from the checkpoint.")
    parser.add_argument("--config", type=Path, help="Config file/directory if separate from the checkpoint.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--attention-backend", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument(
        "--sequence-mode", choices=("windowed", "full"), default="windowed",
        help="windowed preserves the original baseline; full encodes the entire BGC in one forward, without truncation.",
    )
    parser.add_argument(
        "--max-tokens", type=int,
        help="Including CLS/SEP: windowed mode uses this window size (default: config limit); "
             "full mode uses it only as an optional hard input limit (default: unlimited). "
             "Exceeding a full-mode limit fails; it never truncates or switches to windows.",
    )
    parser.add_argument("--window-batch-size", type=int, default=1, help="Windows per forward; full mode requires 1.")
    parser.add_argument(
        "--max-bp", type=int,
        help="Optional inclusive whole-record DNA length cutoff in bp (e.g. 40000). "
             "Longer records are reported and excluded before tokenization, never truncated. "
             "Independent of --max-tokens; omitted means no bp cutoff.",
    )
    parser.add_argument("--ids-from", type=Path, help="Optional existing bgc_ids.tsv to select the same BGC cohort.")
    parser.add_argument("--exclude-id", action="append", default=[], help="Explicit BGC exclusion, repeatable and recorded.")
    parser.add_argument("--skip-invalid", action="store_true", help="Record invalid inputs and export the remaining BGCs; memory errors and full-input limits remain fatal.")
    return parser


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _write_failures(path, failures):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("bgc_id", "source", "stage", "error"), delimiter="\t")
        writer.writeheader()
        writer.writerows(failures)


def _select_paths(args):
    paths = discover_gbk(args.gbk)
    if args.ids_from is not None:
        with args.ids_from.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames or "bgc_id" not in reader.fieldnames:
                raise ValueError("--ids-from must contain a bgc_id column.")
            ids = [row["bgc_id"] for row in reader]
        if not ids or any(not key or key.strip() != key for key in ids) or len(ids) != len(set(ids)):
            raise ValueError("--ids-from must contain nonempty, unique BGC IDs without surrounding whitespace.")
        missing = set(ids) - {path.stem for path in paths}
        if missing:
            raise ValueError(f"Missing GenBank files for requested IDs: {sorted(missing)[:10]}")
        selected = set(ids)
        paths = [path for path in paths if path.stem in selected]
    exclusions = set(args.exclude_id)
    unknown = exclusions - {path.stem for path in paths}
    if unknown:
        raise ValueError(f"Excluded IDs are absent from the selected cohort: {sorted(unknown)}")
    return paths


def _versions():
    result = {"python": sys.version.split()[0]}
    for package in ("numpy", "torch", "transformers", "tokenizers", "biopython", "safetensors"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def _is_out_of_memory(exc):
    """OOM is an execution failure, never grounds for silently reducing a cohort."""
    import torch

    return isinstance(exc, (MemoryError, torch.OutOfMemoryError)) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


def run(args):
    if args.max_bp is not None and args.max_bp <= 0:
        raise ValueError("max-bp must be positive.")
    if args.window_batch_size <= 0 or (args.max_tokens is not None and args.max_tokens <= 2):
        raise ValueError("window-batch-size must be positive and max-tokens must exceed 2.")
    if args.sequence_mode == "full" and args.window_batch_size != 1:
        raise ValueError("Full sequence mode requires --window-batch-size 1.")
    paths = _select_paths(args)
    if len(set(args.exclude_id)) == len(paths):
        raise ValueError("All selected BGCs were explicitly excluded.")
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in OUTPUT_FILES):
        raise ValueError("Output artifacts already exist; choose a new --output directory.")
    failures, ids, vectors, records = [], [], [], {}
    length_exclusions = []
    length_filter = None if args.max_bp is None else {
        "max_bp": args.max_bp, "inclusive": True,
        "length_definition": "entire single GenBank record, including ambiguous IUPAC bases",
        "action": "exclude whole records exceeding the cutoff; never truncate",
    }
    attempted_ids = []
    encoder = None
    try:
        from data_preparation.frozen_dna import DNASequenceTooLongError, FrozenDNA

        encoder = FrozenDNA.from_checkpoint(
            args.checkpoint, tokenizer_path=args.tokenizer, config_path=args.config,
            device=args.device, dtype=args.dtype, attention_backend=args.attention_backend,
            max_tokens=args.max_tokens, batch_size=args.window_batch_size,
            sequence_mode=args.sequence_mode,
        )
        for index, path in enumerate(paths):
            if path.stem in args.exclude_id:
                failures.append({"bgc_id": path.stem, "source": str(path), "stage": "user_exclusion",
                                 "error": "Explicitly excluded with --exclude-id"})
                continue
            stage = "gbk_to_dna"
            attempted_ids.append(path.stem)
            try:
                sequence, record = read_dna(path)
                if args.max_bp is not None and record["base_count"] > args.max_bp:
                    reason = f"Whole-record length {record['base_count']} bp exceeds --max-bp {args.max_bp}"
                    length_exclusions.append({**record, "reason": reason})
                    failures.append({"bgc_id": path.stem, "source": str(path),
                                     "stage": "length_filter", "error": reason})
                    continue
                stage = "dna_encoding"
                vector, encoding = encoder.encode(sequence)
                array = vector.detach().cpu().numpy().astype(np.float32, copy=False)
                if array.ndim != 1 or not len(array) or not np.isfinite(array).all() or not np.any(array):
                    raise ValueError("DNA encoder produced an invalid vector.")
                if vectors and array.shape != vectors[0].shape:
                    raise ValueError("DNA encoder output dimension changed between BGCs.")
                if not np.isclose(np.linalg.norm(array), 1, atol=1e-5):
                    raise ValueError("DNA vector is not L2 normalized.")
                if not isinstance(encoding.get("token_count"), int) or encoding["token_count"] <= 0:
                    raise ValueError("DNA encoder must report a positive content token_count.")
                if args.sequence_mode == "full" and (
                    encoding.get("window_count") != 1 or encoding.get("forward_pass_count") != 1
                ):
                    raise ValueError("Full sequence mode must encode exactly one complete window in one forward.")
                ids.append(path.stem)
                vectors.append(array)
                records[path.stem] = {**record, **encoding}
            except (ValueError, RuntimeError, OSError, TypeError, KeyError, MemoryError) as exc:
                out_of_memory = _is_out_of_memory(exc)
                failures.append({"bgc_id": path.stem, "source": str(path), "stage": stage,
                                 "error": f"{type(exc).__name__}: {exc}"})
                if out_of_memory:
                    failures[-1]["stage"] = "resource_oom"
                    raise RuntimeError(
                        f"Out of memory while encoding {path.stem}; --skip-invalid does not skip resource failures. "
                        "No windowed fallback was attempted; see failures.tsv and metadata.json."
                    ) from exc
                if isinstance(exc, DNASequenceTooLongError):
                    failures[-1]["stage"] = "resource_token_limit"
                    raise
                if not args.skip_invalid:
                    raise ValueError(f"Input failed for {path.stem}; see failures.tsv.") from exc
            if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(paths):
                print(f"DNA encoding: {index + 1}/{len(paths)} inputs, {len(ids)} vectors", file=sys.stderr, flush=True)
        if not vectors:
            raise ValueError("No valid BGC vectors remain.")
        array = np.stack(vectors)
        root = Path(__file__).resolve().parents[1]
        metadata = {
            "format": "bgc_vectors_v1",
            "status": "complete_with_exclusions" if failures else "complete",
            "route": "dna", "count": len(ids), "dimension": int(array.shape[1]), "dtype": "float32",
            "sequence_mode": args.sequence_mode,
            "pooling": (
                "content_token_mean_single_full_sequence" if args.sequence_mode == "full"
                else "content_token_mean_across_nonoverlapping_windows"
            ),
            "normalization": "l2",
            "token_count_unit": "dna_bpe_tokens",
            "similarity": "cosine equals dot product of exported vectors",
            "device": args.device, "inference_dtype": args.dtype, "window_batch_size": args.window_batch_size,
            "extractor": encoder.metadata,
            "input": {
                "format": DNA_INPUT_VERSION, "path": str(args.gbk.resolve()), "input_count": len(paths),
                "sequence_scope": "entire single GenBank record; includes coding and noncoding sequence",
                "preprocessing": "uppercase; retain ambiguous IUPAC bases; retain original orientation",
                "reverse_complement_averaging": False,
                "training_preprocessing_verified": False,
                "cohort": None if args.ids_from is None else {
                    "path": str(args.ids_from.resolve()), "sha256": file_sha256(args.ids_from)},
                "explicit_exclusions": sorted(set(args.exclude_id)),
                "length_filter": length_filter,
            },
            "records": records, "excluded_count": len(failures),
            "length_exclusions": length_exclusions,
            "length_excluded_count": len(length_exclusions),
            "invalid_count": sum(item["stage"] not in {"user_exclusion", "length_filter"} for item in failures),
            "versions": _versions(),
            "source_sha256": {name: file_sha256(root / name) for name in (
                "experiment/extract_dna_features.py", "data_preparation/dna_inputs.py",
                "data_preparation/frozen_dna.py")},
        }
        np.save(args.output / "embeddings.npy", array, allow_pickle=False)
        with (args.output / "bgc_ids.tsv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(("row", "bgc_id", "token_count"))
            writer.writerows((row, key, records[key]["token_count"]) for row, key in enumerate(ids))
        _write_failures(args.output / "failures.tsv", failures)
        _write_json(args.output / "metadata.json", metadata)
        return metadata
    except Exception as exc:
        _write_failures(args.output / "failures.tsv", failures)
        _write_json(args.output / "metadata.json", {
            "format": "bgc_vectors_v1", "status": "failed", "route": "dna",
            "error": f"{type(exc).__name__}: {exc}", "excluded_count": len(failures),
            "sequence_mode": args.sequence_mode,
            "length_filter": length_filter,
            "length_exclusions": length_exclusions,
            "length_excluded_count": len(length_exclusions),
            "device": args.device, "inference_dtype": args.dtype,
            "attention_backend": args.attention_backend,
            "requested_max_tokens": args.max_tokens,
            "extractor": None if encoder is None else encoder.metadata,
            "input_count": len(paths),
            "selected_ids": [path.stem for path in paths],
            "explicit_exclusions": sorted(set(args.exclude_id)),
            "attempted_ids": attempted_ids,
            "completed_count": len(ids), "completed_ids": ids,
            "completed_records": records,
            "export_complete": False,
            "unprocessed_ids": [
                path.stem for path in paths
                if path.stem not in attempted_ids and path.stem not in args.exclude_id
            ],
        })
        raise


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        metadata = run(args)
    except (ValueError, RuntimeError, OSError, TypeError, KeyError, ImportError, MemoryError) as exc:
        parser.exit(1, f"DNA feature extraction failed: {exc}\n")
    print(f"Saved {metadata['count']} DNA vectors ({metadata['dimension']} dimensions) to {args.output}")
    if metadata["excluded_count"]:
        print(f"Excluded {metadata['excluded_count']} inputs; see failures.tsv.")


if __name__ == "__main__":
    main()
