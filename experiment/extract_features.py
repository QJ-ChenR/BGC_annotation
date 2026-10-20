"""Extract candidate-independent, frozen BGC vectors without training.

Run from the repository root with ``python -m experiment.extract_features``.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from data_preparation.feature_cache import (
    CACHE_FORMAT,
    file_sha256,
    load_feature_cache,
    token_tensor,
)
from model.features import BGCFeatureExtractor


OUTPUT_FILES = ("embeddings.npy", "bgc_ids.tsv", "metadata.json", "failures.tsv")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--esm-cache", type=Path, help="BGC -> ESM token tensor cache.")
    source.add_argument("--gbk", type=Path, help="One GenBank file or directory (recursive).")
    parser.add_argument("--route", choices=("esm", "mac", "map"), default="esm")
    parser.add_argument("--checkpoint", type=Path, help="Trained MAC/MAP .ckpt file.")
    parser.add_argument("--member", type=int, default=0, help="Zero-based ensemble member.")
    parser.add_argument("--device", default="cpu", help="Explicit torch device, e.g. cpu or cuda:0.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True, help="Directory for vectors and manifests.")
    parser.add_argument("--esm-checkpoint", type=Path, help="Local ESM2 checkpoint for GBK input.")
    parser.add_argument("--esm-pooling", choices=("bos", "mean"), default="bos")
    parser.add_argument("--esm-batch-size", type=int, default=8)
    parser.add_argument("--max-residues", type=int, default=1021)
    parser.add_argument("--save-cache", type=Path, help="Save versioned token cache for GBK input.")
    parser.add_argument("--skip-invalid", action="store_true", help="Export valid BGCs and list every exclusion.")
    parser.add_argument("--trusted-input", action="store_true",
                        help="Allow legacy pickle deserialization for trusted caches/checkpoints.")
    return parser


def _write_failures(path, failures):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("bgc_id", "source", "stage", "error"), delimiter="\t")
        writer.writeheader()
        writer.writerows(failures)


def _json_write(path, data):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _failure(bgc_id, source, stage, exc):
    return {"bgc_id": bgc_id, "source": str(source), "stage": stage,
            "error": f"{type(exc).__name__}: {exc}"}


def _raw_inputs(args, failures):
    from data_preparation.feature_inputs import PARSER_VERSION, parse_bgc
    from data_preparation.frozen_esm import FrozenESM2

    if args.gbk.is_file():
        paths = [args.gbk]
    elif args.gbk.is_dir():
        paths = sorted(path for path in args.gbk.rglob("*")
                       if path.is_file() and path.suffix.lower() in {".gb", ".gbk", ".genbank"})
    else:
        raise ValueError(f"GenBank path does not exist: {args.gbk}")
    if not paths:
        raise ValueError("No GenBank files found.")
    ids = [path.stem for path in paths]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate GenBank basenames would collide as BGC IDs.")
    encoder = FrozenESM2.from_checkpoint(
        args.esm_checkpoint, device=args.device, pooling=args.esm_pooling,
        max_residues=args.max_residues, batch_size=args.esm_batch_size,
        trusted=args.trusted_input,
    )
    embeddings, records = {}, {}
    for path in paths:
        try:
            record = parse_bgc(path)
            embeddings[record["bgc_id"]] = encoder.encode(record["tokens"])
            records[record["bgc_id"]] = {
                **{key: value for key, value in record.items() if key != "tokens"},
                "tokens": encoder.token_metadata(record["tokens"]),
            }
        except (ValueError, RuntimeError, KeyError, OSError) as exc:
            failures.append(_failure(path.stem, path, "gbk_to_esm", exc))
            if not args.skip_invalid:
                break
    provenance = {
        "format": CACHE_FORMAT,
        "parser": PARSER_VERSION,
        "order": "genomic CDS order; protein N-to-C domain order",
        "unannotated_regions": "omitted within proteins that have domains",
        "orientation_invariant": False,
        "esm": encoder.metadata,
        "input_count": len(paths),
    }
    return embeddings, records, provenance


def _versions():
    result = {"python": sys.version.split()[0], "torch": str(torch.__version__),
              "numpy": str(np.__version__)}
    for package in ("fair-esm", "rotary-embedding-torch", "biopython"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def run(args):
    if args.batch_size <= 0 or args.esm_batch_size <= 0 or args.max_residues <= 0:
        raise ValueError("Batch sizes and max-residues must be positive.")
    if args.route == "esm" and (args.checkpoint is not None or args.member != 0):
        raise ValueError("The ESM route does not use a BGC checkpoint or ensemble member.")
    if args.route != "esm" and args.checkpoint is None:
        raise ValueError("MAC/MAP extraction requires --checkpoint; random weights are not supported.")
    if args.gbk is not None and args.esm_checkpoint is None:
        raise ValueError("GBK input requires --esm-checkpoint.")
    if args.esm_cache is not None and (args.save_cache is not None or args.esm_checkpoint is not None):
        raise ValueError("--save-cache and --esm-checkpoint are only used with --gbk.")
    if args.esm_cache is not None and (args.esm_pooling != "bos" or args.max_residues != 1021):
        raise ValueError("Existing cache pooling/windowing cannot be changed; regenerate from --gbk.")
    if args.member < 0:
        raise ValueError("--member must be nonnegative.")
    if args.save_cache is not None and args.save_cache.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("--save-cache must use .pt or .pth (a torch tensor cache).")
    if args.save_cache is not None and args.save_cache.exists():
        raise ValueError(f"Cache output already exists: {args.save_cache}")
    if args.save_cache is not None and args.save_cache.resolve() in {
        (args.output / name).resolve() for name in OUTPUT_FILES
    }:
        raise ValueError("--save-cache must not collide with a feature export artifact.")
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in OUTPUT_FILES):
        raise ValueError("Output artifacts already exist; choose a new --output directory.")

    failures = []
    upstream_failures = []
    try:
        extractor = None
        if args.route != "esm":
            extractor = BGCFeatureExtractor.from_checkpoint(
                args.checkpoint, route=args.route, member=args.member,
                device=args.device, trusted=args.trusted_input,
            )
        if args.esm_cache is not None:
            embeddings, records, provenance = load_feature_cache(
                args.esm_cache, trusted=args.trusted_input
            )
        else:
            embeddings, records, provenance = _raw_inputs(args, failures)
        for failure in provenance.get("exclusions", []):
            if not isinstance(failure, dict) or not all(
                key in failure for key in ("bgc_id", "source", "stage", "error")
            ):
                raise ValueError("Malformed upstream exclusion in cache provenance.")
            upstream_failures.append({
                key: str(failure[key]) for key in ("bgc_id", "source", "stage", "error")
            })
            upstream_failures[-1]["stage"] = "upstream/" + upstream_failures[-1]["stage"]
        if failures and not args.skip_invalid:
            raise ValueError(f"Input failed for {failures[-1]['bgc_id']}; see failures.tsv.")
        prepared = {}
        expected_dim = extractor.input_dim if extractor is not None else None
        for bgc_id in sorted(embeddings):
            try:
                tensor = token_tensor(embeddings[bgc_id], expected_dim=expected_dim)
                if expected_dim is None:
                    expected_dim = tensor.shape[1]
                prepared[bgc_id] = tensor
            except (ValueError, RuntimeError, TypeError) as exc:
                failures.append(_failure(bgc_id, args.esm_cache or args.gbk, "cache_validation", exc))
        if failures and not args.skip_invalid:
            raise ValueError("Invalid inputs; see failures.tsv. Use --skip-invalid to export valid BGCs.")
        if not prepared:
            raise ValueError("No valid BGCs remain.")
        if extractor is None:
            extractor = BGCFeatureExtractor(esm_size=expected_dim).to(args.device)
        ids, vectors = [], []
        ordered_ids = sorted(prepared)
        for offset in range(0, len(ordered_ids), args.batch_size):
            batch_ids = ordered_ids[offset:offset + args.batch_size]
            batch = [prepared[key] for key in batch_ids]
            padded = pad_sequence(batch, batch_first=True).to(args.device)
            lengths = torch.tensor([len(item) for item in batch], device=args.device)
            try:
                result = extractor(padded, lengths).cpu()
                ids.extend(batch_ids)
                vectors.extend(result.unbind(0))
            except (ValueError, RuntimeError) as exc:
                # Isolate a bad sample without silently discarding the whole batch.
                for bgc_id, tensor in zip(batch_ids, batch):
                    try:
                        result = extractor(
                            tensor[None].to(args.device),
                            torch.tensor([len(tensor)], device=args.device),
                        ).cpu()[0]
                        ids.append(bgc_id)
                        vectors.append(result)
                    except (ValueError, RuntimeError) as sample_exc:
                        failures.append(_failure(bgc_id, args.esm_cache or args.gbk,
                                                 "bgc_encoding", sample_exc))
                if failures and not args.skip_invalid:
                    raise ValueError("BGC encoding failed; see failures.tsv.") from exc
        if not vectors:
            raise ValueError("No valid BGC vectors remain.")
        array = torch.stack(vectors).numpy().astype(np.float32, copy=False)
        metadata = {
            "format": "bgc_vectors_v1",
            "status": "complete_with_exclusions" if failures or upstream_failures else "complete",
            "route": args.route,
            "count": len(ids),
            "dimension": int(array.shape[1]),
            "dtype": "float32",
            "pooling": "masked_mean",
            "normalization": "l2",
            "similarity": "cosine equals dot product of exported vectors",
            "device": args.device,
            "batch_size": args.batch_size,
            "extractor": extractor.metadata,
            "input": provenance,
            "records": {key: records.get(key, {}) for key in ids},
            "excluded_count": len(failures),
            "upstream_excluded_count": len(upstream_failures),
            "versions": _versions(),
            "source_sha256": {
                str(path.relative_to(Path(__file__).resolve().parents[1])): file_sha256(path)
                for path in (
                    Path(__file__).resolve(),
                    Path(__file__).resolve().parents[1] / "model" / "features.py",
                    Path(__file__).resolve().parents[1] / "model" / "encoders.py",
                    Path(__file__).resolve().parents[1] / "data_preparation" / "feature_cache.py",
                    Path(__file__).resolve().parents[1] / "data_preparation" / "feature_inputs.py",
                    Path(__file__).resolve().parents[1] / "data_preparation" / "frozen_esm.py",
                )
            },
        }
        if args.save_cache is not None:
            args.save_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"format": CACHE_FORMAT, "embeddings": prepared,
                        "records": records,
                        "provenance": {**provenance, "exclusions": [
                            failure for failure in failures
                            if failure["stage"] in {"gbk_to_esm", "cache_validation"}
                        ]}},
                       args.save_cache)
            metadata["saved_cache"] = {
                "path": str(args.save_cache.resolve()), "sha256": file_sha256(args.save_cache)
            }
        np.save(args.output / "embeddings.npy", array, allow_pickle=False)
        with (args.output / "bgc_ids.tsv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(("row", "bgc_id", "token_count"))
            writer.writerows((row, key, len(prepared[key])) for row, key in enumerate(ids))
        _json_write(args.output / "metadata.json", metadata)
        _write_failures(args.output / "failures.tsv", upstream_failures + failures)
        return metadata
    except Exception as exc:
        _write_failures(args.output / "failures.tsv", upstream_failures + failures)
        _json_write(args.output / "metadata.json", {
            "format": "bgc_vectors_v1", "status": "failed",
            "route": args.route, "error": f"{type(exc).__name__}: {exc}",
            "excluded_count": len(failures),
            "upstream_excluded_count": len(upstream_failures),
        })
        raise


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        metadata = run(args)
    except (ValueError, RuntimeError, OSError, TypeError, KeyError) as exc:
        parser.exit(1, f"Feature extraction failed: {exc}\n")
    print(f"Saved {metadata['count']} BGC vectors ({metadata['dimension']} dimensions) to {args.output}")
    if metadata["excluded_count"]:
        print(f"Excluded {metadata['excluded_count']} inputs; see failures.tsv.")
    if metadata["upstream_excluded_count"]:
        print(f"Input cache records {metadata['upstream_excluded_count']} earlier exclusions; see failures.tsv.")


if __name__ == "__main__":
    main()
