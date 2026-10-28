"""Validate full-BGC inference against actual backbone calls and windowed inference.

This is an implementation/resource diagnostic, not a biological retrieval score.
Outputs are validation artifacts, not substitutes for a complete feature export.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from data_preparation.dna_inputs import read_dna
from data_preparation.feature_cache import file_sha256
from experiment.extract_dna_features import _select_paths, _versions, _write_json


ARTIFACTS = (
    "summary.json", "results.tsv", "inventory.tsv", "failures.tsv",
    "full_embeddings.npy", "full_bgc_ids.tsv",
    "windowed_embeddings.npy", "windowed_bgc_ids.tsv",
)
RESULT_FIELDS = (
    "bgc_id", "base_count", "token_count", "full_status", "windowed_status",
    "validation_passed", "full_forward_calls", "full_actual_input_tokens",
    "full_seconds", "full_peak_allocated_bytes", "full_peak_reserved_bytes",
    "windowed_seconds", "windowed_peak_allocated_bytes", "windowed_peak_reserved_bytes",
    "reference_pooling_matches", "single_window_matches", "repeat_matches",
    "full_windowed_cosine", "full_windowed_max_abs_difference", "full_error", "windowed_error", "error",
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gbk", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--attention-backend", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--max-tokens", type=int, help="Optional full-input resource cap including CLS/SEP; exceeding it fails, never truncates.")
    parser.add_argument("--window-max-tokens", type=int, help="Windowed comparison size including CLS/SEP; defaults to checkpoint config.")
    parser.add_argument("--sample-count", type=int, default=5, help="Default sample count at evenly spaced ranks of BPE length, including shortest/longest when >=2.")
    parser.add_argument("--all", action="store_true", help="Validate all selected inputs instead of length sampling.")
    parser.add_argument("--ids-from", type=Path, help="Explicit bgc_ids.tsv cohort; validates every listed ID instead of sampling.")
    parser.add_argument("--exclude-id", action="append", default=[])
    parser.add_argument("--repeat-all", action="store_true", help="Repeat all full encodings; by default repeat only inputs fitting one comparison window.")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    return parser


def _table(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parameter_sha256(model):
    """Fingerprint parameter bytes after loading/casting, including bf16 safely."""
    import torch

    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(json.dumps([name, str(parameter.dtype), list(parameter.shape)]).encode())
        raw = parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def _frozen(model):
    return not any(module.training for module in model.modules()) and all(
        not parameter.requires_grad and parameter.grad is None for parameter in model.parameters()
    )


def _measure(device, operation):
    """Return ordinary failures and OOM as explicit measured records."""
    import torch

    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    value = None
    measurement = {"status": "passed", "error": None}
    try:
        value = operation()
        if cuda:
            torch.cuda.synchronize(device)
    except Exception as exc:
        value = None  # A CUDA synchronization failure also invalidates a returned result.
        is_oom = isinstance(exc, (torch.OutOfMemoryError, MemoryError)) or (
            isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
        )
        measurement["status"] = "oom" if is_oom else "failed"
        measurement["error"] = f"{type(exc).__name__}: {exc}"
    measurement["seconds"] = time.perf_counter() - started
    measurement["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device)) if cuda else None
    measurement["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device)) if cuda else None
    if measurement["status"] != "passed":
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
    return value, measurement


def _checked_vector(vector, dimension):
    import torch

    if vector.shape != (dimension,) or vector.requires_grad or not torch.isfinite(vector).all():
        raise ValueError("Invalid vector shape, gradient state, or finite values.")
    array = vector.detach().cpu().float().numpy()
    if not np.isclose(np.linalg.norm(array), 1.0, atol=1e-5):
        raise ValueError("Vector is not unit normalized.")
    return array


def _audited_full(encoder, sequence, ids, atol, rtol):
    """Observe the real backbone input/output independently of encode metadata."""
    import torch

    observed, references = [], []
    expected = [encoder.special_ids["cls"], *ids, encoder.special_ids["sep"]]

    def before(module, positional, kwargs):
        actual = kwargs["input_ids"]
        observed.append({"shape": list(actual.shape), "input_tokens": actual.numel()})
        if len(observed) != 1 or actual.shape != (1, len(expected)):
            raise ValueError("Full mode did not execute one complete BGC in a single forward.")
        if actual.detach().cpu().tolist()[0] != expected:
            raise ValueError("Backbone input IDs differ from complete BPE content plus CLS/SEP.")
        mask = kwargs["attention_mask"]
        if mask.shape != actual.shape or not bool(mask.eq(1).all()):
            raise ValueError("Full input contains masked or padded content.")
        if not torch.is_inference_mode_enabled() or torch.is_grad_enabled() or not _frozen(module):
            raise ValueError("Backbone is not in frozen inference mode.")

    def after(module, positional, kwargs, output):
        hidden = output.last_hidden_state
        if hidden.shape != (1, len(expected), encoder.embedding_dim) or hidden.requires_grad:
            raise ValueError("Unexpected backbone hidden-state shape or gradient state.")
        # Independently select the content slice; UNK remains a content token.
        mean = hidden[0, 1:-1].float().mean(dim=0).cpu()
        reference = mean / torch.linalg.vector_norm(mean)
        references.append(reference)

    pre = encoder.model.register_forward_pre_hook(before, with_kwargs=True)
    post = encoder.model.register_forward_hook(after, with_kwargs=True)
    try:
        vector, metadata = encoder.encode(sequence)
    finally:
        pre.remove()
        post.remove()
    if len(observed) != 1 or len(references) != 1:
        raise ValueError("Full encoding did not produce exactly one observed backbone result.")
    array = _checked_vector(vector, encoder.embedding_dim)
    if not torch.allclose(vector.cpu(), references[0], atol=atol, rtol=rtol):
        raise ValueError("Exported vector differs from independently pooled full hidden states.")
    if metadata["token_count"] != len(ids) or metadata["window_count"] != 1:
        raise ValueError("Full-mode coverage metadata is inconsistent.")
    expected_coverage = [{"token_start": 0, "token_end": len(ids), "base_start": 0, "base_end": len(sequence)}]
    if metadata["windows"] != expected_coverage:
        raise ValueError("Full-mode token/base interval does not cover the entire sequence.")
    return array, metadata, observed


def _select_samples(scanned, count, use_all):
    ordered = sorted(scanned, key=lambda item: (item["token_count"], item["bgc_id"]))
    if use_all:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, min(count, len(ordered)), dtype=int)
    return [ordered[index] for index in indices]


def run(args):
    if args.sample_count < 1 or any(
        value is not None and value <= 2 for value in (args.max_tokens, args.window_max_tokens)
    ):
        raise ValueError("sample-count must be positive and token limits must exceed 2.")
    if not all(np.isfinite(v) and v >= 0 for v in (args.atol, args.rtol)):
        raise ValueError("Tolerances must be finite and nonnegative.")
    paths = _select_paths(args)
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ARTIFACTS):
        raise ValueError("Validation artifacts already exist; choose a new --output directory.")
    results, inventory, failures = [], [], []
    vectors = {mode: {} for mode in ("full", "windowed")}
    summary = {
        "format": "dna_full_validation_v1", "status": "running",
        "scope": "Implementation and resource validation; does not establish retrieval improvement or feasibility for unsampled inputs.",
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "versions": _versions(), "input_count": len(paths), "results": results,
        "sampling": "all explicit IDs" if args.ids_from else "all inputs" if args.all else "evenly spaced ranks of content BPE length",
        "memory_units": "bytes; CUDA peaks include resident model and validation hooks; reserved memory also includes allocator cache from earlier phases; CPU memory not measured",
        "comparison": "Same backbone, tokenizer, dtype, attention backend and token-mean/L2 readout; only context policy differs.",
        "failures": failures,
    }
    encoder = None

    def save():
        summary["selected_count"] = sum(bool(item.get("selected")) for item in inventory)
        summary["completed_count"] = len(results)
        summary["passed_count"] = sum(bool(row.get("validation_passed")) for row in results)
        _write_json(args.output / "summary.json", summary)
        _table(args.output / "results.tsv", RESULT_FIELDS, results)
        _table(args.output / "inventory.tsv", (
            "bgc_id", "source", "base_count", "token_count", "selected", "status", "reason",
            "source_sha256", "sequence_sha256",
        ), inventory)
        _table(args.output / "failures.tsv", ("bgc_id", "stage", "status", "error"), failures)
        for mode, values in vectors.items():
            keys = sorted(values)
            dimension = encoder.embedding_dim if encoder is not None else 0
            array = np.stack([values[key] for key in keys]) if keys else np.empty((0, dimension), dtype=np.float32)
            np.save(args.output / f"{mode}_embeddings.npy", array, allow_pickle=False)
            _table(args.output / f"{mode}_bgc_ids.tsv", ("row", "bgc_id"), [
                {"row": index, "bgc_id": key} for index, key in enumerate(keys)
            ])

    def failure(bid, stage, measure):
        failures.append({"bgc_id": bid, "stage": stage, "status": measure["status"], "error": measure["error"]})

    save()
    try:
        import torch
        from data_preparation.frozen_dna import FrozenDNA

        encoder = FrozenDNA.from_checkpoint(
            args.checkpoint, tokenizer_path=args.tokenizer, config_path=args.config,
            device=args.device, dtype=args.dtype, attention_backend=args.attention_backend,
            sequence_mode="full", max_tokens=args.max_tokens, batch_size=1,
        )
        comparison = FrozenDNA(
            encoder.model, encoder.tokenizer, device=args.device, dtype=args.dtype,
            attention_backend=args.attention_backend, sequence_mode="windowed",
            max_tokens=args.window_max_tokens, batch_size=1, provenance=encoder.provenance,
        )
        summary["extractor"] = encoder.metadata
        summary["windowed_extractor"] = comparison.metadata
        if encoder.device.type == "cuda":
            props = torch.cuda.get_device_properties(encoder.device)
            summary["gpu"] = {"name": props.name, "total_memory_bytes": props.total_memory}
        summary["frozen_before"] = _frozen(encoder.model)
        summary["parameters_sha256_before"] = parameter_sha256(encoder.model)
        if not summary["frozen_before"]:
            raise ValueError("Model is not frozen before validation.")
        scanned = []
        for path in paths:
            item = {"bgc_id": path.stem, "source": str(path.resolve()), "selected": False}
            inventory.append(item)
            if path.stem in args.exclude_id:
                item.update(status="excluded", reason="Explicit --exclude-id")
                continue
            try:
                sequence, record = read_dna(path)
                ids, _ = encoder._tokenize(sequence)
                item.update(record, token_count=len(ids), status="eligible", reason="")
                scanned.append(item)
            except Exception as exc:
                item.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
                failure(path.stem, "scan", {"status": "failed", "error": item["reason"]})
        if not scanned:
            raise ValueError("No valid nonexcluded BGC inputs remain.")
        selected = _select_samples(scanned, args.sample_count, args.all or args.ids_from is not None)
        for item in scanned:
            item["status"] = "not_sampled"
            item["reason"] = "Outside deterministic length sample"
        for item in selected:
            item.update(selected=True, status="selected", reason="")
        save()
        for index, item in enumerate(selected):
            bid = item["bgc_id"]
            row = {"bgc_id": bid, "base_count": item["base_count"], "token_count": item["token_count"],
                   "full_status": "not_run", "windowed_status": "not_run", "validation_passed": False}
            results.append(row)
            try:
                sequence, reread = read_dna(item["source"])
                if reread["source_sha256"] != item["source_sha256"]:
                    raise ValueError("GenBank file changed after selection scan.")
                ids, _ = encoder._tokenize(sequence)
                full, measurement = _measure(encoder.device, lambda: _audited_full(encoder, sequence, ids, args.atol, args.rtol))
                row.update({f"full_{key}": value for key, value in measurement.items()})
                if full is None:
                    failure(bid, "full", measurement)
                else:
                    full_vector, full_meta, calls = full
                    vectors["full"][bid] = full_vector
                    row.update(full_forward_calls=len(calls), full_actual_input_tokens=calls[0]["input_tokens"],
                               reference_pooling_matches=True, full_metadata=full_meta)
                windowed, measured_window = _measure(encoder.device, lambda: comparison.encode(sequence))
                row.update({f"windowed_{key}": value for key, value in measured_window.items()})
                if windowed is None:
                    failure(bid, "windowed", measured_window)
                else:
                    window_vector = _checked_vector(windowed[0], encoder.embedding_dim)
                    vectors["windowed"][bid] = window_vector
                    row["windowed_metadata"] = windowed[1]
                if full is not None and windowed is not None:
                    row["full_windowed_cosine"] = float(np.clip(np.dot(full_vector, window_vector), -1, 1))
                    row["full_windowed_max_abs_difference"] = float(np.max(np.abs(full_vector - window_vector)))
                    single_window = len(ids) <= comparison.max_content_tokens
                    if single_window:
                        row["single_window_matches"] = bool(np.allclose(full_vector, window_vector, atol=args.atol, rtol=args.rtol))
                        if not row["single_window_matches"]:
                            raise ValueError("Single-window and full vectors differ beyond tolerance.")
                    if single_window or args.repeat_all:
                        repeated, repeated_measure = _measure(encoder.device, lambda: encoder.encode(sequence))
                        row["repeat_measurement"] = repeated_measure
                        if repeated is None:
                            failure(bid, "repeat_full", repeated_measure)
                            raise ValueError("Repeated full encoding failed; see repeat_measurement.")
                        row["repeat_matches"] = bool(np.allclose(full_vector, _checked_vector(repeated[0], encoder.embedding_dim), atol=args.atol, rtol=args.rtol))
                        if not row["repeat_matches"]:
                            raise ValueError("Repeated full encoding differs beyond tolerance.")
                    row["validation_passed"] = True
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                failure(bid, "validation", {"status": "failed", "error": row["error"]})
            item["status"] = "passed" if row["validation_passed"] else "failed"
            save()
            print(f"DNA validation {index + 1}/{len(selected)}: {bid} full={row['full_status']} windowed={row['windowed_status']} passed={row['validation_passed']}", file=sys.stderr, flush=True)
        summary["parameters_sha256_after"] = parameter_sha256(encoder.model)
        summary["parameters_unchanged"] = summary["parameters_sha256_before"] == summary["parameters_sha256_after"]
        summary["frozen_after"] = _frozen(encoder.model)
        if not summary["parameters_unchanged"] or not summary["frozen_after"]:
            raise ValueError("Backbone parameters or frozen state changed during validation.")
        summary["status"] = "passed" if not failures and all(row["validation_passed"] for row in results) else "failed"
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        failure("", "run", {"status": "failed", "error": summary["error"]})
    finally:
        root = Path(__file__).resolve().parents[1]
        summary["source_sha256"] = {name: file_sha256(root / name) for name in (
            "experiment/validate_dna_encoding.py", "experiment/extract_dna_features.py",
            "data_preparation/frozen_dna.py", "data_preparation/dna_inputs.py",
        )}
        save()
    return summary


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run(args)
    except (ValueError, RuntimeError, OSError, TypeError, KeyError, ImportError) as exc:
        parser.exit(1, f"DNA validation failed: {exc}\n")
    print(f"DNA validation {summary['status']}: {summary['passed_count']}/{summary['selected_count']} selected inputs passed; {args.output / 'summary.json'}")
    if summary["status"] != "passed":
        parser.exit(1, "One or more checks failed; see summary.json and failures.tsv. No windowed fallback was substituted for full encoding.\n")


if __name__ == "__main__":
    main()
