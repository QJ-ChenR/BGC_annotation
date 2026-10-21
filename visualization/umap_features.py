"""Plot exported frozen BGC vectors with UMAP; labels only control coloring."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Matches the multi-hot column order in experiment/train.py; no training import.
CLASS_NAMES = ("NRPS", "Other", "PKS", "RiPP", "Saccharide", "Terpene")
ALIASES = {name.lower(): name for name in CLASS_NAMES}
ALIASES.update({"ribosomal": "RiPP", "ripps": "RiPP"})
OUTPUT_FILES = ("umap_classes.png", "umap_classes.pdf", "umap_token_count.png",
                "umap_token_count.pdf", "umap_coordinates.tsv", "umap_metadata.json")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True,
                        help="Export from experiment.extract_features (any encoder route).")
    parser.add_argument("--labels", type=Path, default=Path("data/BGC_4.0/MAC_metadata.pkl"),
                        help="Author-provided, trusted pandas pickle with BGC_number/biosyn_class.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--point-size", type=float, default=12)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def _validate_ids(values, source):
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise ValueError(f"{source}: BGC IDs must be nonempty strings.")
    ids = values.str.strip()
    duplicates = ids[ids.duplicated(keep=False)].unique().tolist()
    if duplicates:
        raise ValueError(f"{source}: duplicate BGC IDs: {duplicates[:10]}")
    return ids


def _integer_column(values, name):
    if not values.str.fullmatch(r"[0-9]+").all():
        raise ValueError(f"{name} must contain nonnegative integers.")
    try:
        return values.astype(np.int64)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} contains an out-of-range integer.") from exc


def load_features(directory):
    directory = Path(directory)
    with (directory / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict) or metadata.get("format") != "bgc_vectors_v1":
        raise ValueError("Expected metadata format bgc_vectors_v1.")
    if metadata.get("status") not in {"complete", "complete_with_exclusions"}:
        raise ValueError("Feature export is not complete; regenerate it before plotting.")
    vectors = np.load(directory / "embeddings.npy", allow_pickle=False)
    if vectors.ndim != 2 or 0 in vectors.shape or vectors.dtype.kind != "f":
        raise ValueError("embeddings.npy must be a nonempty real floating-point [N, D] matrix.")
    if metadata.get("count") != vectors.shape[0] or metadata.get("dimension") != vectors.shape[1]:
        raise ValueError("Feature count/dimension disagrees with metadata.json.")
    rows = pd.read_csv(directory / "bgc_ids.tsv", sep="\t", dtype=str, keep_default_na=False)
    if not {"row", "bgc_id", "token_count"}.issubset(rows.columns):
        raise ValueError("bgc_ids.tsv requires row, bgc_id and token_count columns.")
    rows = rows[["row", "bgc_id", "token_count"]].copy()
    rows["bgc_id"] = _validate_ids(rows["bgc_id"], "bgc_ids.tsv")
    for name in ("row", "token_count"):
        rows[name] = _integer_column(rows[name], name)
    rows = rows.sort_values("row").reset_index(drop=True)
    if not np.array_equal(rows["row"].to_numpy(), np.arange(len(vectors))):
        raise ValueError("row must contain every embeddings.npy row exactly once, from 0 to N-1.")
    if (rows["token_count"] <= 0).any():
        raise ValueError("token_count must be positive for every BGC.")
    invalid = ~np.isfinite(vectors).all(axis=1) | ~np.any(vectors != 0, axis=1)
    if invalid.any():
        ids = rows.loc[invalid, "bgc_id"].tolist()
        raise ValueError(f"Nonfinite or zero BGC vectors ({len(ids)}): {ids[:10]}")
    return vectors, rows, metadata


def normalize_labels(value):
    """Keep every class, deduplicate, and use the author's six-class order."""
    if isinstance(value, np.ndarray) and value.ndim != 1:
        raise ValueError("biosyn_class arrays must be one-dimensional.")
    if isinstance(value, str):
        value = re.split(r"[+/,;]", value) if value.strip() else []
    elif not isinstance(value, (list, tuple, np.ndarray)):
        if value is None or value is pd.NA or (np.isscalar(value) and pd.isna(value)):
            return ()
        raise ValueError(f"Invalid biosyn_class value: {value!r}")
    values = list(value)
    if not values:
        return ()
    if all(isinstance(item, (int, float, bool, np.number, np.bool_)) for item in values):
        if len(values) != len(CLASS_NAMES) or any(item not in (0, 1) for item in values):
            raise ValueError("Numeric biosyn_class must be a six-element binary multi-hot vector.")
        return tuple(name for name, present in zip(CLASS_NAMES, values) if present)
    labels = set()
    for item in values:
        if not isinstance(item, str) or item.strip().lower() not in ALIASES:
            raise ValueError(f"Unrecognized biosyn_class label: {item!r}")
        labels.add(ALIASES[item.strip().lower()])
    return tuple(name for name in CLASS_NAMES if name in labels)


def load_labels(path):
    # The input is the author's local metadata artifact, not an arbitrary upload.
    metadata = pd.read_pickle(path)
    if not isinstance(metadata, pd.DataFrame) or not {"BGC_number", "biosyn_class"}.issubset(metadata.columns):
        raise ValueError("Label metadata requires a DataFrame with BGC_number and biosyn_class.")
    ids = _validate_ids(metadata["BGC_number"], "label metadata")
    labels = {}
    for bgc_id, value in zip(ids, metadata["biosyn_class"]):
        try:
            labels[bgc_id] = normalize_labels(value)
        except ValueError as exc:
            raise ValueError(f"{bgc_id}: {exc}") from exc
    return labels


def fit_umap(vectors, *, metric, n_neighbors, min_dist, random_state):
    import umap

    # Labels and token counts never enter the reducer, including as target y.
    return umap.UMAP(n_components=2, metric=metric, n_neighbors=n_neighbors,
                     min_dist=min_dist, random_state=random_state, n_jobs=1).fit_transform(vectors)


def _category_colors(categories):
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    palette = [to_hex(color) for name in ("tab20", "tab20b", "tab20c", "Set1")
               for color in colormaps[name].colors]
    known = sorted(set(categories) - {"Unknown"})
    colors = {label: palette[index] for index, label in enumerate(known)}
    if "Unknown" in categories:
        colors["Unknown"] = "#999999"
    return colors


def plot_outputs(table, colors, output, point_size, alpha, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coords = table[["umap_1", "umap_2"]].to_numpy()
    lower, upper = coords.min(axis=0), coords.max(axis=0)
    margin = np.maximum(upper - lower, 1) * 0.05
    for mode in ("classes", "token_count"):
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_axes((0.08, 0.1, 0.50, 0.82))
        try:
            if mode == "classes":
                for label, color in colors.items():
                    subset = table[table["labels"] == label]
                    ax.scatter(subset["umap_1"], subset["umap_2"], c=color,
                               s=point_size, alpha=alpha, linewidths=0,
                               label=f"{label} ({len(subset)})")
                ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False,
                          ncol=2 if len(colors) > 16 else 1, fontsize=8,
                          title="Biosynthetic class combination (count)")
                title = "BGC classes"
            else:
                points = ax.scatter(coords[:, 0], coords[:, 1], c=table["token_count"],
                                    cmap="viridis", s=point_size, alpha=alpha, linewidths=0)
                bar = fig.colorbar(points, cax=fig.add_axes((0.62, 0.2, 0.025, 0.6)))
                bar.set_label("Token count (input domains / whole-protein fallbacks)")
                title = "BGC token count"
            ax.set(xlim=(lower[0] - margin[0], upper[0] + margin[0]),
                   ylim=(lower[1] - margin[1], upper[1] + margin[1]),
                   xlabel="UMAP 1", ylabel="UMAP 2", title=f"{title} (N={len(table)})")
            for extension in ("png", "pdf"):
                fig.savefig(Path(output) / f"umap_{mode}.{extension}", dpi=dpi)
        finally:
            plt.close(fig)


def _sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _versions():
    result = {"python": sys.version.split()[0]}
    for package in ("numpy", "pandas", "matplotlib", "umap-learn", "scikit-learn",
                    "scipy", "numba", "pynndescent", "llvmlite"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def run(args):
    vectors, rows, source = load_features(args.features_dir)
    if len(vectors) < 4 or not 2 <= args.n_neighbors < len(vectors):
        raise ValueError("UMAP requires at least 4 BGCs and 2 <= n-neighbors < BGC count.")
    if not 0 <= args.min_dist <= 1:
        raise ValueError("min-dist must be between 0 and 1 (UMAP spread is fixed at 1).")
    if not 0 <= args.random_state < 2**32:
        raise ValueError("random-state must be between 0 and 2**32 - 1.")
    if not np.isfinite(args.point_size) or args.point_size <= 0 or not 0 < args.alpha <= 1 or args.dpi <= 0:
        raise ValueError("point-size and dpi must be positive; alpha must be in (0, 1].")
    if any((args.output / name).exists() for name in OUTPUT_FILES):
        raise ValueError("Output artifacts already exist; choose a new --output directory.")
    labels = load_labels(args.labels)
    all_labels = [labels.get(bgc_id, ()) for bgc_id in rows["bgc_id"]]
    table = rows.copy()
    table["labels"] = ["+".join(value) if value else "Unknown" for value in all_labels]
    table["label_count"] = [len(value) for value in all_labels]
    unknown_ids = table.loc[table["label_count"] == 0, "bgc_id"].tolist()
    print(f"Loaded {len(vectors)} BGCs, dimension {vectors.shape[1]}; Unknown labels: {len(unknown_ids)}.",
          flush=True)
    parameters = {name: getattr(args, name) for name in
                  ("metric", "n_neighbors", "min_dist", "random_state")}
    coords = np.asarray(fit_umap(vectors, **parameters))
    if coords.shape != (len(vectors), 2) or not np.isfinite(coords).all():
        raise ValueError("UMAP returned invalid coordinates; no BGCs will be silently omitted.")
    table[["umap_1", "umap_2"]] = coords
    table = table[["row", "bgc_id", "labels", "label_count", "token_count", "umap_1", "umap_2"]]
    colors = _category_colors(table["labels"].tolist())
    input_paths = [args.features_dir / name for name in ("embeddings.npy", "bgc_ids.tsv", "metadata.json")]
    metadata = {
        "format": "bgc_umap_v1", "count": len(table), "feature_dimension": int(vectors.shape[1]),
        "features_dir": str(args.features_dir.resolve()), "labels_file": str(args.labels.resolve()),
        "feature_source": {key: value for key, value in source.items() if key != "records"},
        "input_sha256": {str(path.resolve()): _sha256(path) for path in [*input_paths, args.labels]},
        "script_sha256": _sha256(Path(__file__)),
        "umap_parameters": {**parameters, "n_components": 2, "n_jobs": 1, "spread": 1.0,
                            "init": "spectral"},
        "labels_used_for_fit": False, "token_count_used_for_fit": False,
        "unknown_count": len(unknown_ids), "unknown_ids": unknown_ids,
        "missing_label_ids": [bgc_id for bgc_id in rows["bgc_id"] if bgc_id not in labels],
        "unused_label_count": len(set(labels) - set(rows["bgc_id"])),
        "category_counts": {label: int((table["labels"] == label).sum()) for label in colors},
        "category_colors": colors, "multi_label_count": int((table["label_count"] > 1).sum()),
        "plot_parameters": {"point_size": args.point_size, "alpha": args.alpha, "dpi": args.dpi},
        "versions": _versions(), "output_files": list(OUTPUT_FILES),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    plot_outputs(table, colors, args.output, args.point_size, args.alpha, args.dpi)
    table.to_csv(args.output / "umap_coordinates.tsv", sep="\t", index=False, float_format="%.9g")
    with (args.output / "umap_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(f"Saved plots, coordinates and parameters to {args.output}", flush=True)
    return metadata


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
