# UMAP of frozen MIBiG features

This module reads an existing frozen feature export and produces two views of
the same two-dimensional UMAP coordinates: biosynthetic class combinations and
input token count. It does not run an encoder, train or fine-tune a model, or
cluster BGCs. Only feature vectors enter the UMAP fit; labels and token counts
are used for coloring afterward.

Run the commands below from the repository root in the configured project
environment.

## Additional dependency

```bash
python -m pip install -e '.[visualization]'
```

The optional `visualization` extra pins `umap-learn==0.5.12`, the only new
direct dependency relative to the project's environment. It also declares
NumPy, pandas and Matplotlib, which are already in `environment.yml` along
with scikit-learn, SciPy and tqdm. Installing UMAP can also add its dependencies
`numba`, `llvmlite` and `pynndescent`; the package resolver may update existing
dependencies if needed. The existing `environment.yml` is unchanged.
The command uses Matplotlib's noninteractive `Agg` backend and works on a
server without a display. Each output records the installed package versions.

## 1. Extract features once

Use only the author's downloaded MIBiG cache and, for MAC/MAP, the corresponding
trained checkpoint. Choose any one route:

```bash
python -m experiment.extract_features \
  --esm-cache data/BGC_4.0/Esm2_rep_mibig.pth \
  --route esm \
  --output outputs/features_esm
```

```bash
python -m experiment.extract_features \
  --esm-cache data/BGC_4.0/Esm2_rep_mibig.pth \
  --route mac \
  --checkpoint ckpt/BGC_MAC/MAC_default/MAC_default.ckpt \
  --member 0 --trusted-input \
  --output outputs/features_mac
```

```bash
python -m experiment.extract_features \
  --esm-cache data/BGC_4.0/Esm2_rep_mibig.pth \
  --route map \
  --checkpoint ckpt/BGC_MAP/MAP_default/MAP_default.ckpt \
  --member 0 --trusted-input \
  --output outputs/features_map
```

These examples use CPU inference. Add `--device cuda:0` to use a GPU for MAC/MAP
extraction. `--trusted-input` allows loading legacy author checkpoint objects;
use it only for files from a trusted source. Each MAC/MAP export loads one
ensemble member and keeps its parameters frozen. See
[Frozen feature extraction](frozen-features.md) for the existing interface.

## 2. Visualize an export

```bash
python -m visualization.umap_features \
  --features-dir outputs/features_mac \
  --labels data/BGC_4.0/MAC_metadata.pkl \
  --metric cosine \
  --n-neighbors 15 \
  --min-dist 0.1 \
  --random-state 42 \
  --output outputs/umap_mac
```

The labels path and all four UMAP settings above are defaults. The label file
is a pandas pickle supplied by the author and must be trusted. For ESM or MAP,
change `--features-dir` and `--output`; vector dimension is read from the export
without assumptions about encoder size.

Rerun this visualization command with a new output directory to change UMAP
or plotting options. The encoder does not run again; UMAP is refitted from the
exported vectors. Plotting options are `--point-size` (default 12), `--alpha`
(0.8) and `--dpi` (300). Existing files with any of the six output filenames
below are never overwritten. Other files in the output directory are allowed.

The input must contain at least four BGCs, with `2 <= n_neighbors < BGC_count`;
choose a smaller neighbor count when visualizing fewer than 16 entries.
`min_dist` must be between 0 and 1 because UMAP's spread is fixed at 1. The
same fixed random seed and single-threaded UMAP settings support repeatability;
different package versions or hardware can still affect the result. Separate
fits for different encoders do not have aligned axes.

## Alignment and labels

The input directory must contain `embeddings.npy`, `bgc_ids.tsv` and
`metadata.json` from a completed feature extraction. The TSV's explicit `row`
column is used to select embedding rows, so reordered TSV lines remain valid.
Rows must cover the matrix exactly once. Duplicate or blank BGC IDs, invalid
rows, inconsistent dimensions, nonfinite or zero vectors, and invalid token
counts stop the run with a clear error rather than being silently discarded.

Labels are joined using `bgc_id` and `BGC_number`. Duplicate label IDs are
rejected. Unmatched IDs and empty labels remain in the plot as gray `Unknown`
points; their count and IDs are reported. Full label sets are retained and
sorted into a stable combination: `['PKS', 'NRPS']` becomes `NRPS+PKS`, not
just one of those classes. `ribosomal` is displayed as `RiPP`. Six-element
binary multi-hot labels are also accepted in the author's order: NRPS, Other,
PKS, RiPP, Saccharide, Terpene. `Other` remains distinct from missing labels.
The legend reports counts for each combination. Unrecognized class names are
rejected rather than converted to `Unknown`.

The token-count plot uses a continuous color scale and the same coordinates
and axis limits as the class plot. `token_count` counts input domain/protein
tokens, not amino-acid residues or necessarily genes.

## Outputs

| File | Content |
| --- | --- |
| `umap_classes.png`, `umap_classes.pdf` | Scatter plot colored by complete class combination |
| `umap_token_count.png`, `umap_token_count.pdf` | The same points colored by input token count |
| `umap_coordinates.tsv` | Row, BGC ID, full labels, label count, token count and two coordinates |
| `umap_metadata.json` | Run parameters, input hashes, source encoder metadata, package versions, class colors and missing-label audit |

UMAP is an exploratory projection: apparent separation does not establish
retrieval quality. In particular, MAC/MAP weights were trained on author data
and may already have seen these MIBiG BGCs or class labels. Using class labels
only for coloring does not remove that pretraining overlap.
