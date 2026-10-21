# Frozen BGC feature and UMAP assessment

Analysis date: 2026-10-21. These are descriptive comparisons of the supplied
server exports, not a held-out retrieval benchmark. No encoder, UMAP, clustering
algorithm or classifier was fitted during this analysis. The supplementary
plot reuses the saved coordinates.

The strongest supported conclusion is that **MAC has the best local agreement
with the supplied biosynthetic classes**. MAP usually improves on ESM in the
overall scores, but the improvement depends on class and neighborhood size.
The visual impression MAP > MAC > ESM does not describe class retrieval quality.
MAP's UMAP does preserve its own original neighborhoods better.

## Comparability and evaluation definition

All three exports contain the same 2,636 BGC IDs, row order and token counts,
with no exclusions or invalid vectors. Their recorded cache SHA256 is identical.
The UMAP inputs' recorded hashes match the local feature files and author label
file. UMAP parameters and versions are identical: cosine, 15 neighbors,
min_dist=0.1, random_state=42, two components and n_jobs=1. All methods use masked
mean and L2 normalization. ESM has 1,280 dimensions, MAC 256 and MAP 1,024.
MAC and MAP use member 0 of nine, not an ensemble.

There are 2,635 labeled BGCs, including 491 with multiple labels and 27 distinct
nonempty label combinations. BGC0002977 is the only Unknown. For label-based
metrics it is omitted from both queries and candidates; it remains in the
original UMAP and supplementary figure. Every query excludes itself. No other
BGC is omitted. Exact or closely related BGCs have not been homology-deduplicated.

Nearest neighbors are computed directly from original, normalized vectors by
cosine distance. Complete label combinations and constituent labels are retained.
For a query label set A and neighbor label set B:

- Exact match is 1 when A=B.
- Any overlap is 1 when A and B share at least one label.
- Jaccard is |A intersection B| / |A union B|.

Each score averages over neighbors and then queries. Random expectations sample
uniformly from all other labeled BGCs and account for the actual class imbalance.
They are expectations, not fitted or simulated models.

## Original-vector nearest neighbors

| Metric, k=15 | ESM | MAC | MAP | Random expectation |
| --- | ---: | ---: | ---: | ---: |
| Exact complete-label match | 56.12% | **66.30%** | 60.17% | 15.80% |
| At least one shared label | 76.84% | **84.26%** | 77.97% | 32.30% |
| Mean label Jaccard | 0.6625 | **0.7507** | 0.6885 | 0.2375 |
| Six-class macro Jaccard | 0.5966 | **0.7051** | 0.6136 | 0.1864 |

Six-class macro Jaccard first averages query scores among BGCs carrying each
class, then averages the six class means equally. Hybrid queries enter each of
their constituent classes. This also favors MAC, so the result is not just an
effect of the large PKS and NRPS categories.

MAC leads the main scores at k=5, 15 and 50. MAP versus ESM is less consistent:
at k=5, any-label overlap is 80.67% for MAP versus 81.31% for ESM, and macro
Jaccard is 0.6543 versus 0.6643. The overall k=15 MAP advantage over ESM should
not be interpreted as a universal improvement.

The following class-specific score uses all labeled candidates, including
hybrids: among queries carrying class C, what fraction of their 15 neighbors
also carry C? Thus a NRPS+PKS neighbor counts for both constituent classes.

| Class | Query count including hybrids | ESM | MAC | MAP |
| --- | ---: | ---: | ---: | ---: |
| NRPS | 917 | 81.21% | **83.16%** | 81.83% |
| Other | 382 | 32.53% | **49.34%** | 38.01% |
| PKS | 1,092 | 86.00% | **87.64%** | 85.36% |
| RiPP | 363 | 69.70% | **86.35%** | 80.68% |
| Saccharide | 203 | 47.62% | **58.78%** | 38.13% |
| Terpene | 192 | 49.06% | **62.81%** | 47.47% |

Restricting both queries and candidates to the 2,144 single-label BGCs gives
k=15 same-class precision of 70.70% / 80.48% / 72.50% for ESM / MAC / MAP.
The corresponding original cosine silhouette scores are 0.0072 / 0.2265 /
0.1603. This is consistent with useful local class structure but incomplete
global separation. The negative silhouette scores for 27 complete combinations
are harder to interpret because several combinations are singletons and hybrids
share biological constituents; they should not be the main quality criterion.

## What the UMAP pictures establish

All pictures contain class-enriched local regions, including RiPP-rich ends and
PKS/NRPS-enriched branches, as well as mixed regions and repeated occurrences of
the same class in different parts of the map. They are projections, not results
of a clustering algorithm. A broad biosynthetic class need not form one isolated
island. Hybrid BGCs may reasonably be close to either of their parent classes.

| Existing UMAP diagnostic, k=15 | ESM | MAC | MAP |
| --- | ---: | ---: | ---: |
| Original/2D neighbor-set overlap | 43.70% | 50.50% | **59.38%** |
| Trustworthiness | 0.9693 | 0.9873 | **0.9949** |
| 2D-neighbor label Jaccard | 0.6393 | **0.7435** | 0.6705 |

The first two rows measure fidelity to each model's own geometry, not biological
relevance. MAP is easier to represent faithfully in these two-dimensional fits,
while MAC retains better label agreement even when evaluated in two dimensions.
This distinction can explain part of the visual impression. The exact overlap
percentages also show why retrieval should use original vectors.

UMAP does not fully preserve density and may introduce artificial separations;
see the [official clustering guidance](https://umap-learn.readthedocs.io/en/latest/clustering.html).
Trustworthiness penalizes neighbors introduced in the projection according to
their original ranks; see the [metric definition](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html).

## Token counts and the exceptional record

The median token count is 20; 96.5% of entries have at most 100 tokens. Except
for BGC0002977, the maximum is 224. BGC0002977 has 4,263 tokens and stretches the
linear color scale so most ordinary BGCs appear nearly identical in color.

The author explicitly excludes BGC0002977 when constructing the MAC and MAP
metadata in [extract_BGCs.py](../../data_preparation/extract_BGCs.py), although
the ESM cache retains it. A local raw MIBiG 4.0 record for the same ID is available
at `../../../BGC/data/raw/mibig_gbk_4.0/BGC0002977.gbk`. It describes a
*Bacillus subtilis* whole-genome shotgun sequence: 4,150,267 bp, 4,117 CDS,
296 aSDomain features, and a region spanning the entire sequence with an
unknown product. The author's comment does not document the exclusion reason;
the observed record is nevertheless not an ordinary local BGC-scale input.

[The logarithmic-color diagnostic](diagnostic_token_counts.png) retains every
point and the original coordinates. It reveals token-count gradients in all
three representations. A [PDF](diagnostic_token_counts.pdf) is also available.

After excluding the Unknown from both query and candidate pools, the correlation
between a query's log2(token_count) and its neighbors' mean log2(token_count) is
0.803 / 0.831 / 0.796 for ESM / MAC / MAP. Mean absolute log2 count ratios within
the 15-neighbor sets are 0.913 / 0.806 / 0.877 versus 1.650 for randomly sampled
pairs. These are count-related associations, not proof that count causes the
observed separation: class, domain composition and token count are related.
Omitting the outlier when scoring saved coordinates does not undo its presence
in the original UMAP fit.

## Representation and supervision limits

Different-BGC mean cosine is 0.9915 for ESM, 0.4825 for MAC and 0.5647 for MAP.
ESM vectors share a strong common direction. Absolute similarity scores should
not be compared across encoders; their neighbor rankings still contain signal.
Centered-covariance effective rank is 13.18 / 9.44 / 4.14, respectively. MAP's
variance is concentrated in fewer directions, which is compatible with a more
faithful two-dimensional projection; this alone does not establish collapse or
inferior biological information.

The comparison does not isolate the amount of training information:

- MAC is trained for six-label classification using class-query cross-attention.
- MAP is trained for binary BGC/product matching using SMILES cross-attention.
  The current training entry point initializes these models separately; it is
  not a MAC-to-MAP continuation with extra labels.
- Their trained BGCEncoder widths differ: MAC 256 versus MAP 1,024. The task,
  capacity, sample construction and optimization also differ.
- The exported vector is a mean of BGCEncoder states **before** those task-specific
  cross-attention operations. Its cosine geometry was not directly optimized
  for BGC-to-BGC retrieval.

See [model architecture](../../model/BGC_models.py),
[training initialization](../../experiment/trainer.py), and
[feature extraction](../../model/features.py).

The supplied set is the author's full MIBiG collection, not an independent test
cohort. Current [split code](../../experiment/dataloaders.py) splits dataframe
rows rather than BGC/homology groups; MAP data include repeated BGCs across
product-pair rows. The exact historical member-0 training IDs were not
reconstructed. Current scores therefore cannot establish generalization to
unseen or nonhomologous BGCs. Exact-vector duplicates also exist: 16 extra rows
for ESM and 14 each for MAC/MAP, despite unique BGC IDs.

MAC's objective directly matches the class-label diagnostic, while MAP may
organize different properties relevant to products. To assess that possibility,
relevance must be defined using independent family/domain-architecture or product
structure information, rather than assuming that a nicer map proves it.

## Practical follow-up without training

First, keep BGC0002977 documented as an exceptional whole-genome-scale record
and compare ordinary-BGC views on the common 2,635-ID cohort. Refitting UMAP on
that filtered vector set would not require encoder inference. Continue using
logarithmic token-count coloring.

For class-based similarity on this collection, MAC member 0 is the stronger
current default. Keep MAP as a separate candidate for product/family similarity,
which has not been measured here. Before new training, test other existing
checkpoint members and a small, predetermined set of UMAP seeds/parameters for
stability, and assess original-vector retrieval with a clearly defined relevance
criterion and a suitable held-out/homology-controlled cohort. Select the encoder
by retrieval evidence rather than by which projection looks cleanest.

## Reproduction and files

`analyze_embeddings.py` uses only NumPy and the standard library. It validates
alignment, computes exact neighbors, preserves the Unknown exclusion explicitly,
and records input hashes and method definitions in `embedding_metrics.json`.
Forty independent queries per encoder were recalculated separately to verify
the three principal k=15 metrics. `diagnostic_token_counts.py` additionally
requires Matplotlib; its local validation environment is isolated under `/tmp`
and the project conda environment was not changed.

```bash
python outputs/analysis_umap/analyze_embeddings.py
python outputs/analysis_umap/diagnostic_token_counts.py
```

- [All retrieval scores](retrieval_metrics.tsv): k=5, 15, 50 and existing 2D k=15.
- [Per-class scores](class_metrics.tsv): constituent-class and single-label-query metrics.
- [Per-query scores](query_metrics.tsv): audit individual BGCs at k=15.
- [Projection metrics](projection_geometry.tsv): overlap and trustworthiness.
- [Full metadata and diagnostics](embedding_metrics.json): definitions, dimensions,
  class counts, random expectations, hashes, geometry and token-count statistics.

Only supplementary analysis files were created. Existing feature exports,
UMAP artifacts and tracked model code were preserved.
