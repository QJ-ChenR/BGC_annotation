# Frozen BGC feature extraction

This workflow outputs one candidate-independent, fixed-size vector per BGC.
It does not train parameters, require product SMILES, or produce class scores.
Run the commands below from the repository root in the configured project
environment. The feature path uses PyTorch, NumPy and, for fresh GBK input,
Biopython and fair-esm. MAC/MAP encoders also use rotary-embedding-torch.

## Routes

| `--route` | Computation | Output dimension |
| --- | --- | --- |
| `esm` | ESM token vectors -> masked mean -> L2 normalization | Input dimension, normally 1280 |
| `mac` | ESM token vectors -> frozen MAC BGCEncoder -> masked mean -> L2 | Checkpoint dimension, normally 256 |
| `map` | ESM token vectors -> frozen MAP BGCEncoder -> masked mean -> L2 | Checkpoint dimension, normally 1024 |

MAC/MAP load only `BGCEncoder.*` from an existing trained checkpoint. The
checkpoint task, configuration and encoder weights must match strictly. Missing
weights never fall back to a randomly initialized encoder. Structure-dependent
checkpoints are rejected by this sequence-only interface.

`--member 0` selects the first ensemble member (indices are zero-based). Each
export uses exactly one member; hidden vectors from different members are not
averaged. Use the same checkpoint and member for queries and database entries.

All loaded parameters have `requires_grad=False`. Inference uses evaluation
mode and disables autograd. The BGC wrapper remains in evaluation mode even if
its caller invokes `train()`. Padding comes from explicit token lengths, so a
valid embedding coordinate equal to zero does not remove a token.

## Use an existing ESM cache

The author's `.pth` cache maps BGC IDs to lists of ESM token tensors. A mapping
to matrices shaped `[number_of_tokens, embedding_dimension]` is also accepted.

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
  --member 0 --device cpu \
  --output outputs/features_mac_member0
```

```bash
python -m experiment.extract_features \
  --esm-cache data/BGC_4.0/Esm2_rep_mibig.pth \
  --route map \
  --checkpoint ckpt/BGC_MAP/MAP_default/MAP_default.ckpt \
  --member 0 --device cpu \
  --output outputs/features_map_member0
```

The default loader uses `torch.load(weights_only=True)`. Some author checkpoints
contain legacy configuration or NumPy objects. If restricted loading rejects a
file whose source you trust, add `--trusted-input`. Legacy `.pkl`/`.pickle`
caches require that flag because their format uses Python pickle. The flag
enables executable pickle deserialization; it does not change the model or
preprocessing. There is no automatic unsafe fallback.

Legacy caches retain their original token order and content. Their ESM pooling,
sequence coverage, and annotation provenance are recorded as unknown. This
workflow cannot repair domain assignments or recover truncated residues from
cached vectors. In particular, `--esm-pooling mean` cannot convert an existing
BOS cache; regenerate its embeddings from GBK instead.

## Generate fresh features from GBK

```bash
python -m experiment.extract_features \
  --gbk data/mibig_gbk_4.0 \
  --esm-checkpoint data/esm2_t33_650M_UR50D.pt \
  --trusted-input \
  --route esm --device cuda:0 \
  --esm-pooling bos --max-residues 1021 --esm-batch-size 8 \
  --save-cache outputs/frozen_tokens.pth \
  --output outputs/features_esm_fresh
```

This example assumes the local ESM checkpoint is trusted; official checkpoints
often contain configuration objects requiring that flag. Weights are loaded
locally and never downloaded automatically. Contact-regression weights are not
needed for embeddings. Use `--device cpu` for CPU inference. `--esm-batch-size`
controls ESM windows; `--batch-size` independently controls BGC aggregation.

`--gbk` accepts one file or a recursively scanned directory (`.gbk`, `.gb`, or
`.genbank`). Each file must contain one record and has its basename as BGC ID.
Duplicate basenames are rejected before encoding. All CDSs are considered;
there is no core-only or six-class filtering, and no MIBiG JSON/Pfam scan is
required.

Fresh tokenization uses the following rules:

1. Resolve aSDomains to CDSs using identifier aliases and genomic containment,
   independently of feature encounter order. Use annotated translations.
2. Sort CDSs by genomic position and domains inside each CDS by protein N-to-C
   coordinates, including negative-strand genes. Domain sequences must match
   the corresponding parent interval. Missing protein coordinates can be
   recovered only from a unique exact translation match.
3. A CDS with domains contributes those domains. Its unannotated gaps are
   excluded; represented and unrepresented residue counts are recorded. A CDS
   without domains contributes its entire translation. Parent and domain
   sequences are not both counted as separate tokens.
4. Preserve identifiers, genomic/protein coordinates, strand, sequence hashes,
   annotation uncertainty and source-file hashes in metadata. Common partial
   boundaries retain their annotated bounds and uncertainty; unresolved or
   ambiguous annotations fail explicitly.

The order is not invariant to reversing the whole BGC, and genomic gene order
is not interpreted as biosynthetic reaction order. Corrected fresh ordering
can differ from the author's cached ordering; that is an input-distribution
change when transferring an existing encoder.

## Long sequences and the two pooling stages

ESM pooling (`--esm-pooling`) is distinct from BGC pooling (always masked mean
in this workflow). The default `bos` takes the final ESM layer's first token,
matching the author's representation choice for short inputs. `mean` averages
only amino-acid positions, excluding BOS, EOS and padding. It produces a
different input distribution for pretrained MAC/MAP weights.

A domain or fallback protein longer than `--max-residues` is split into
consecutive, nonoverlapping windows. Every residue is covered; BOS/EOS are
preserved for every window. For an original token of length L, window vectors
are combined as `sum(window_length / L * window_vector)`. This produces
**one vector per original token**, before the BGC encoder, so window count does
not multiply a long protein's contribution to BGC pooling. All boundaries,
window lengths and the aggregation rule are recorded.

This is an approximation that loses context across windows, not a claim of
equivalence to encoding the full protein. The default window limit is 1021
residues (at most 1023 tokens including BOS/EOS). Unlike the original script's
post-tokenization truncation, the same rule applies on CPU and CUDA. Domain-rich
proteins still contribute more original tokens than proteins without domains;
the workflow does not claim equal weighting per gene.

The saved `.pt`/`.pth` cache contains primitive metadata and CPU tensors in the
`bgc_token_embeddings_v1` format. It can be reused by all three routes without
recomputing ESM. Cache provenance keeps upstream exclusions, so subsequent
exports remain auditable.

## Output and errors

Each new output directory contains:

- `embeddings.npy`: float32 matrix `[BGC_count, dimension]`, with unit-length rows.
- `bgc_ids.tsv`: `row`, `bgc_id`, `token_count`, matching matrix rows exactly.
- `metadata.json`: route/member, checkpoint and code hashes, package versions,
  preprocessing settings, source provenance and available token/parent records.
- `failures.tsv`: excluded IDs, source paths, processing stages and reasons.
  Earlier cache exclusions are marked with an `upstream/` stage prefix.

By default, invalid inputs stop extraction with a nonzero exit status and a
failed manifest. `--skip-invalid` explicitly permits exporting valid BGCs and
records every exclusion. Empty BGCs, inconsistent dimensions, nonfinite valid
vectors and zero pooled norms are rejected. Failed runs must not be consumed as
feature exports. Output artifacts and saved caches are never overwritten; use
a new output path for a rerun.

`excluded_count` describes this run; `upstream_excluded_count` describes input
cache omissions. `status` is `complete_with_exclusions` if either is nonzero.
IDs are sorted for reproducibility. Existing vector axes can only be compared
across exports using the same encoder, member and preprocessing convention.

For two compatible exports, normalized vectors can be compared directly:

```python
import numpy as np

query = np.load("outputs/query/embeddings.npy", allow_pickle=False)
database = np.load("outputs/database/embeddings.npy", allow_pickle=False)
cosine_similarity = query @ database.T
```

Use `bgc_ids.tsv` to interpret ranks and remove self-matches as appropriate.
No similarity threshold or clustering boundary is learned or calibrated here.

## Python interface and validation

```python
import torch
from model.features import BGCFeatureExtractor

extractor = BGCFeatureExtractor.from_checkpoint(
    "ckpt/BGC_MAC/MAC_default/MAC_default.ckpt",
    route="mac", member=0, device="cpu",
)
# pros: [batch, padded_token_count, 1280]; lengths: [batch], integer.
vectors = extractor(pros, lengths)

# Baseline using the same token vectors:
baseline = BGCFeatureExtractor(esm_size=1280)
baseline_vectors = baseline(pros, lengths)
```

```bash
python -m unittest discover -s tests -v
```

Tests cover frozen/evaluation behavior, deterministic pooling, padding and
batch invariance, strict checkpoint/member loading, domain-parent resolution,
negative-strand order, long-sequence coverage, cache/export alignment and
reviewable errors. Tiny synthetic checkpoints exercise the real model and
loader code without downloading pretrained weights. Passing those tests does
not establish retrieval quality. Published checkpoints trained on MIBiG may
have seen a proposed retrieval test BGC; freezing does not remove that overlap.
