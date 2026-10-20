"""Tensor-only feature caches, with explicit provenance for legacy inputs."""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Mapping
from pathlib import Path

import torch


CACHE_FORMAT = "bgc_token_embeddings_v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_feature_cache(path, *, trusted=False):
    """Load the author's BGC->tensor-list cache or a versioned feature cache.

    Legacy preprocessing cannot be inferred from tensor dimensions. Preserve
    that uncertainty rather than assigning the new parser's provenance to it.
    """
    path = Path(path)
    if path.suffix.lower() in {".pkl", ".pickle"}:
        if not trusted:
            raise ValueError("Pickle caches require --trusted-input for a trusted file.")
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    else:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=not trusted)
        except pickle.UnpicklingError as exc:
            raise ValueError(
                "Cache cannot be loaded in tensor-only mode. Only for a trusted "
                "legacy file, use --trusted-input."
            ) from exc
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("Feature cache must be a nonempty mapping.")
    cache_format = payload.get("format")
    if isinstance(cache_format, str) and cache_format == CACHE_FORMAT:
        embeddings = payload.get("embeddings")
        records = payload.get("records", {})
        provenance = payload.get("provenance", {})
        if not isinstance(records, Mapping) or not isinstance(provenance, Mapping):
            raise ValueError("Cache records and provenance must be mappings.")
    elif isinstance(cache_format, str):
        raise ValueError(f"Unsupported feature cache format: {cache_format!r}")
    else:
        embeddings = payload
        records = {}
        provenance = {
            "format": "legacy_bgc_embedding_mapping",
            "preprocessing": "unknown; original token order is preserved",
            "esm_pooling": "unknown",
            "long_sequence_handling": "unknown",
        }
    if not isinstance(embeddings, Mapping) or not embeddings:
        raise ValueError("Cache embeddings must be a nonempty BGC ID mapping.")
    if any(not isinstance(key, str) or not key.strip() for key in embeddings):
        raise ValueError("All BGC IDs must be nonempty strings.")
    return embeddings, dict(records), {
        **dict(provenance),
        "cache_path": str(path.resolve()),
        "cache_sha256": file_sha256(path),
    }


def token_tensor(value, *, expected_dim=None):
    """Validate one unpadded BGC, including valid vectors containing zeros."""
    if isinstance(value, (list, tuple)):
        if not value or not all(isinstance(item, torch.Tensor) for item in value):
            raise ValueError("Expected a nonempty list of token tensors.")
        value = torch.stack(value)
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError("Expected token embeddings with shape [tokens, features].")
    if value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError("BGC token embeddings cannot be empty.")
    if not value.is_floating_point():
        raise ValueError("Token embeddings must have floating point dtype.")
    if expected_dim is not None and value.shape[1] != expected_dim:
        raise ValueError(
            f"Expected {expected_dim} features, received {value.shape[1]}."
        )
    value = value.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(value).all():
        raise ValueError("Token embeddings contain NaN or infinity.")
    return value
