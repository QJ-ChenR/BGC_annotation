"""Frozen, candidate-independent BGC embeddings.

This module deliberately has no classification queries, product representation,
trainable readout, or randomly initialized encoder fallback.
"""

from collections.abc import Mapping
import hashlib
from pathlib import Path
import pickle

import torch
from torch import nn


class BGCFeatureExtractor(nn.Module):
    """Masked mean and L2 normalization, optionally after a pretrained encoder.

    ``BGCFeatureExtractor(esm_size=1280)`` aggregates ESM vectors directly.
    Use :meth:`from_checkpoint` to include an author's trained MAC/MAP encoder.
    Inputs are floating tensors ``[batch, tokens, input_dim]`` and integer
    lengths ``[batch]``. Valid tokens must occupy the beginning of each row.
    Outputs are float32 unit vectors ``[batch, output_dim]``.

    Calling ``train()`` keeps this inference-only module and its children in
    evaluation mode. All parameters are frozen, and forward disables autograd.
    """

    def __init__(self, esm_size=1280):
        super().__init__()
        _positive_int(esm_size, "esm_size")
        self.input_dim = esm_size
        self.output_dim = esm_size
        self.encoder = None
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)
        self.metadata = {
            "route": "esm",
            "input_dim": esm_size,
            "output_dim": esm_size,
            "pooling": "masked_mean",
            "normalization": "l2",
            "frozen": True,
        }
        self.train(False)

    @classmethod
    def from_checkpoint(
        cls, path, route, member=0, device="cpu", trusted=False
    ):
        """Load exactly one encoder from the author's ensemble checkpoint.

        Expected format: ``{"config": {...}, "state_dict": [state, ...]}``.
        Only ``BGCEncoder.*`` weights are used; they are loaded strictly.
        ``route`` must agree with the checkpoint's original task. Structural
        checkpoints are rejected because this extractor only consumes sequence.

        PyTorch's restricted ``weights_only=True`` loader is the default.
        Set ``trusted=True`` only for a checkpoint whose source you trust:
        that explicitly enables Python pickle loading, which can execute code.
        """
        if route not in {"mac", "map"}:
            raise ValueError("route must be 'mac' or 'map'")
        if isinstance(member, bool) or not isinstance(member, int) or member < 0:
            raise ValueError("member must be a nonnegative integer")
        if not isinstance(trusted, bool):
            raise TypeError("trusted must be a boolean")
        path = Path(path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        try:
            checkpoint = torch.load(
                path, map_location="cpu", weights_only=not trusted
            )
        except pickle.UnpicklingError as exc:
            raise ValueError(
                "Checkpoint could not be loaded with the selected loading "
                "policy. The default uses weights_only=True. If this is an "
                "author legacy checkpoint and you trust its source, explicitly "
                "set trusted=True; this enables executable pickle loading."
            ) from exc
        except TypeError as exc:
            raise ValueError(
                "Checkpoint loading requires a PyTorch version supporting "
                "torch.load(..., weights_only=...). No unsafe fallback was used."
            ) from exc
        if not isinstance(checkpoint, Mapping):
            raise ValueError("Checkpoint must be a dictionary")
        config = checkpoint.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("Checkpoint config must be a dictionary")
        expected_task = {"mac": "classification", "map": "product_matching"}[route]
        if config.get("task") != expected_task:
            raise ValueError(
                f"Route {route!r} requires task {expected_task!r}; "
                f"checkpoint has {config.get('task')!r}"
            )
        data_config = config.get("data", {})
        if not isinstance(data_config, Mapping):
            raise ValueError("Checkpoint config.data must be a dictionary")
        for section in (config, data_config):
            for flag in ("use_structure", "structure_only"):
                value = section.get(flag, False)
                if not isinstance(value, (bool, int)) or value not in (0, 1):
                    raise ValueError(f"Checkpoint {flag} must be boolean or 0/1")
                if value:
                    raise ValueError(
                        f"Sequence-only extraction does not support {flag}=True"
                    )
        parameters = config.get("model_parameters")
        if not isinstance(parameters, Mapping):
            raise ValueError("Checkpoint model_parameters must be a dictionary")
        required = (
            "esm_size", "gearnet_size", "attention_dim", "num_heads", "dropout"
        )
        missing = [key for key in required if key not in parameters]
        if missing:
            raise ValueError(f"Missing checkpoint model parameters: {missing}")
        encoder_parameters = {key: parameters[key] for key in required}
        for key in required[:-1]:
            _positive_int(encoder_parameters[key], f"model_parameters.{key}")
        dropout = encoder_parameters["dropout"]
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float))
            or not 0 <= dropout <= 1
        ):
            raise ValueError("Checkpoint dropout must be a number in [0, 1]")
        dimension = encoder_parameters["attention_dim"]
        heads = encoder_parameters["num_heads"]
        if dimension % heads or (dimension // heads) % 2 or dimension // heads < 4:
            raise ValueError(
                "attention_dim must be divisible by num_heads, with an even "
                "RoPE head dimension of at least 4"
            )
        members = checkpoint.get("state_dict")
        if not isinstance(members, (list, tuple)) or not members:
            raise ValueError("Checkpoint state_dict must be a nonempty member list")
        if member >= len(members):
            raise ValueError(
                f"member {member} is out of range for {len(members)} members"
            )
        state = members[member]
        if not isinstance(state, Mapping) or not all(
            isinstance(key, str) for key in state
        ):
            raise ValueError("Selected checkpoint member must be a state dictionary")
        prefix = "BGCEncoder."
        encoder_state = {
            key[len(prefix):]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if not encoder_state:
            raise ValueError("Selected member contains no BGCEncoder.* weights")
        for key, value in encoder_state.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Encoder weight {key!r} is not a tensor")
            if not torch.isfinite(value).all():
                raise ValueError(f"Encoder weight {key!r} contains nonfinite values")

        # Keep the ESM-only baseline independent of rotary-embedding dependencies.
        from model.encoders import BGCEncoder

        encoder = BGCEncoder(**encoder_parameters)
        try:
            encoder.load_state_dict(encoder_state, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                f"Checkpoint BGCEncoder weights do not strictly match config: {exc}"
            ) from exc
        model = cls(esm_size=encoder_parameters["esm_size"])
        model.encoder = encoder
        model.output_dim = dimension
        model.metadata.update({
            "route": route,
            "output_dim": dimension,
            "checkpoint_path": str(path.resolve()),
            "checkpoint_sha256": _sha256(path),
            "checkpoint_task": expected_task,
            "checkpoint_member": member,
            "checkpoint_members": len(members),
            "checkpoint_loading": "trusted_pickle" if trusted else "weights_only",
            "model_parameters": encoder_parameters,
        })
        model.requires_grad_(False)
        model.to(device)
        model.train(False)
        return model

    def train(self, mode=True):
        """This module is permanently in evaluation mode."""
        if not isinstance(mode, bool):
            raise ValueError("training mode must be a boolean")
        return super().train(False)

    @torch.inference_mode()
    def forward(self, pros, lengths):
        if not isinstance(pros, torch.Tensor) or pros.ndim != 3:
            raise ValueError("pros must be a tensor shaped [batch, tokens, input_dim]")
        batch, tokens, features = pros.shape
        if batch == 0 or tokens == 0 or features != self.input_dim:
            raise ValueError(
                f"pros must have nonempty batch/tokens and last dimension "
                f"{self.input_dim}; got {tuple(pros.shape)}"
            )
        if not pros.is_floating_point():
            raise TypeError("pros must contain floating point embeddings")
        if not isinstance(lengths, torch.Tensor) or lengths.shape != (batch,):
            raise ValueError(f"lengths must be an integer tensor shaped [{batch}]")
        if lengths.dtype not in (
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
        ):
            raise TypeError("lengths must have an integer dtype")
        if pros.device != self._device_anchor.device:
            raise ValueError(
                f"Input device {pros.device} differs from model device "
                f"{self._device_anchor.device}; move model and inputs together"
            )
        lengths = lengths.to(device=pros.device, dtype=torch.long)
        if ((lengths <= 0) | (lengths > tokens)).any():
            raise ValueError(f"Every length must be between 1 and {tokens}")
        valid = torch.arange(tokens, device=pros.device)[None, :] < lengths[:, None]
        if not torch.isfinite(pros[valid]).all():
            raise ValueError("Valid token embeddings contain nonfinite values")
        # Clearing padding first avoids NaN * 0 propagation in legacy encoders.
        inputs = pros.masked_fill(~valid[..., None], 0).float()
        self.train(False)
        hidden = (
            inputs if self.encoder is None
            else self.encoder(inputs, mask=~valid, structure=None)
        )
        if not torch.isfinite(hidden[valid]).all():
            raise ValueError("Encoder produced nonfinite valid token embeddings")
        pooled = hidden.masked_fill(~valid[..., None], 0).sum(dim=1)
        pooled = pooled / lengths[:, None].float()
        norms = torch.linalg.vector_norm(pooled, dim=-1, keepdim=True)
        if not torch.isfinite(norms).all():
            raise ValueError("Pooled embedding has a nonfinite norm")
        if (norms == 0).any():
            raise ValueError("Pooled embedding has zero norm; cannot normalize")
        return pooled / norms


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
