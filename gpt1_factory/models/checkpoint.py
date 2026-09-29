from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from ..tokenization import TokenizerCompatibilityError


_BLOCK_KEY = re.compile(r"(?:^|\.)blocks\.(\d+)\.")


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optim: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
    extra_modules: Optional[Mapping[str, torch.nn.Module]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """Persist the backbone plus optional task modules and immutable metadata."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    obj: dict[str, Any] = {
        "model": model.state_dict(),
        "step": step,
    }
    if optim is not None:
        obj["optim"] = optim.state_dict()
    if extra_modules:
        obj["modules"] = {
            name: module.state_dict()
            for name, module in extra_modules.items()
        }
    if metadata:
        obj["metadata"] = dict(metadata)

    torch.save(obj, str(path))


def select_pretrained_state_dict(
    state: Mapping[str, torch.Tensor],
    transfer_layers: int,
) -> dict[str, torch.Tensor]:
    """Select embeddings/shared weights plus the first K Transformer blocks."""

    if transfer_layers < -1:
        raise ValueError("transfer_layers must be -1 or a non-negative integer.")

    if transfer_layers == -1:
        return dict(state)

    filtered: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        match = _BLOCK_KEY.search(key)
        if match is None:
            filtered[key] = value
            continue

        layer_idx = int(match.group(1))
        if layer_idx < transfer_layers:
            filtered[key] = value

    return filtered


def _prepare_state_for_model(
    model: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    prepared = dict(state)

    if (
        getattr(model, "tie_emb", False)
        and "tok_emb.weight" in prepared
        and "lm_head.weight" in prepared
    ):
        prepared.pop("lm_head.weight")

    return prepared


def load_pretrained_partial(
    model: torch.nn.Module,
    path: str | Path,
    transfer_layers: int = -1,
    expected_tokenizer_fingerprint: str | None = None,
) -> dict[str, list[str]]:
    """Load pretrained weights after validating tokenizer identity."""

    ckpt = torch.load(str(path), map_location="cpu")
    metadata = ckpt.get("metadata", {})

    if expected_tokenizer_fingerprint is not None:
        actual = metadata.get("tokenizer_fingerprint")
        if actual is None:
            raise TokenizerCompatibilityError(
                "Checkpoint has no tokenizer fingerprint. Re-run pretraining with "
                "the tokenizer-artifact pipeline before paper-fidelity finetuning."
            )
        if actual != expected_tokenizer_fingerprint:
            raise TokenizerCompatibilityError(
                "Checkpoint/tokenizer mismatch: "
                f"checkpoint={actual}, artifact={expected_tokenizer_fingerprint}"
            )

    state = ckpt["model"]
    filtered = select_pretrained_state_dict(state, transfer_layers)
    prepared = _prepare_state_for_model(model, filtered)

    incompatible = model.load_state_dict(prepared, strict=False)
    return {
        "loaded_keys": sorted(prepared.keys()),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
