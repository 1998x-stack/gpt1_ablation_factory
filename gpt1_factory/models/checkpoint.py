from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, Optional

import torch


_BLOCK_KEY = re.compile(r"(?:^|\.)blocks\.(\d+)\.")


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optim: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
    extra_modules: Optional[Mapping[str, torch.nn.Module]] = None,
) -> None:
    """Persist the backbone plus optional task-specific modules."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    obj = {
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

    torch.save(obj, str(path))


def select_pretrained_state_dict(
    state: Mapping[str, torch.Tensor],
    transfer_layers: int,
) -> dict[str, torch.Tensor]:
    """Select embeddings/shared weights plus the first K Transformer blocks.

    transfer_layers=-1 loads the entire pretrained state. transfer_layers=0
    transfers only non-block parameters such as token/position embeddings.
    """

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


def load_pretrained_partial(
    model: torch.nn.Module,
    path: str | Path,
    transfer_layers: int = -1,
) -> dict[str, list[str]]:
    """Load all pretrained weights or embeddings/shared weights plus first K blocks."""

    ckpt = torch.load(str(path), map_location="cpu")
    state = ckpt["model"]
    filtered = select_pretrained_state_dict(state, transfer_layers)

    incompatible = model.load_state_dict(filtered, strict=False)
    return {
        "loaded_keys": sorted(filtered.keys()),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
