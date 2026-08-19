from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(path: str | Path, model: torch.nn.Module, optim: Optional[torch.optim.Optimizer] = None, step: int = 0) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {"model": model.state_dict(), "step": step}
    if optim is not None:
        obj["optim"] = optim.state_dict()
    torch.save(obj, str(path))


def load_pretrained_partial(model: torch.nn.Module, path: str | Path, transfer_layers: int = -1) -> None:
    """Partially load pretrained weights: only the first K layers (transfer_layers>0) or all (-1)."""
    ckpt = torch.load(str(path), map_location="cpu")
    state = ckpt["model"]
    if transfer_layers == -1:
        model.load_state_dict(state, strict=False)
        return
    # filter only the first K layer weights
    filtered = {}
    for k, v in state.items():
        if ".blocks." in k:
            # parse layer number
            try:
                layer_idx = int(k.split(".blocks.")[1].split(".")[0])
                if layer_idx < transfer_layers:
                    filtered[k] = v
            except Exception:
                pass
        else:
            # keep shared layers (embedding/pos_emb/ln_f etc)
            filtered[k] = v
    model.load_state_dict(filtered, strict=False)
