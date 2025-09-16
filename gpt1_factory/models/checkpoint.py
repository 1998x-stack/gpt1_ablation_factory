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
    """部分加载预训练：仅前 K 层（transfer_layers>0）或全部（-1）。"""
    ckpt = torch.load(str(path), map_location="cpu")
    state = ckpt["model"]
    if transfer_layers == -1:
        model.load_state_dict(state, strict=False)
        return
    # 过滤仅前 K 层相关的权重
    filtered = {}
    for k, v in state.items():
        if ".blocks." in k:
            # 解析层号
            try:
                layer_idx = int(k.split(".blocks.")[1].split(".")[0])
                if layer_idx < transfer_layers:
                    filtered[k] = v
            except Exception:
                pass
        else:
            # 共享层（嵌入/pos_emb/ln_f等）保留
            filtered[k] = v
    model.load_state_dict(filtered, strict=False)
