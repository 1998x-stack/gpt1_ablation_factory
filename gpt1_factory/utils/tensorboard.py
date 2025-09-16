from __future__ import annotations
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


def create_tb_writer(out_dir: str | Path) -> SummaryWriter:
    """创建 TensorBoard 写入器。"""
    tb_dir = Path(out_dir) / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(str(tb_dir))
