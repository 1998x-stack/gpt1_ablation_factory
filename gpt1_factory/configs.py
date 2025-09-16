from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ExpConfig:
    """Experiment-level configuration.

    Attributes:
        out_dir: 输出目录。
        seed: 随机种子。
    """
    out_dir: str
    seed: int = 42


@dataclass
class OptimConfig:
    """Optimizer/training schedule config for pretraining.

    按论文：Adam + 2000 warmup + cosine 衰减。
    """
    lr: float = 2.5e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    max_steps: int = 500_000
    scheduler: str = "cosine"
    grad_clip: float = 1.0
    amp: bool = True


@dataclass
class FinetuneConfig:
    """Finetuning config.

    Attributes:
        pretrained_path: 预训练权重路径；为空则从头训练。
        aux_lm_lambda: 辅助 LM loss 系数 (λ)，论文建议 0.5。
        epochs: 训练轮数，论文常用 3。
        lr: 微调学习率，论文 6.25e-5。
        warmup_ratio: 线性 warmup 比例（如 0.002 = 0.2%）。
        weight_decay: AdamW 权重衰减。
        grad_clip: 梯度裁剪阈值。
        amp: 是否启用混合精度。
        transfer_layers: 从预训练中迁移的层数；-1 表示全部。
        head_dropout: 分类头 dropout。
    """
    pretrained_path: str = ""
    aux_lm_lambda: float = 0.5
    epochs: int = 3
    lr: float = 6.25e-5
    warmup_ratio: float = 0.002
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    amp: bool = True
    transfer_layers: int = -1
    head_dropout: float = 0.1


@dataclass
class DataConfig:
    """Data config shared by modules."""
    name: str
    batch_size: int
    num_workers: int = 4
    max_len: int | None = None
    seq_len: int | None = None
    task: Optional[str] = None
    text_column: Optional[str] = None
    bpe: Optional[Dict[str, Any]] = None


@dataclass
class ModelConfig:
    """Model architecture config."""
    name: str
    vocab_size: int = 50257
    n_layer: int | None = None
    n_head: int | None = None
    d_model: int = 768
    d_ff: int | None = None
    max_len: int = 512
    dropout: float = 0.1
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    tie_emb: bool = False
    gelu: bool = True
    lstm_hidden: int | None = None
    num_layers: int | None = None


@dataclass
class CheckpointConfig:
    """Checkpoint saving behavior."""
    save_every: int = 10_000
    keep_last: int = 5


def dataclass_from_dict(dc_cls, d: dict):
    """将 dict 递归映射到 dataclass。"""
    fieldset = {f.name for f in dataclasses.fields(dc_cls)}
    kwargs = {k: v for k, v in d.items() if k in fieldset}
    return dc_cls(**kwargs)
