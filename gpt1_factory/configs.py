from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ExpConfig:
    """Experiment-level configuration."""

    out_dir: str
    seed: int = 42


@dataclass
class OptimConfig:
    """Optimizer/training schedule config for pretraining."""

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
    """Finetuning config."""

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
    cache_dir: Optional[str] = None
    local_text_dir: Optional[str] = None
    export_dir: Optional[str] = None


@dataclass
class ModelConfig:
    """Model architecture config.

    Defaults intentionally match the GPT-1 reference architecture: Post-LN,
    no extra final LayerNorm, and tied token/output embeddings.
    """

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
    tie_emb: bool = True
    gelu: bool = True
    norm_style: str = "post_ln"
    final_layer_norm: bool = False
    lstm_hidden: int | None = None
    num_layers: int | None = None


@dataclass
class CheckpointConfig:
    """Checkpoint saving behavior."""

    save_every: int = 10_000
    keep_last: int = 5


def dataclass_from_dict(dc_cls, d: dict):
    """Map a dict onto a dataclass, ignoring unknown keys."""

    fieldset = {field.name for field in dataclasses.fields(dc_cls)}
    kwargs = {key: value for key, value in d.items() if key in fieldset}
    return dc_cls(**kwargs)
