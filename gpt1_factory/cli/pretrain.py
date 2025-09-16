from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import yaml
from loguru import logger
from torch.utils.data import DataLoader

from ..configs import ExpConfig, OptimConfig, DataConfig, ModelConfig, CheckpointConfig, dataclass_from_dict
from ..utils.logging import setup_loguru
from ..utils.seed import set_seed
from ..registry import MODELS
from ..data import load_dataset_factory
from ..trainers.pretrain_trainer import PretrainTrainer


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _apply_includes(cfg: Dict[str, Any]) -> Dict[str, Any]:
    incs = cfg.pop("include", []) or []
    merged: Dict[str, Any] = {}
    for p in incs:
        merged.update(_load_yaml(p))
    merged.update(cfg)
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()

    # include -> merge first
    cfg = _apply_includes(_load_yaml(args.cfg))

    exp = dataclass_from_dict(ExpConfig, cfg.get("exp", {}))
    optim_cfg = dataclass_from_dict(OptimConfig, cfg.get("optim", {}))
    data_cfg = dataclass_from_dict(DataConfig, cfg.get("data", {}))
    model_cfg = dataclass_from_dict(ModelConfig, cfg.get("model", {}))
    ckpt_cfg = dataclass_from_dict(CheckpointConfig, cfg.get("checkpoint", {}))

    Path(exp.out_dir).mkdir(parents=True, exist_ok=True)
    setup_loguru(Path(exp.out_dir) / "log.txt")
    set_seed(exp.seed)

    # 数据
    bundle = load_dataset_factory(data_cfg)
    train_loader = DataLoader(
        bundle.train,
        batch_size=data_cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        collate_fn=bundle.collator,
        drop_last=True,
    )

    # 模型（去掉 name 这个键，避免 Registry.create 重复参数）
    model_kwargs = {k: v for k, v in model_cfg.__dict__.items() if k != "name"}
    model = MODELS.create(model_cfg.name, **model_kwargs)

    # 训练
    trainer = PretrainTrainer(exp, optim_cfg, model, train_loader, out_dir=exp.out_dir, amp=optim_cfg.amp)
    trainer.train(save_every=ckpt_cfg.save_every, keep_last=ckpt_cfg.keep_last)


if __name__ == "__main__":
    main()
