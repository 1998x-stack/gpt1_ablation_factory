from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import yaml
import torch
from torch.utils.data import DataLoader
from loguru import logger

from ..configs import ExpConfig, OptimConfig, DataConfig, ModelConfig, CheckpointConfig, dataclass_from_dict
from ..utils.logging import setup_loguru
from ..utils.seed import set_seed
from ..registry import MODELS
from ..data import load_dataset_factory
from ..trainers.pretrain_trainer import PretrainTrainer


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # 处理 include
    if "include" in cfg:
        for p in cfg["include"]:
            with open(p, "r") as fi:
                inc = yaml.safe_load(fi)
                cfg.update(inc)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
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
    train_loader = DataLoader(bundle.train, batch_size=data_cfg.batch_size, shuffle=True,
                              num_workers=data_cfg.num_workers, collate_fn=bundle.collator, drop_last=True)

    # 模型
    model = MODELS.create(model_cfg.name, **model_cfg.__dict__)

    # 训练
    trainer = PretrainTrainer(exp, optim_cfg, model, train_loader, out_dir=exp.out_dir, amp=optim_cfg.amp)
    trainer.train(save_every=cfg.get("checkpoint", {}).get("save_every", 10000),
                  keep_last=cfg.get("checkpoint", {}).get("keep_last", 5))


if __name__ == "__main__":
    main()
