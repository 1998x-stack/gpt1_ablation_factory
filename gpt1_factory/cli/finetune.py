from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml
import torch
from torch.utils.data import DataLoader
from loguru import logger

from ..configs import ExpConfig, FinetuneConfig, DataConfig, ModelConfig, dataclass_from_dict
from ..utils.logging import setup_loguru
from ..utils.seed import set_seed
from ..registry import MODELS
from ..data import load_dataset_factory
from ..trainers.finetune_trainer import FinetuneTrainer


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if "include" in cfg:
        for p in cfg["include"]:
            with open(p, "r") as fi:
                inc = yaml.safe_load(fi)
                cfg.update(inc)
    return cfg


def override_cfg(cfg: Dict[str, Any], kvs: list[str]) -> Dict[str, Any]:
    # 简易 k1.k2=v 覆盖
    for kv in kvs:
        if "=" not in kv: continue
        k, v = kv.split("=", 1)
        cur = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            if p not in cur: cur[p] = {}
            cur = cur[p]
        # 类型推断
        if v.lower() in ("true", "false"):
            v = v.lower() == "true"
        else:
            try:
                if "." in v: v = float(v)
                else: v = int(v)
            except ValueError:
                pass
        cur[parts[-1]] = v
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("overrides", nargs="*", help="Override like a.b=c")
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    cfg = override_cfg(cfg, args.overrides)

    exp = dataclass_from_dict(ExpConfig, cfg.get("exp", {}))
    data_cfg = dataclass_from_dict(DataConfig, cfg.get("data", {}))
    model_cfg = dataclass_from_dict(ModelConfig, cfg.get("model", {}))
    ft_cfg = dataclass_from_dict(FinetuneConfig, cfg.get("finetune", {}))

    Path(exp.out_dir).mkdir(parents=True, exist_ok=True)
    setup_loguru(Path(exp.out_dir) / "log.txt")
    set_seed(exp.seed)

    bundle = load_dataset_factory(data_cfg)
    train_loader = DataLoader(bundle.train, batch_size=data_cfg.batch_size, shuffle=True,
                              num_workers=data_cfg.num_workers, collate_fn=bundle.collator, drop_last=True)
    valid_loader = DataLoader(bundle.valid, batch_size=data_cfg.batch_size, shuffle=False,
                              num_workers=data_cfg.num_workers, collate_fn=bundle.collator)

    backbone = MODELS.create(model_cfg.name, **model_cfg.__dict__)
    trainer = FinetuneTrainer(exp, ft_cfg, backbone, bundle.num_labels or 2,
                              train_loader, valid_loader, task_name=data_cfg.task or data_cfg.name)
    best, detail = trainer.train()
    logger.info(f"Best metric: {best:.4f} detail: {detail}")


if __name__ == "__main__":
    main()
