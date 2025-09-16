from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import yaml
from loguru import logger
from torch.utils.data import DataLoader

from ..configs import ExpConfig, FinetuneConfig, DataConfig, ModelConfig, dataclass_from_dict
from ..utils.logging import setup_loguru
from ..utils.seed import set_seed
from ..registry import MODELS
from ..data import load_dataset_factory
from ..trainers.finetune_trainer import FinetuneTrainer


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _override_cfg(cfg: Dict[str, Any], kvs: List[str]) -> Dict[str, Any]:
    """支持 a.b=c / include+=path.yaml。先记录 include+=，稍后合并。"""
    includes_append: List[str] = []
    for kv in kvs:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k in ("include+", "include+="):
            includes_append.append(v)
            continue

        cur = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        # 粗略类型推断
        if v.lower() in ("true", "false"):
            val: Any = (v.lower() == "true")
        else:
            try:
                val = float(v) if "." in v else int(v)
            except ValueError:
                val = v
        cur[parts[-1]] = val

    if includes_append:
        base_includes = cfg.get("include", []) or []
        cfg["include"] = base_includes + includes_append
    return cfg


def _apply_includes(cfg: Dict[str, Any]) -> Dict[str, Any]:
    incs = cfg.pop("include", []) or []
    merged: Dict[str, Any] = {}
    for p in incs:
        merged.update(_load_yaml(p))
    merged.update(cfg)
    return merged


def load_cfg_with_overrides(path: str, overrides: List[str]) -> Dict[str, Any]:
    cfg = _load_yaml(path)
    cfg = _override_cfg(cfg, overrides)   # 先接收 include+= 与键值覆盖
    cfg = _apply_includes(cfg)            # 再实际合并 include
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("overrides", nargs="*", help="Override like a.b=c or include+=path.yaml")
    args = parser.parse_args()

    cfg = load_cfg_with_overrides(args.cfg, args.overrides)

    exp = dataclass_from_dict(ExpConfig, cfg.get("exp", {}))
    data_cfg = dataclass_from_dict(DataConfig, cfg.get("data", {}))
    model_cfg = dataclass_from_dict(ModelConfig, cfg.get("model", {}))
    ft_cfg = dataclass_from_dict(FinetuneConfig, cfg.get("finetune", {}))

    Path(exp.out_dir).mkdir(parents=True, exist_ok=True)
    setup_loguru(Path(exp.out_dir) / "log.txt")
    set_seed(exp.seed)

    bundle = load_dataset_factory(data_cfg)
    train_loader = DataLoader(
        bundle.train, batch_size=data_cfg.batch_size, shuffle=True,
        num_workers=data_cfg.num_workers, collate_fn=bundle.collator, drop_last=True
    )
    valid_loader = DataLoader(
        bundle.valid, batch_size=data_cfg.batch_size, shuffle=False,
        num_workers=data_cfg.num_workers, collate_fn=bundle.collator
    )

    # 模型（去掉 name 以避免 Registry.create(name=..., name=...) 冲突）
    model_kwargs = {k: v for k, v in model_cfg.__dict__.items() if k != "name"}
    backbone = MODELS.create(model_cfg.name, **model_kwargs)

    trainer = FinetuneTrainer(
        exp, ft_cfg, backbone, bundle.num_labels or 2, train_loader, valid_loader,
        task_name=data_cfg.task or data_cfg.name
    )
    best, detail = trainer.train()
    logger.info(f"Best metric: {best:.4f} detail: {detail}")


if __name__ == "__main__":
    main()
