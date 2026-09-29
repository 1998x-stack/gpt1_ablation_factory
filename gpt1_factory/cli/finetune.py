from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import yaml
from loguru import logger
from torch.utils.data import DataLoader

from ..configs import (
    DataConfig,
    ExpConfig,
    FinetuneConfig,
    ModelConfig,
    dataclass_from_dict,
)
from ..data import load_dataset_factory
from ..registry import MODELS
from ..tokenization import (
    TokenizerArtifact,
    infer_tokenizer_artifact_path,
)
from ..trainers.finetune_trainer import FinetuneTrainer
from ..utils.logging import setup_loguru
from ..utils.seed import set_seed


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def _override_cfg(
    cfg: Dict[str, Any],
    kvs: List[str],
) -> Dict[str, Any]:
    includes_append: List[str] = []
    for kv in kvs:
        if "=" not in kv:
            continue
        key, value = kv.split("=", 1)
        key, value = key.strip(), value.strip()
        if key in ("include+", "include+="):
            includes_append.append(value)
            continue

        cur = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in cur or not isinstance(cur[part], dict):
                cur[part] = {}
            cur = cur[part]

        if value.lower() in ("true", "false"):
            parsed: Any = value.lower() == "true"
        else:
            try:
                parsed = float(value) if "." in value else int(value)
            except ValueError:
                parsed = value
        cur[parts[-1]] = parsed

    if includes_append:
        base_includes = cfg.get("include", []) or []
        cfg["include"] = base_includes + includes_append
    return cfg


def _apply_includes(cfg: Dict[str, Any]) -> Dict[str, Any]:
    incs = cfg.pop("include", []) or []
    merged: Dict[str, Any] = {}
    for path in incs:
        merged.update(_load_yaml(path))
    merged.update(cfg)
    return merged


def _ensure_finetune_data_defaults(cfg: Dict[str, Any]) -> None:
    data = cfg.setdefault("data", {})
    data.setdefault("name", "glue")
    data.setdefault("batch_size", 32)
    data.setdefault("num_workers", 2)
    data.setdefault("max_len", 256)
    data.setdefault(
        "cache_dir",
        str(Path("gpt1_ablation_factory/data/hf_cache").resolve()),
    )
    data.setdefault(
        "export_dir",
        str(Path("gpt1_ablation_factory/data/exports").resolve()),
    )
    data.setdefault(
        "local_text_dir",
        str(Path("gpt1_ablation_factory/data/text").resolve()),
    )


def load_cfg_with_overrides(
    path: str,
    overrides: List[str],
) -> Dict[str, Any]:
    cfg = _load_yaml(path)
    cfg = _apply_includes(cfg)
    cfg = _override_cfg(cfg, overrides)
    _ensure_finetune_data_defaults(cfg)
    return cfg


def _resolve_tokenizer_artifact(
    data_cfg: DataConfig,
    ft_cfg: FinetuneConfig,
) -> TokenizerArtifact:
    if data_cfg.tokenizer_artifact:
        artifact_path = Path(data_cfg.tokenizer_artifact)
    elif ft_cfg.pretrained_path:
        artifact_path = infer_tokenizer_artifact_path(
            ft_cfg.pretrained_path
        )
    else:
        raise ValueError(
            "Finetuning requires data.tokenizer_artifact when no pretrained "
            "checkpoint path is supplied."
        )
    return TokenizerArtifact.load(artifact_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Override like a.b=c or include+=path.yaml",
    )
    args = parser.parse_args()

    cfg = load_cfg_with_overrides(args.cfg, args.overrides)

    exp = dataclass_from_dict(ExpConfig, cfg.get("exp", {}))
    data_cfg = dataclass_from_dict(
        DataConfig,
        cfg.get("data", {}),
    )
    model_cfg = dataclass_from_dict(
        ModelConfig,
        cfg.get("model", {}),
    )
    ft_cfg = dataclass_from_dict(
        FinetuneConfig,
        cfg.get("finetune", {}),
    )

    Path(exp.out_dir).mkdir(parents=True, exist_ok=True)
    setup_loguru(Path(exp.out_dir) / "log.txt")
    set_seed(exp.seed)

    artifact = _resolve_tokenizer_artifact(data_cfg, ft_cfg)
    logger.info(
        "[tokenizer] inherited artifact={} fingerprint={} vocab={}",
        artifact.path,
        artifact.fingerprint,
        artifact.vocab_size,
    )

    bundle = load_dataset_factory(
        data_cfg,
        tokenizer=artifact.tokenizer,
    )
    if bundle.train is None:
        raise ValueError(
            f"Dataset {data_cfg.name!r} has no training split in this pipeline."
        )
    if bundle.valid is None:
        raise ValueError(
            f"Dataset {data_cfg.name!r} has no validation split."
        )

    train_loader = DataLoader(
        bundle.train,
        batch_size=data_cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        collate_fn=bundle.collator,
        drop_last=True,
    )
    valid_loader = DataLoader(
        bundle.valid,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        collate_fn=bundle.collator,
    )

    model_cfg.vocab_size = artifact.vocab_size
    model_kwargs = {
        key: value
        for key, value in model_cfg.__dict__.items()
        if key != "name"
    }
    backbone = MODELS.create(
        model_cfg.name,
        **model_kwargs,
    )

    trainer = FinetuneTrainer(
        exp,
        ft_cfg,
        backbone,
        bundle.num_labels or 2,
        train_loader,
        valid_loader,
        task_name=data_cfg.task or data_cfg.name,
        tokenizer_fingerprint=artifact.fingerprint,
    )
    best, detail = trainer.train()
    logger.info(
        "Best metric: {:.4f} detail: {}",
        best,
        detail,
    )


if __name__ == "__main__":
    main()
