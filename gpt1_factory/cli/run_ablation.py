from __future__ import annotations

import argparse
from typing import Any, Dict, List

import yaml
from loguru import logger
import subprocess
import sys


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    abls: List[Dict[str, Any]] = cfg.get("ablations", [])
    results = []

    for abl in abls:
        name = abl["name"]
        finetune_cfg = abl["finetune_cfg"]
        model_cfg = abl.get("model_cfg")
        overrides = abl.get("overrides", {})
        ov_list = [f"{k}={v}" for k, v in overrides.items()]
        if model_cfg:
            # 借助 include+= 将 model 覆盖文件注入 finetune 配置
            ov_list.append(f"include+={model_cfg}")
        logger.info(f"[ablation] running {name}")
        cmd = [sys.executable, "-m", "gpt1_factory.cli.finetune", "--cfg", finetune_cfg] + ov_list
        subprocess.run(cmd, check=True)
        results.append({"name": name, "status": "done"})

    logger.info(f"Ablations finished: {results}")


if __name__ == "__main__":
    main()
