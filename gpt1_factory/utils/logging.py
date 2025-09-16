from __future__ import annotations

import os
from pathlib import Path
from loguru import logger


def setup_loguru(log_file: str | Path) -> None:
    """配置 Loguru 到文件与控制台。"""
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), colorize=True)
    logger.add(str(log_file), rotation="10 MB", retention=5, enqueue=True)
    logger.info(f"Log file: {log_file}")
