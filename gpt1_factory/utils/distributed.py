from __future__ import annotations
import os
import torch
from contextlib import contextmanager


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def is_main_process() -> bool:
    return rank() == 0


@contextmanager
def main_process_first():
    """在多卡情况下，仅主进程先执行（例如 tokenizer 训练），其余等待。"""
    if world_size() > 1:
        torch.distributed.barrier()
        if is_main_process():
            yield
            torch.distributed.barrier()
        else:
            torch.distributed.barrier()
            yield
            torch.distributed.barrier()
    else:
        yield
