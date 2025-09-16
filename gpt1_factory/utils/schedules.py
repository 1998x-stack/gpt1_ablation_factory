from __future__ import annotations

import math
from typing import Callable


def linear_warmup_cosine_decay(warmup_steps: int, max_steps: int) -> Callable[[int], float]:
    """线性 warmup + cosine 衰减的学习率比例函数。"""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda
