from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr


def compute_metrics(task: str, y_true, y_pred) -> Dict[str, float]:
    """GLUE 常用的指标集合。"""
    if task in ("sst2", "mnli", "qqp", "mrpc"):
        return {"acc": float(accuracy_score(y_true, y_pred))}
    if task == "cola":
        return {"matthews": float(matthews_corrcoef(y_true, y_pred))}
    if task == "stsb":
        p = pearsonr(y_true, y_pred)[0]
        return {"pearson": float(p)}
    return {"acc": float(accuracy_score(y_true, y_pred))}
