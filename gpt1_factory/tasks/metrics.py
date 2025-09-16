from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr


def compute_metrics(task: str, y_true: Sequence, y_pred: Sequence) -> Dict[str, float]:
    """GLUE 主要任务指标：
    - SST2/MNLI: acc
    - MRPC/QQP:  acc & f1（macro）
    - CoLA:     matthews
    - STS-B:    pearson & spearman （y_pred 应为回归输出的连续值）
    """
    if task in ("sst2", "mnli"):
        return {"acc": float(accuracy_score(y_true, y_pred))}
    if task in ("mrpc", "qqp"):
        return {
            "acc": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred)),
        }
    if task == "cola":
        return {"matthews": float(matthews_corrcoef(y_true, y_pred))}
    if task == "stsb":
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        p = float(pearsonr(y_true, y_pred)[0])
        s = float(spearmanr(y_true, y_pred)[0])
        return {"pearson": p, "spearman": s}
    # 兜底
    return {"acc": float(accuracy_score(y_true, y_pred))}
