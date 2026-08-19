from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr


def compute_metrics(task: str, y_true: Sequence, y_pred: Sequence) -> Dict[str, float]:
    """GLUE primary-task metrics.
    - SST-2/MNLI: accuracy
    - MRPC/QQP: accuracy & macro-F1
    - CoLA: Matthews correlation
    - STS-B: Pearson & Spearman (y_pred should be continuous regression outputs)
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
    # fallback
    return {"acc": float(accuracy_score(y_true, y_pred))}
