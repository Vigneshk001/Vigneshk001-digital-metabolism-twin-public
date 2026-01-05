"""Metric helpers."""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    auc = roc_auc_score(y_true, proba)
    ap = average_precision_score(y_true, proba)
    brier = brier_score_loss(y_true, proba)
    return {"auc": float(auc), "ap": float(ap), "brier": float(brier)}
