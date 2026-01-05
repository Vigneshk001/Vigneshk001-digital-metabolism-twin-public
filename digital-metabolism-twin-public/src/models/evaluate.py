"""Evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from src.models.ensemble import InferenceEngine
from src.utils.metrics import classification_metrics


@dataclass
class ComparisonResult:
    aligned_samples: int
    positive_labels: int
    without_latent: Dict[str, float]
    with_latent: Optional[Dict[str, float]] = None
    delta: Optional[Dict[str, float]] = None


def evaluate_latent_gain(
    X_no_latent: np.ndarray,
    X_with_latent: np.ndarray,
    labels: np.ndarray,
    engine: InferenceEngine,
    model_type: str = "ensemble",
) -> ComparisonResult:
    result_no_latent = engine.predict_batch(X_no_latent, model_type=model_type)
    metrics_no = classification_metrics(labels, result_no_latent.probabilities)

    try:
        result_with_latent = engine.predict_batch(X_with_latent, model_type=model_type)
        metrics_with = classification_metrics(labels, result_with_latent.probabilities)
        delta = {
            "auc": metrics_with["auc"] - metrics_no["auc"],
            "ap": metrics_with["ap"] - metrics_no["ap"],
            "brier": metrics_with["brier"] - metrics_no["brier"],
        }
        return ComparisonResult(
            aligned_samples=len(labels),
            positive_labels=int(labels.sum()),
            without_latent=metrics_no,
            with_latent=metrics_with,
            delta=delta,
        )
    except Exception:
        return ComparisonResult(
            aligned_samples=len(labels),
            positive_labels=int(labels.sum()),
            without_latent=metrics_no,
        )
