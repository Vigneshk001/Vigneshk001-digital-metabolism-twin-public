"""Inference engines."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import psutil

from src.models.base import SecureModelLoader


@dataclass
class InferenceResult:
    predictions: np.ndarray
    probabilities: np.ndarray
    confidence_scores: Optional[np.ndarray]
    inference_time_ms: float
    memory_usage_mb: float


class InferenceEngine:
    """Batch inference with optional ensemble averaging."""

    def __init__(self, model_loader: SecureModelLoader, batch_size: int = 2048):
        self.model_loader = model_loader
        self.batch_size = batch_size

    def predict_batch(self, X: np.ndarray, model_type: str = "ensemble") -> InferenceResult:
        start_time = time.time()
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024**2

        probs_all = []
        for i in range(0, len(X), self.batch_size):
            batch = X[i : i + self.batch_size]
            if model_type == "ensemble":
                probs = self._ensemble_proba(batch)
            else:
                probs = self._single_model_proba(batch, model_type)
            probs_all.append(probs)

        probabilities = np.concatenate(probs_all)
        predictions = (probabilities >= 0.5).astype(int)
        confidence = 2 * np.abs(probabilities - 0.5)

        mem_after = process.memory_info().rss / 1024**2
        duration_ms = (time.time() - start_time) * 1000

        return InferenceResult(
            predictions=predictions,
            probabilities=probabilities,
            confidence_scores=confidence,
            inference_time_ms=duration_ms,
            memory_usage_mb=mem_after - mem_before,
        )

    def _ensemble_proba(self, X: np.ndarray) -> np.ndarray:
        probs = []
        for name, model in self.model_loader.models.items():
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X)[:, 1]
                probs.append(p)
        if not probs:
            raise RuntimeError("No models available for ensemble inference")
        return np.mean(probs, axis=0)

    def _single_model_proba(self, X: np.ndarray, name: str) -> np.ndarray:
        model = self.model_loader.models.get(name)
        if model is None:
            raise ValueError(f"Model '{name}' not loaded")
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        raise TypeError(f"Model '{name}' does not support probability prediction")
