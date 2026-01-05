"""Preprocessing utilities."""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def apply_imputer_scaler(
    df: pd.DataFrame,
    imputer,
    scaler,
) -> Tuple[np.ndarray, List[str]]:
    """Align columns to imputer expectations and apply imputation + scaling."""
    feature_names = list(df.columns)

    if imputer is None or scaler is None:
        logger.warning("Preprocessing artifacts missing; returning raw values")
        return df.values, feature_names

    if hasattr(imputer, "feature_names_in_"):
        expected = list(imputer.feature_names_in_)
        missing = set(expected) - set(df.columns)
        extra = set(df.columns) - set(expected)
        if missing:
            raise ValueError(f"Missing features for imputer: {missing}")
        if extra:
            logger.info("Dropping unexpected features before imputation", extra={"extra": list(extra)})
            df = df[expected]
        feature_names = expected

    X_imputed = imputer.transform(df)

    if scaler.n_features_in_ != X_imputed.shape[1]:
        raise ValueError(
            f"Scaler expects {scaler.n_features_in_} features, got {X_imputed.shape[1]}"
        )

    X_scaled = scaler.transform(X_imputed)
    return X_scaled, feature_names
