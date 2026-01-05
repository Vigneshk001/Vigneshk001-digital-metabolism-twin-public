"""Data loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd

from src.data.validate import FeatureContract
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_inference_features(processed_dir: Path, contract: FeatureContract | None = None) -> pd.DataFrame:
    path = processed_dir / "inference_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Inference features not found: {path}")
    df = pd.read_parquet(path)
    logger.info("Loaded inference features", extra={"rows": len(df), "cols": len(df.columns)})
    if contract:
        df = contract.validate_and_align(df)
        logger.info("Aligned to feature contract", extra={"cols": len(df.columns)})
    return df


def load_latent_states(processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / "latent_metabolic_states.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Latent states not found: {path}")
    df = pd.read_parquet(path)
    logger.info("Loaded latent states", extra={"rows": len(df), "cols": len(df.columns)})
    return df


def load_labels(cache_dir: Path) -> pd.DataFrame:
    path = cache_dir / "nhanes_merged.parquet"
    if not path.exists():
        raise FileNotFoundError(f"NHANES merged labels not found: {path}")
    df = pd.read_parquet(path)[["SEQN", "DIQ010"]]
    df["label"] = (df["DIQ010"] == 1).astype(int)
    df = df[["SEQN", "label"]].dropna()
    logger.info("Loaded labels", extra={"rows": len(df), "positives": int(df["label"].sum())})
    return df


def load_preprocessing_artifacts(model_dir: Path):
    imputer = joblib.load(model_dir / "imputer_simple.joblib") if (model_dir / "imputer_simple.joblib").exists() else None
    scaler = joblib.load(model_dir / "scaler_robust.joblib") if (model_dir / "scaler_robust.joblib").exists() else None
    return imputer, scaler
