#!/usr/bin/env python3
"""Evaluate latent vs baseline performance using existing models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config.paths import ProjectPaths, load_config
from src.data.load_data import load_inference_features, load_latent_states, load_labels, load_preprocessing_artifacts
from src.data.validate import FeatureContract
from src.data.preprocess import apply_imputer_scaler
from src.models.base import SecureModelLoader
from src.models.ensemble import InferenceEngine
from src.models.evaluate import evaluate_latent_gain
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate latent gain")
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--model-type", default="ensemble", choices=["ensemble", "xgb", "lgb", "rf", "cb"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / args.config)
    paths = ProjectPaths.from_config(root, config)
    paths.ensure()

    model_dir = paths.models_dir / args.experiment
    contract = FeatureContract(model_dir)
    X_df = load_inference_features(paths.processed_dir, contract)
    latent = load_latent_states(paths.processed_dir)
    labels_df = load_labels(paths.cache_dir)

    seqn_common = set(latent["SEQN"]) & set(labels_df["SEQN"])
    latent_filtered = latent[latent["SEQN"].isin(seqn_common)].reset_index(drop=True)
    labels_filtered = labels_df[labels_df["SEQN"].isin(seqn_common)].reset_index(drop=True)
    y = labels_filtered["label"].values

    if "SEQN" in X_df.columns:
        X_baseline_df = X_df[X_df["SEQN"].isin(seqn_common)].reset_index(drop=True)
    else:
        X_baseline_df = X_df.iloc[: len(latent_filtered)].copy()

    imputer, scaler = load_preprocessing_artifacts(model_dir)
    X_baseline, _ = apply_imputer_scaler(X_baseline_df, imputer, scaler)

    latent_cols = [c for c in latent_filtered.columns if c.startswith("latent_")]
    X_with_latent = np.column_stack([X_baseline, latent_filtered[latent_cols].values])

    loader = SecureModelLoader(model_dir)
    loader.load_xgboost()
    loader.load_lightgbm()
    loader.load_random_forest()
    loader.load_catboost()

    engine = InferenceEngine(loader, batch_size=config.get("inference", {}).get("batch_size", 2048))
    comparison = evaluate_latent_gain(X_baseline, X_with_latent, y, engine, model_type=args.model_type)
    logger.info("Evaluation complete", extra=comparison.__dict__)


if __name__ == "__main__":
    main()
