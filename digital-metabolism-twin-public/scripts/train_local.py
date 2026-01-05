#!/usr/bin/env python3
"""Train baseline and latent-augmented models locally."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from src.config.paths import ProjectPaths, load_config
from src.data.load_data import load_inference_features, load_latent_states, load_labels
from src.data.validate import FeatureContract
from src.models.train import train_baseline_and_augmented
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline + augmented models")
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--experiment", default=None, help="Optional experiment name; default timestamped")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", action="store_true", help="Persist trained models to models/saved")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / args.config)
    paths = ProjectPaths.from_config(root, config)
    paths.ensure()

    experiment = args.experiment or datetime.now().strftime("exp_%Y%m%d_%H%M%S")
    save_dir = paths.models_dir / experiment if args.save else None

    X_df = load_inference_features(paths.processed_dir)
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

    latent_cols = [c for c in latent_filtered.columns if c.startswith("latent_")]
    X_baseline = X_baseline_df.values
    X_augmented = np.column_stack([X_baseline, latent_filtered[latent_cols].values])

    result = train_baseline_and_augmented(
        X_baseline,
        X_augmented,
        y,
        test_size=args.test_size,
        seed=args.seed,
        save_dir=save_dir,
    )

    logger.info("Training complete", extra=result.to_serializable())


if __name__ == "__main__":
    main()
