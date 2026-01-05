"""Entry point for inference and evaluation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.paths import ProjectPaths, load_config
from src.data.load_data import load_inference_features, load_latent_states, load_labels, load_preprocessing_artifacts
from src.data.preprocess import apply_imputer_scaler
from src.data.validate import FeatureContract
from src.explainability.shap_analysis import run_shap
from src.models.base import SecureModelLoader
from src.models.ensemble import InferenceEngine
from src.models.evaluate import evaluate_latent_gain
from src.utils.helpers import hash_files, list_model_files
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_inference(paths: ProjectPaths, config: dict, experiment: str, model_type: str) -> None:
    model_dir = paths.models_dir / experiment
    contract = FeatureContract(model_dir)
    X_df = load_inference_features(paths.processed_dir, contract)
    imputer, scaler = load_preprocessing_artifacts(model_dir)
    X, feature_names = apply_imputer_scaler(X_df, imputer, scaler)

    loader = SecureModelLoader(model_dir)
    loader.load_xgboost()
    loader.load_lightgbm()
    loader.load_random_forest()
    loader.load_catboost()

    engine = InferenceEngine(loader, batch_size=config.get("inference", {}).get("batch_size", 2048))
    result = engine.predict_batch(X, model_type=model_type)

    predictions_path = paths.results_dir / "predictions.csv"
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "prediction": result.predictions,
            "probability": result.probabilities,
            "confidence": result.confidence_scores,
        }
    ).to_csv(predictions_path, index=False)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "experiment": experiment,
        "model_type": model_type,
        "sample_count": int(len(result.predictions)),
        "positive_rate": float(result.predictions.mean()),
        "inference_time_ms": float(result.inference_time_ms),
        "memory_delta_mb": float(result.memory_usage_mb),
        "feature_count": len(feature_names),
        "model_hash": hash_files(list_model_files(model_dir)),
    }
    summary_path = paths.results_dir / "inference_summary.json"
    summary_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Inference complete", extra={"predictions": str(predictions_path)})


def run_latent_comparison(paths: ProjectPaths, config: dict, experiment: str, model_type: str) -> None:
    model_dir = paths.models_dir / experiment
    contract = FeatureContract(model_dir)
    X_df = load_inference_features(paths.processed_dir, contract)
    latent = load_latent_states(paths.processed_dir)
    labels_df = load_labels(paths.cache_dir)

    seqn_common = set(latent["SEQN"]) & set(labels_df["SEQN"])
    latent_filtered = latent[latent["SEQN"].isin(seqn_common)].reset_index(drop=True)
    labels_filtered = labels_df[labels_df["SEQN"].isin(seqn_common)].reset_index(drop=True)
    y_true = labels_filtered["label"].values

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

    comparison = evaluate_latent_gain(X_baseline, X_with_latent, y_true, engine, model_type=model_type)
    out = {
        "timestamp": datetime.now().isoformat(),
        "aligned_samples": comparison.aligned_samples,
        "positive_labels": comparison.positive_labels,
        "without_latent": comparison.without_latent,
    }
    if comparison.with_latent:
        out["with_latent"] = comparison.with_latent
    if comparison.delta:
        out["delta"] = comparison.delta

    out_path = paths.results_dir / "auc_comparison.json"
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("Latent comparison saved", extra={"path": str(out_path)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Digital Metabolism Twin")
    parser.add_argument("--config", default="src/config/config.yaml", help="Path to YAML config")
    parser.add_argument("--experiment", required=True, help="Experiment directory under models/saved")
    parser.add_argument("--mode", choices=["inference", "latent_compare", "shap"], default="inference")
    parser.add_argument("--model-type", choices=["ensemble", "xgb", "lgb", "rf", "cb"], default="ensemble")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / args.config)
    paths = ProjectPaths.from_config(root, config)
    paths.ensure()

    if args.mode == "inference":
        run_inference(paths, config, args.experiment, args.model_type)
    elif args.mode == "latent_compare":
        run_latent_comparison(paths, config, args.experiment, args.model_type)
    elif args.mode == "shap":
        run_shap(paths.models_dir / args.experiment, paths.results_dir, config.get("explainability", {}).get("shap_top_k", 10))


if __name__ == "__main__":
    main()
