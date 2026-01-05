"""SHAP explainability for tree-based models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import shap

from src.data.load_data import load_inference_features
from src.data.preprocess import apply_imputer_scaler
from src.models.base import SecureModelLoader
from src.utils.helpers import hash_files, list_model_files
from src.utils.logger import get_logger

logger = get_logger(__name__)


def prepare_features(root: Path, model_dir: Path) -> Tuple[np.ndarray, list[str]]:
    X_df = load_inference_features(root / "data" / "processed")
    imputer = joblib.load(model_dir / "imputer_simple.joblib") if (model_dir / "imputer_simple.joblib").exists() else None
    scaler = joblib.load(model_dir / "scaler_robust.joblib") if (model_dir / "scaler_robust.joblib").exists() else None
    X_proc, feature_names = apply_imputer_scaler(X_df, imputer, scaler)
    return X_proc, feature_names


def run_shap(
    model_dir: Path,
    out_dir: Path,
    shap_top_k: int = 10,
    shap_max_local: int = 50,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    root = out_dir.parents[1]

    X, feature_names = prepare_features(root, model_dir)

    loader = SecureModelLoader(model_dir)
    model = loader.load_xgboost()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    expected_value = explainer.expected_value

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
        if isinstance(expected_value, list):
            expected_value = expected_value[1]

    shap_abs = np.abs(shap_values)
    df_global = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": shap_abs.mean(axis=0),
            "std_abs_shap": shap_abs.std(axis=0),
            "max_abs_shap": shap_abs.max(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    df_global.to_csv(out_dir / "shap_global.csv", index=False)

    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)
    high_risk_idx = np.where(proba >= 0.5)[0]
    if len(high_risk_idx) == 0:
        top_idx = np.argsort(proba)[::-1][:shap_max_local]
    else:
        top_idx = high_risk_idx[:shap_max_local]

    df_local = pd.DataFrame(shap_values[top_idx], columns=feature_names)
    df_local.insert(0, "probability", proba[top_idx])
    df_local.insert(0, "prediction", preds[top_idx])
    df_local.insert(0, "sample_index", top_idx)
    df_local.insert(0, "base_value", expected_value if np.isscalar(expected_value) else expected_value[1])
    df_local.to_csv(out_dir / "shap_local_high_risk.csv", index=False)

    model_hash = hash_files(list_model_files(model_dir))
    shap_meta = {
        "model_dir": str(model_dir),
        "model_hash": model_hash,
        "sample_count": int(len(X)),
        "expected_value": float(expected_value if np.isscalar(expected_value) else expected_value[1]),
        "shap_top_k": shap_top_k,
        "local_explanations": int(len(top_idx)),
    }
    with open(out_dir / "shap_summary.json", "w", encoding="utf-8") as f:
        json.dump(shap_meta, f, indent=2)

    logger.info("SHAP analysis complete", extra={"out_dir": str(out_dir)})
