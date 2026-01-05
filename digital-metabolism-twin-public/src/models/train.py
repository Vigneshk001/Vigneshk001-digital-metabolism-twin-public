"""Training routines for baseline and latent-augmented models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
import xgboost as xgb

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelMetrics:
    auc: float
    ap: float

    def to_dict(self) -> Dict[str, float]:
        return {"auc": float(self.auc), "ap": float(self.ap)}


@dataclass
class TrainingResult:
    timestamp: str
    samples_train: int
    samples_test: int
    test_positive_rate: float
    baseline: Dict[str, ModelMetrics]
    augmented: Dict[str, ModelMetrics]

    def to_serializable(self) -> Dict[str, Dict]:
        return {
            "timestamp": self.timestamp,
            "samples_train": self.samples_train,
            "samples_test": self.samples_test,
            "test_positive_rate": self.test_positive_rate,
            "baseline": {k: v.to_dict() for k, v in self.baseline.items()},
            "augmented": {k: v.to_dict() for k, v in self.augmented.items()},
        }


DEFAULT_PARAMS = {
    "xgb": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "random_state": 42, "verbosity": 0},
    "lgb": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "random_state": 42, "verbose": -1},
    "rf": {"n_estimators": 100, "max_depth": 10, "random_state": 42, "n_jobs": -1},
    "cb": {"iterations": 100, "depth": 5, "learning_rate": 0.1, "random_seed": 42, "verbose": 0},
}


def _train_and_score(model, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> ModelMetrics:
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    return ModelMetrics(
        auc=roc_auc_score(y_test, proba),
        ap=average_precision_score(y_test, proba),
    )


def train_baseline_and_augmented(
    X_baseline: np.ndarray,
    X_augmented: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
    save_dir: Optional[Path] = None,
) -> TrainingResult:
    X_base_train, X_base_test, X_aug_train, X_aug_test, y_train, y_test = train_test_split(
        X_baseline, X_augmented, y, test_size=test_size, random_state=seed, stratify=y
    )

    logger.info("Split data", extra={"train": len(y_train), "test": len(y_test)})

    baseline: Dict[str, ModelMetrics] = {}
    augmented: Dict[str, ModelMetrics] = {}

    # XGBoost
    xgb_base = xgb.XGBClassifier(**DEFAULT_PARAMS["xgb"])
    baseline["xgb"] = _train_and_score(xgb_base, X_base_train, X_base_test, y_train, y_test)
    xgb_aug = xgb.XGBClassifier(**DEFAULT_PARAMS["xgb"])
    augmented["xgb"] = _train_and_score(xgb_aug, X_aug_train, X_aug_test, y_train, y_test)

    # LightGBM
    lgb_base = lgb.LGBMClassifier(**DEFAULT_PARAMS["lgb"])
    baseline["lgb"] = _train_and_score(lgb_base, X_base_train, X_base_test, y_train, y_test)
    lgb_aug = lgb.LGBMClassifier(**DEFAULT_PARAMS["lgb"])
    augmented["lgb"] = _train_and_score(lgb_aug, X_aug_train, X_aug_test, y_train, y_test)

    # Random Forest
    rf_base = RandomForestClassifier(**DEFAULT_PARAMS["rf"])
    baseline["rf"] = _train_and_score(rf_base, X_base_train, X_base_test, y_train, y_test)
    rf_aug = RandomForestClassifier(**DEFAULT_PARAMS["rf"])
    augmented["rf"] = _train_and_score(rf_aug, X_aug_train, X_aug_test, y_train, y_test)

    # CatBoost
    cb_base = CatBoostClassifier(**DEFAULT_PARAMS["cb"])
    baseline["cb"] = _train_and_score(cb_base, X_base_train, X_base_test, y_train, y_test)
    cb_aug = CatBoostClassifier(**DEFAULT_PARAMS["cb"])
    augmented["cb"] = _train_and_score(cb_aug, X_aug_train, X_aug_test, y_train, y_test)

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(xgb_base, save_dir / "xgb_model.joblib")
        joblib.dump(rf_base, save_dir / "rf_model.joblib")
        cb_base.save_model(save_dir / "cb_model.cbm")
        lgb_base.booster_.save_model(str(save_dir / "lgb_model.txt"))
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "seed": seed,
            "test_size": test_size,
            "metrics": {
                "baseline": {k: v.to_dict() for k, v in baseline.items()},
                "augmented": {k: v.to_dict() for k, v in augmented.items()},
            },
        }
        (save_dir / "training_summary.json").write_text(json.dumps(metadata, indent=2))
        logger.info("Saved trained models", extra={"dir": str(save_dir)})

    return TrainingResult(
        timestamp=datetime.now().isoformat(),
        samples_train=len(y_train),
        samples_test=len(y_test),
        test_positive_rate=float(y_test.mean()),
        baseline=baseline,
        augmented=augmented,
    )
