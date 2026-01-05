"""Model loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SecureModelLoader:
    """Load trained models and preprocessing artifacts with light validation."""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.models: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self._validate_directory()
        self._load_metadata()

    def _validate_directory(self) -> None:
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        summary = self.model_dir / "training_summary.json"
        if not summary.exists():
            raise FileNotFoundError(f"Missing required file: {summary}")

    def _load_metadata(self) -> None:
        with open(self.model_dir / "training_summary.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        logger.info("Loaded training metadata", extra={"model_dir": str(self.model_dir)})

    def load_xgboost(self, filename: str = "xgb_model.joblib"):
        path = self.model_dir / filename
        model = joblib.load(path)
        if not hasattr(model, "predict_proba"):
            raise TypeError("Loaded object does not look like a classifier")
        self.models["xgb"] = model
        return model

    def load_random_forest(self, filename: str = "rf_model.joblib"):
        path = self.model_dir / filename
        model = joblib.load(path)
        self.models["rf"] = model
        return model

    def load_lightgbm(self, filename: str = "lgb_model.txt"):
        import lightgbm as lgb

        path = self.model_dir / filename
        model = lgb.Booster(model_file=str(path))
        self.models["lgb"] = model
        return model

    def load_catboost(self, filename: str = "cb_model.cbm"):
        from catboost import CatBoostClassifier

        path = self.model_dir / filename
        model = CatBoostClassifier()
        model.load_model(str(path))
        self.models["cb"] = model
        return model

    def load_vae_checkpoint(self, filename: str = "vae_model.pt") -> Dict[str, Any]:
        path = self.model_dir / filename
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if "model_state_dict" not in checkpoint:
            raise ValueError("Invalid VAE checkpoint format")
        self.models["vae"] = checkpoint
        return checkpoint

    def list_loaded_models(self) -> List[str]:
        return list(self.models.keys())
