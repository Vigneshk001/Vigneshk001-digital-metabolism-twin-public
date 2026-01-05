"""Feature contract enforcement."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureContractError(Exception):
    """Raised when feature contract requirements are violated."""


class FeatureContract:
    """Enforce strict feature compatibility between training and inference."""

    def __init__(self, experiment_dir: Path):
        self.experiment_dir = Path(experiment_dir)
        self.expected_features: List[str] = self._load_expected_features()

    def _load_expected_features(self) -> List[str]:
        path = self.experiment_dir / "feature_importance.csv"
        if not path.exists():
            raise FileNotFoundError(f"feature_importance.csv not found: {path}")

        df = pd.read_csv(path)
        if "feature" in df.columns:
            features = df["feature"].tolist()
        else:
            features = df.iloc[:, 0].tolist()

        if not features:
            raise FeatureContractError("No features found in feature_importance.csv")

        logger.info("Loaded feature contract", extra={"count": len(features)})
        return features

    def validate_and_align(self, df: pd.DataFrame) -> pd.DataFrame:
        expected = self.expected_features
        provided = df.columns.tolist()

        missing = sorted(set(expected) - set(provided))
        extra = sorted(set(provided) - set(expected))

        if missing:
            raise FeatureContractError(f"Missing required features ({len(missing)}): {missing}")

        if extra:
            logger.warning("Dropping unexpected features", extra={"extra": extra})
            df = df.drop(columns=extra)

        aligned = df[expected]
        return aligned
