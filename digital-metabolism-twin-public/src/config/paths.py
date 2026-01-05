"""Configuration and path utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ProjectPaths:
    """Container for well-defined project-relative paths."""

    root: Path
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    cache_dir: Path
    results_dir: Path
    models_dir: Path

    @classmethod
    def from_config(cls, root: Path, config: Dict[str, Any]) -> "ProjectPaths":
        paths_cfg = config.get("paths", {})
        return cls(
            root=root,
            data_dir=root / paths_cfg.get("data_dir", "data"),
            raw_dir=root / paths_cfg.get("raw_dir", "data/raw"),
            processed_dir=root / paths_cfg.get("processed_dir", "data/processed"),
            cache_dir=root / paths_cfg.get("cache_dir", "data/processed/cache"),
            results_dir=root / paths_cfg.get("results_dir", "results"),
            models_dir=root / paths_cfg.get("models_dir", "models/saved"),
        )

    def ensure(self) -> None:
        for path in [self.data_dir, self.raw_dir, self.processed_dir, self.cache_dir, self.results_dir, self.models_dir]:
            path.mkdir(parents=True, exist_ok=True)
            logger.debug("Ensured path exists", extra={"path": str(path)})


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.debug("Loaded config", extra={"config_path": str(config_path)})
    return config
