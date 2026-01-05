#!/usr/bin/env python3
"""Run inference with saved models."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config.paths import ProjectPaths, load_config
from src.main import run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference")
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
    run_inference(paths, config, args.experiment, args.model_type)


if __name__ == "__main__":
    main()
