"""Feature engineering helpers for dietary + clinical data."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

NUTRIENTS = ["TKCAL", "TPROT", "TTFAT", "TCARB", "TSUGR", "TFIBE"]


def load_day(file_path: Path, prefix: str) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    df = pd.read_parquet(file_path)
    cols = [f"DR{prefix}{nut}" for nut in NUTRIENTS if f"DR{prefix}{nut}" in df.columns]
    cols = ["SEQN"] + cols
    df = df[cols].copy()
    rename_map = {c: f"day{1 if prefix=='1' else 2}_{c[4:].lower()}" for c in cols if c != "SEQN"}
    return df.rename(columns=rename_map)


def build_diet_daily(cache_dir: Path, out_path: Path) -> pd.DataFrame:
    dr1 = load_day(cache_dir / "nhanes_DR1TOT.parquet", "1")
    dr2 = load_day(cache_dir / "nhanes_DR2TOT.parquet", "2")
    df = dr1.merge(dr2, on="SEQN", how="left").fillna(0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Generated diet_daily", extra={"rows": len(df), "cols": len(df.columns)})
    return df
