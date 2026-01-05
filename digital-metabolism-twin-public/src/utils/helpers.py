"""Helper utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List


def hash_files(paths: Iterable[Path], limit: int = 5) -> str:
    hasher = hashlib.sha256()
    for path in list(paths)[:limit]:
        if path.exists():
            hasher.update(path.read_bytes())
    return hasher.hexdigest()[:16]


def list_model_files(model_dir: Path) -> List[Path]:
    patterns = ["*.joblib", "*.txt", "*.cbm", "*.pt", "*.onnx"]
    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted(model_dir.glob(pat)))
    return files
