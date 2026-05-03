from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def load_yaml(path: str | Path) -> dict[str, Any]:
    with project_path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data
