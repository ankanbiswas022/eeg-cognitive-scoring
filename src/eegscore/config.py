from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML experiment config (defaults to configs/default.yaml)."""
    path = Path(path) if path else PROJECT_ROOT / "configs" / "default.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def resolve(path: str | Path) -> Path:
    """Resolve a path relative to the project root unless it is absolute."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
