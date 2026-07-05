"""YAML-backed configuration loading.

Every Crucible pipeline stage and experiment is driven by a pydantic model
loaded from YAML, so any artifact can be reproduced from its config alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file that must contain a mapping at the top level."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level, got {type(data)}")
    return data


def load_config(model: type[ModelT], path: Path | None, **overrides: Any) -> ModelT:
    """Build ``model`` from a YAML file (if given) plus keyword overrides.

    Overrides with value ``None`` are ignored so CLI flags can default to
    ``None`` and only override when explicitly set.
    """
    raw = load_yaml(path) if path is not None else {}
    raw.update({key: value for key, value in overrides.items() if value is not None})
    return model.model_validate(raw)
