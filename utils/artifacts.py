from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.config import ARTIFACTS_DIR, MODEL_DIR


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def dataset_stats() -> dict[str, Any] | None:
    return read_json(ARTIFACTS_DIR / "dataset_stats.json")


def original_run() -> dict[str, Any] | None:
    return read_json(ARTIFACTS_DIR / "original_run.json")


def test_metrics() -> dict[str, Any] | None:
    return read_json(ARTIFACTS_DIR / "test_metrics.json")


def training_history() -> dict[str, Any] | None:
    return read_json(ARTIFACTS_DIR / "training_history.json")


def fp16_equivalence() -> dict[str, Any] | None:
    return read_json(ARTIFACTS_DIR / "fp16_equivalence.json")


def model_card() -> dict[str, Any] | None:
    return read_json(MODEL_DIR / "model_card.json")


def misclassifications() -> dict[str, Any] | None:
    return read_json(ARTIFACTS_DIR / "misclassifications.json")


def deployed_accuracy() -> float | None:
    metrics = test_metrics()
    if not metrics:
        return None
    value = metrics.get("metrics", {}).get("Accuracy")
    return float(value) if value is not None else None


def original_best() -> dict[str, Any] | None:
    data = original_run()
    if not data:
        return None
    rows = data.get("comparison", [])
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("Accuracy", 0)))


def metric_value(metrics: dict[str, Any] | None, key: str) -> float | None:
    if not metrics:
        return None
    value = metrics.get("metrics", {}).get(key)
    return float(value) if value is not None else None


def format_percent(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "Pending"
    return f"{value * 100:.{digits}f}%"


def evaluated_date() -> str | None:
    metrics = test_metrics()
    if not metrics:
        return None
    value = metrics.get("evaluated_at")
    if not value:
        return None
    return str(value)[:10]
