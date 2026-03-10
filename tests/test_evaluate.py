"""Tests for src/evaluate.py"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.evaluate import compute_metrics, save_metrics


def test_compute_metrics_perfect_predictions() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.9, 0.8])
    metrics = compute_metrics(y_true, y_scores)
    assert metrics["auroc"] == 1.0
    assert metrics["f1"] > 0.9


def test_compute_metrics_returns_all_keys() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_scores = np.array([0.3, 0.7, 0.4, 0.6])
    metrics = compute_metrics(y_true, y_scores)
    assert set(metrics.keys()) == {"auroc", "auprc", "f1"}


def test_save_metrics_writes_json(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    save_metrics({"auroc": 0.95, "f1": 0.88}, path)
    loaded = json.loads(path.read_text())
    assert loaded["auroc"] == 0.95
