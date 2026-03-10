"""Tests for src/preprocessing.py"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing import build_graph_data, load_raw_logs, save_processed


@pytest.fixture()
def raw_dir(tmp_path: Path) -> Path:
    """Create a temporary raw data directory with a sample CSV."""
    d = tmp_path / "raw"
    d.mkdir()
    df = pd.DataFrame({"timestamp": [1, 2, 3], "level": ["INFO", "WARN", "ERROR"], "msg": ["a", "b", "c"]})
    df.to_csv(d / "sample.csv", index=False)
    return d


def test_load_raw_logs_reads_csv(raw_dir: Path) -> None:
    df = load_raw_logs(raw_dir)
    assert len(df) == 3
    assert "timestamp" in df.columns


def test_load_raw_logs_missing_dir() -> None:
    with pytest.raises(FileNotFoundError):
        load_raw_logs("/nonexistent/path")


def test_build_graph_data_returns_expected_keys(raw_dir: Path) -> None:
    df = load_raw_logs(raw_dir)
    result = build_graph_data(df)
    assert "node_features" in result
    assert "edge_index" in result
    assert "labels" in result


def test_save_processed_creates_file(tmp_path: Path) -> None:
    graph_data = {"node_features": [[1, 2]], "edge_index": [], "labels": [0]}
    out_dir = tmp_path / "processed"
    save_processed(graph_data, out_dir)
    saved = json.loads((out_dir / "graph_data.json").read_text())
    assert saved == graph_data
