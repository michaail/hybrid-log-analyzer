"""Preprocessing stage: raw logs → graph-structured data.

Reads raw log files from data/raw/, parses them into structured events,
builds graph representations, and writes processed data to data/processed/.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


def load_raw_logs(raw_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Load all raw log files from *raw_dir* into a single DataFrame."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    frames: list[pd.DataFrame] = []
    for path in sorted(raw_dir.glob("*.csv")):
        frames.append(pd.read_csv(path))
    for path in sorted(raw_dir.glob("*.json")):
        with open(path) as fh:
            frames.append(pd.DataFrame(json.load(fh)))

    if not frames:
        raise FileNotFoundError(f"No CSV/JSON files found in {raw_dir}")

    return pd.concat(frames, ignore_index=True)


def build_graph_data(df: pd.DataFrame) -> dict:
    """Convert a log DataFrame into graph-structured tensors.

    Returns a dict with keys ``node_features``, ``edge_index``, ``labels``
    that downstream stages can load.
    """
    # TODO: implement real graph construction from parsed logs
    return {
        "node_features": df.values.tolist(),
        "edge_index": [],
        "labels": [],
    }


def save_processed(graph_data: dict, out_dir: str | Path = "data/processed") -> None:
    """Persist processed graph data as JSON."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "graph_data.json", "w") as fh:
        json.dump(graph_data, fh)


def main() -> None:
    """Entry-point called by ``dvc repro preprocess``."""
    raw_dir = os.environ.get("RAW_DATA_DIR", "data/raw")
    out_dir = os.environ.get("PROCESSED_DATA_DIR", "data/processed")

    df = load_raw_logs(raw_dir)
    graph_data = build_graph_data(df)
    save_processed(graph_data, out_dir)
    print(f"Preprocessing complete → {out_dir}")


if __name__ == "__main__":
    main()
