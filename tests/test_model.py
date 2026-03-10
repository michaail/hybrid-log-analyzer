"""Tests for src/model.py"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.model import load_checkpoint, save_checkpoint


def test_save_and_load_checkpoint(tmp_path: Path) -> None:
    """Round-trip test: save a checkpoint and load it back."""
    # Use a simple linear model to avoid torch-geometric dependency in tests
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    path = str(tmp_path / "test_ckpt.pt")

    save_checkpoint(model, optimizer, epoch=5, loss=0.42, metrics={"f1": 0.8}, path=path)
    ckpt = load_checkpoint(path)

    assert ckpt["epoch"] == 5
    assert abs(ckpt["loss"] - 0.42) < 1e-6
    assert ckpt["metrics"]["f1"] == 0.8
    assert "model_state_dict" in ckpt
    assert "optimizer_state_dict" in ckpt


def test_load_checkpoint_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint("/nonexistent/checkpoint.pt")
