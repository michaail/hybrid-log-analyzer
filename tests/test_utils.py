"""Tests for src/utils.py"""

from __future__ import annotations

import torch

from src.utils import get_device, is_ci, seed_everything


def test_is_ci_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("CI", "true")
    assert is_ci() is True

    monkeypatch.setenv("CI", "false")
    assert is_ci() is False


def test_seed_everything_is_deterministic() -> None:
    seed_everything(123)
    a = torch.rand(5)
    seed_everything(123)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_get_device_returns_valid() -> None:
    device = get_device()
    assert isinstance(device, torch.device)
