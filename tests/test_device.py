"""Check actual CPU allocation and both selection branches without a GPU."""

import random

import numpy as np
import pytest
import torch

from src.utils.device import get_device
from src.utils.seed import set_seed


def test_explicit_cpu_allocates_tensor():
    device = get_device("cpu")
    assert torch.ones(2, device=device).device.type == "cpu"


@pytest.mark.parametrize("available,expected", [(False, "cpu"), (True, "cuda")])
def test_auto_device_selection(monkeypatch, available, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    assert get_device("auto").type == expected


def test_explicit_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        get_device("cuda")


def test_explicit_cuda_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert get_device("cuda").type == "cuda"


def test_invalid_device():
    with pytest.raises(ValueError):
        get_device("mps")


def test_seed_repeats_cpu_generators():
    set_seed(42)
    first = (random.random(), np.random.rand(), torch.rand(3))
    set_seed(42)
    second = (random.random(), np.random.rand(), torch.rand(3))
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])
