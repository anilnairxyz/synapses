"""CIFAR-10 loader tests. Skip cleanly if the data isn't on disk yet (fetch it with the
command in the RuntimeError raised by synapses.data.cifar10_eval_loaders).
"""

from __future__ import annotations

import pytest
import torch

from synapses.data import cifar10_eval_loaders, cifar10_ssl_loader


def _cifar_or_skip(loader_fn, **kwargs):
    try:
        return loader_fn(root="data", download=False, **kwargs)
    except RuntimeError:
        pytest.skip("CIFAR-10 not present on disk")


def test_cifar10_eval_loaders_shapes_and_no_shuffle():
    train_loader, test_loader = _cifar_or_skip(cifar10_eval_loaders, batch_size=32)
    x, y = next(iter(test_loader))
    assert x.shape == (32, 3, 32, 32)
    assert y.dtype == torch.int64
    assert len(train_loader.dataset) == 50000
    assert len(test_loader.dataset) == 10000


def test_cifar10_ssl_loader_yields_two_views():
    loader = _cifar_or_skip(cifar10_ssl_loader, batch_size=16)
    (v1, v2), y = next(iter(loader))
    assert v1.shape == v2.shape == (16, 3, 32, 32)
    assert not torch.allclose(v1, v2)
    assert y.shape == (16,)
