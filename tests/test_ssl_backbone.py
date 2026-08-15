"""CPU-only tests for the CIFAR-adapted ResNet-18 backbone."""

from __future__ import annotations

import torch

from synapses.ssl_backbone import CifarResNet18


def test_forward_shape():
    model = CifarResNet18()
    out = model(torch.randn(4, 3, 32, 32))
    assert out.shape == (4, 512)


def test_out_dim_is_configurable():
    model = CifarResNet18(out_dim=128)
    out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 128)
    assert model.out_dim == 128


def test_stem_preserves_spatial_resolution():
    """The CIFAR stem is 3x3 stride-1 with no maxpool: 32x32 in, 32x32 out of the stem."""
    model = CifarResNet18()
    h = model.stem(torch.randn(1, 3, 32, 32))
    assert h.shape[-2:] == (32, 32)


def test_gradient_reaches_the_stem():
    model = CifarResNet18()
    model(torch.randn(2, 3, 32, 32)).sum().backward()
    grad = model.stem[0].weight.grad
    assert grad is not None
    assert grad.abs().sum() > 0
