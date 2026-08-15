"""CPU-only tests for the two-view augmentation pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from synapses.ssl_augment import TwoViewTransform, eval_transform, ssl_transform


def _fake_image() -> Image.Image:
    arr = (np.random.rand(32, 32, 3) * 255).astype("uint8")
    return Image.fromarray(arr)


@pytest.mark.parametrize("strength", ["strong", "weak", "minimal"])
def test_ssl_transform_builds_at_every_strength(strength):
    out = ssl_transform(strength)(_fake_image())
    assert out.shape == (3, 32, 32)


def test_ssl_transform_rejects_unknown_strength():
    with pytest.raises(ValueError, match="unknown strength"):
        ssl_transform("extreme")


def test_two_view_transform_gives_two_independently_sampled_views():
    torch.manual_seed(0)
    tfm = TwoViewTransform(ssl_transform("strong"))
    v1, v2 = tfm(_fake_image())
    assert v1.shape == v2.shape == (3, 32, 32)
    assert not torch.allclose(v1, v2)


def test_eval_transform_is_deterministic():
    tfm = eval_transform()
    img = _fake_image()
    torch.testing.assert_close(tfm(img), tfm(img))
