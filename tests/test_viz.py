"""CPU-only tests for the Step 2 helpers: load_vae, encode_latents, latent_scatter.

No GPU, no data download — everything the Mac can validate. Run (on the PC):
    uv run pytest tests/ -v
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless

from dataclasses import dataclass

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from synapses.extract import encode_latents
from synapses.run import save_checkpoint
from synapses.vae import MLPVAE, load_vae
from synapses.viz import DIGIT_PALETTE, latent_scatter

DIM, LATENT = 784, 2


@dataclass
class _Cfg:
    """Stand-in for Phase1Config. A dataclass, like the real one — so save_checkpoint's
    _to_plain serializes it to a dict, which is what load_vae reads back."""

    latent_dim: int
    hidden_dim: int


def _fake_loader(n=200, n_classes=10, batch_size=64):
    x = torch.bernoulli(torch.rand(n, DIM))
    y = torch.randint(0, n_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def test_load_vae_roundtrip(tmp_path):
    """save_checkpoint then load_vae reproduces the model exactly, in eval mode."""
    torch.manual_seed(0)
    model = MLPVAE(DIM, hidden_dim=64, latent_dim=LATENT)
    opt = torch.optim.Adam(model.parameters())
    ckpt = tmp_path / "epoch_0001.pt"
    save_checkpoint(ckpt, model, opt, epoch=1, config=_Cfg(LATENT, 64))

    loaded, cfg = load_vae(str(ckpt), device="cpu")
    assert cfg["latent_dim"] == LATENT and cfg["hidden_dim"] == 64
    assert loaded.training is False
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert k1 == k2
        torch.testing.assert_close(v1, v2)


def test_encode_latents_shapes_and_labels():
    model = MLPVAE(DIM, hidden_dim=32, latent_dim=LATENT).eval()
    loader = _fake_loader(n=200)
    mu, labels = encode_latents(model, loader, device="cpu")
    assert mu.shape == (200, LATENT)
    assert labels.shape == (200,)
    assert mu.device.type == "cpu"
    # labels must come back in loader order, unchanged
    all_y = torch.cat([y for _, y in loader])
    torch.testing.assert_close(labels, all_y)


def test_encode_latents_mu_deterministic_sample_stochastic():
    """sample=False returns fixed mu; sample=True injects reparameterization noise."""
    torch.manual_seed(0)
    model = MLPVAE(DIM, hidden_dim=32, latent_dim=LATENT).eval()
    loader = _fake_loader(n=128)

    mu_a, _ = encode_latents(model, loader, device="cpu", sample=False)
    mu_b, _ = encode_latents(model, loader, device="cpu", sample=False)
    torch.testing.assert_close(mu_a, mu_b)  # mu path is deterministic

    z_a, _ = encode_latents(model, loader, device="cpu", sample=True)
    z_b, _ = encode_latents(model, loader, device="cpu", sample=True)
    assert not torch.allclose(z_a, z_b)  # sampled path is not


def test_palette_has_ten_distinct_colors():
    assert len(DIGIT_PALETTE) == 10
    assert len(set(DIGIT_PALETTE)) == 10


def test_latent_scatter_builds_with_legend_and_prior():
    torch.manual_seed(0)
    mu = torch.randn(500, 2)
    labels = torch.randint(0, 10, (500,))
    ax = latent_scatter(mu, labels, prior_sigmas=(1.0, 2.0))

    # one legend entry per class present
    assert len(ax.get_legend().get_texts()) == len(torch.unique(labels))
    # two prior circles drawn as patches
    from matplotlib.patches import Circle

    assert sum(isinstance(p, Circle) for p in ax.patches) == 2


def test_latent_scatter_rejects_non_2d():
    with pytest.raises(ValueError, match="2D latents"):
        latent_scatter(torch.randn(10, 8), torch.zeros(10, dtype=torch.long))
