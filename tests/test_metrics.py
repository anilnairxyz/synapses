"""CPU-only tests for the Step 5 metrics, moment extraction, the new data helper, and the
two sweep viz helpers. No GPU; the data-loader test skips cleanly without MNIST on disk.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from synapses.extract import encode_moments
from synapses.losses import vae_loss
from synapses.metrics import active_units, kl_per_dim, linear_probe
from synapses.vae import MLPVAE
from synapses.viz import kl_spectrum_bar, line_plot

DIM, LATENT = 784, 8


def test_kl_per_dim_matches_losses():
    """metrics.kl_per_dim must equal the per-dim KL inside losses.vae_loss."""
    torch.manual_seed(0)
    mu = torch.randn(64, LATENT, dtype=torch.float64)
    logvar = torch.randn(64, LATENT, dtype=torch.float64) * 0.5
    zeros = torch.zeros(64, DIM, dtype=torch.float64)
    from_losses = vae_loss(zeros, zeros, mu, logvar).kl_per_dim
    torch.testing.assert_close(kl_per_dim(mu, logvar), from_losses)


def test_active_units_counts_varying_dims():
    # dim 0 varies a lot, dim 1 is constant, dim 2 varies a little (below threshold).
    mu = torch.stack([
        torch.linspace(-3, 3, 100),          # var ~ 3.06  -> active
        torch.full((100,), 2.0),             # var 0       -> inactive
        torch.linspace(-0.05, 0.05, 100),    # var ~ 8e-4  -> inactive at 0.01
    ], dim=1)
    assert active_units(mu, threshold=0.01) == 1


def test_linear_probe_separable():
    """Two well-separated Gaussian blobs -> a linear probe nails it."""
    torch.manual_seed(0)
    a = torch.randn(200, 2) + torch.tensor([5.0, 5.0])
    b = torch.randn(200, 2) + torch.tensor([-5.0, -5.0])
    x = torch.cat([a, b])
    y = torch.cat([torch.zeros(200), torch.ones(200)]).long()
    acc = linear_probe(x[::2], y[::2], x[1::2], y[1::2])
    assert acc > 0.98


def test_encode_moments_shapes():
    model = MLPVAE(DIM, hidden_dim=32, latent_dim=LATENT).eval()
    x = torch.bernoulli(torch.rand(120, DIM))
    y = torch.randint(0, 10, (120,))
    loader = DataLoader(TensorDataset(x, y), batch_size=32)
    mu, logvar, labels = encode_moments(model, loader, device="cpu")
    assert mu.shape == (120, LATENT)
    assert logvar.shape == (120, LATENT)
    assert labels.shape == (120,)


def test_line_plot_and_kl_spectrum_build():
    ax = line_plot([2, 8, 32], {"a": [1, 2, 3], "b": [3, 2, 1]},
                   xlabel="L", ylabel="v", logx=True, baselines={"base": 2.5})
    assert ax.get_legend() is not None  # 2 series + baseline -> legend

    ax2 = kl_spectrum_bar([0.1, 3.0, 0.0, 1.5])
    assert len(ax2.patches) == 4
    # bars come out sorted descending
    heights = [p.get_height() for p in ax2.patches]
    assert heights == sorted(heights, reverse=True)


def test_fixed_binarized_loader_values():
    pytest.importorskip("torchvision")
    from synapses.data import fixed_binarized_loader
    try:
        loader = fixed_binarized_loader("data", train=False, batch_size=64, download=False)
    except RuntimeError:
        pytest.skip("MNIST not present on disk")
    x, y = next(iter(loader))
    assert x.shape[1] == 784
    assert set(x.unique().tolist()) <= {0.0, 1.0}  # binary
    assert y.dtype == torch.int64
