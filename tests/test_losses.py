"""Verify the hand-derived VAE loss numerically.

``synapses/losses.py`` writes the Gaussian KL and the Bernoulli likelihood out by hand
rather than calling ``torch.distributions``. These tests check the derivations against
the library versions, so a sign error surfaces here instead of as a mysteriously
collapsing model three hours later.

Run (on the PC):  uv run pytest tests/ -v
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli, Normal, kl_divergence

from synapses.losses import vae_loss
from synapses.vae import MLPVAE

BATCH, DIM, LATENT = 16, 784, 8


@pytest.fixture
def batch():
    torch.manual_seed(0)
    logits = torch.randn(BATCH, DIM, dtype=torch.float64)
    x = torch.bernoulli(torch.rand(BATCH, DIM, dtype=torch.float64))
    mu = torch.randn(BATCH, LATENT, dtype=torch.float64)
    logvar = torch.randn(BATCH, LATENT, dtype=torch.float64) * 0.5
    return logits, x, mu, logvar


def test_kl_matches_torch_distributions(batch):
    """Our closed form == KL(N(mu, sigma^2) || N(0, I)) from torch.distributions."""
    _, _, mu, logvar = batch
    zeros = torch.zeros(BATCH, DIM, dtype=torch.float64)
    out = vae_loss(zeros, zeros, mu, logvar)

    q = Normal(mu, torch.exp(0.5 * logvar))
    p = Normal(torch.zeros_like(mu), torch.ones_like(mu))
    expected_per_dim = kl_divergence(q, p).mean(dim=0)

    torch.testing.assert_close(out.kl_per_dim, expected_per_dim)
    torch.testing.assert_close(out.kl, expected_per_dim.sum())


def test_recon_matches_bernoulli_log_prob(batch):
    """The BCE term == -log p(x|z) under a Bernoulli likelihood."""
    logits, x, mu, logvar = batch
    out = vae_loss(logits, x, mu, logvar)

    expected = -Bernoulli(logits=logits).log_prob(x).sum(dim=1).mean()
    torch.testing.assert_close(out.recon, expected)


def test_recon_is_summed_over_pixels_not_averaged(batch):
    """Guard against the reduction='mean' trap.

    With uniform logits the reconstruction term must be 784 * ln2 ~ 543.4 nats. If the
    loss ever averages over pixels this drops to ~0.693 and beta becomes meaningless.
    """
    _, x, mu, logvar = batch
    out = vae_loss(torch.zeros_like(x), x, mu, logvar)
    assert out.recon.item() == pytest.approx(DIM * math.log(2), rel=1e-9)


def test_beta_weights_only_the_training_objective(batch):
    """beta scales `loss` but leaves recon, kl and the reported ELBO untouched."""
    logits, x, mu, logvar = batch
    a = vae_loss(logits, x, mu, logvar, beta=1.0)
    b = vae_loss(logits, x, mu, logvar, beta=4.0)

    torch.testing.assert_close(a.recon, b.recon)
    torch.testing.assert_close(a.kl, b.kl)
    torch.testing.assert_close(a.elbo, b.elbo)
    torch.testing.assert_close(b.loss, a.recon + 4.0 * a.kl)


def test_kl_is_zero_for_standard_normal_posterior():
    """q == p should give exactly zero rate."""
    mu = torch.zeros(BATCH, LATENT, dtype=torch.float64)
    logvar = torch.zeros(BATCH, LATENT, dtype=torch.float64)
    x = torch.zeros(BATCH, DIM, dtype=torch.float64)
    out = vae_loss(torch.zeros_like(x), x, mu, logvar)
    assert out.kl.abs().item() < 1e-12


def test_model_shapes_and_gradients():
    """Forward pass shapes, and that gradients reach both encoder heads."""
    torch.manual_seed(0)
    model = MLPVAE(input_dim=DIM, hidden_dim=32, latent_dim=LATENT)
    x = torch.bernoulli(torch.rand(BATCH, DIM))

    logits, mu, logvar = model(x)
    assert logits.shape == (BATCH, DIM)
    assert mu.shape == (BATCH, LATENT)
    assert logvar.shape == (BATCH, LATENT)

    vae_loss(logits, x, mu, logvar).loss.backward()
    assert model.fc_mu.weight.grad is not None
    assert model.fc_logvar.weight.grad is not None
    assert model.fc_logvar.weight.grad.abs().sum() > 0, "no gradient through the variance head"


def test_decoder_returns_logits_not_probabilities():
    """reconstruct() applies the sigmoid; decode() must not."""
    torch.manual_seed(0)
    model = MLPVAE(input_dim=DIM, hidden_dim=32, latent_dim=LATENT)
    x = torch.bernoulli(torch.rand(BATCH, DIM))

    probs = model.reconstruct(x)
    assert probs.min() >= 0.0 and probs.max() <= 1.0

    logits = model.decode(torch.randn(BATCH, LATENT) * 5.0)
    assert logits.min() < 0.0, "decode() looks bounded — is a sigmoid applied twice?"


def test_reparameterize_is_stochastic_and_centred():
    """z = mu + sigma * eps: different draws, right mean and variance."""
    torch.manual_seed(0)
    model = MLPVAE(input_dim=DIM, hidden_dim=32, latent_dim=LATENT)
    mu = torch.full((100_000, 1), 2.0)
    logvar = torch.full((100_000, 1), math.log(9.0))  # sigma = 3

    z = model.reparameterize(mu, logvar)
    assert not torch.allclose(z[:2], z[2:4])
    assert z.mean().item() == pytest.approx(2.0, abs=0.05)
    assert z.std().item() == pytest.approx(3.0, abs=0.05)


def test_binary_cross_entropy_with_logits_is_stable_at_extremes():
    """The fused sigmoid must not produce inf/nan where a manual sigmoid+log would."""
    logits = torch.tensor([[-100.0, 100.0]])
    x = torch.tensor([[0.0, 1.0]])
    loss = F.binary_cross_entropy_with_logits(logits, x, reduction="none")
    assert torch.isfinite(loss).all()
