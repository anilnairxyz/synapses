"""Verify nt_xent_loss by hand, the same treatment vae_loss got in Phase 1.

A brute-force, unvectorized reference implementation checks that the vectorized
masked-softmax trick really computes the textbook per-pair formula -- two independently
written code paths must agree. Run: uv run pytest tests/test_ssl_losses.py -v
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from synapses.ssl_losses import nt_xent_loss


def _naive_nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """Directly follows the l_{i,j} formula with an explicit double loop -- no masking
    trick, no batched matmul. Independent of nt_xent_loss's implementation."""
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    n2 = z.size(0)
    n = z1.size(0)
    losses = []
    for i in range(n2):
        pos = (i + n) % n2
        numerator = torch.exp(z[i] @ z[pos] / temperature)
        denom = sum(
            torch.exp(z[i] @ z[k] / temperature) for k in range(n2) if k != i
        )
        losses.append(-torch.log(numerator / denom))
    return torch.stack(losses).mean()


@pytest.fixture
def small_views():
    torch.manual_seed(0)
    n, d = 6, 8
    z1 = torch.randn(n, d, dtype=torch.float64)
    z2 = torch.randn(n, d, dtype=torch.float64)
    return z1, z2


def test_matches_naive_reference_implementation(small_views):
    z1, z2 = small_views
    vectorized = nt_xent_loss(z1, z2, temperature=0.3)
    naive = _naive_nt_xent(z1, z2, temperature=0.3)
    torch.testing.assert_close(vectorized, naive)


def test_random_init_matches_log_2n_minus_1():
    """At random init, similarities concentrate near 0, so softmax over 2N-1 roughly-equal
    logits is ~uniform -- the loss should land near log(2N-1). This is the same invariant
    train_simclr.py's own init sanity check looks for on real data."""
    torch.manual_seed(0)
    n, d = 64, 128  # large-ish D so random cosine similarities concentrate near 0
    z1 = torch.randn(n, d)
    z2 = torch.randn(n, d)
    loss = nt_xent_loss(z1, z2, temperature=0.5)
    expected = math.log(2 * n - 1)
    assert loss.item() == pytest.approx(expected, abs=0.3)


def test_near_zero_for_near_perfect_positives():
    """Orthogonal anchors, z2 == z1 (positives match exactly, negatives orthogonal):
    softmax overwhelmingly picks the true positive, so the loss should be near 0."""
    n = 4
    z1 = torch.eye(n, dtype=torch.float64)  # orthogonal unit vectors
    z2 = z1.clone()
    loss = nt_xent_loss(z1, z2, temperature=0.1)
    assert loss.item() < 0.05


def test_symmetric_in_the_two_views(small_views):
    z1, z2 = small_views
    torch.testing.assert_close(nt_xent_loss(z1, z2), nt_xent_loss(z2, z1))


def test_gradient_reaches_both_views(small_views):
    z1, z2 = small_views
    z1 = z1.clone().requires_grad_(True)
    z2 = z2.clone().requires_grad_(True)
    nt_xent_loss(z1, z2).backward()
    assert z1.grad is not None and z1.grad.abs().sum() > 0
    assert z2.grad is not None and z2.grad.abs().sum() > 0
