"""Verify nt_xent_loss, simsiam_loss, and vicreg_loss by hand, the same treatment
vae_loss got in Phase 1. A brute-force, unvectorized reference implementation checks
that the vectorized masked-softmax trick really computes the textbook per-pair formula
-- two independently written code paths must agree. Run: uv run pytest tests/test_ssl_losses.py -v
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from synapses.ssl_losses import (
    _covariance_term,
    _variance_term,
    nt_xent_loss,
    simsiam_loss,
    vicreg_loss,
)


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


# --- simsiam_loss -----------------------------------------------------------------


@pytest.fixture
def small_pz():
    torch.manual_seed(1)
    n, d = 6, 8
    p1 = torch.randn(n, d, dtype=torch.float64)
    z1 = torch.randn(n, d, dtype=torch.float64)
    p2 = torch.randn(n, d, dtype=torch.float64)
    z2 = torch.randn(n, d, dtype=torch.float64)
    return p1, z1, p2, z2


def test_simsiam_loss_bounded():
    torch.manual_seed(0)
    p1, z1, p2, z2 = (torch.randn(10, 16, dtype=torch.float64) for _ in range(4))
    loss = simsiam_loss(p1, z1, p2, z2)
    assert -1.0 <= loss.item() <= 1.0


def test_simsiam_loss_minus_one_for_identical_normalised_vectors():
    """p == z (after normalisation) -> perfect alignment -> D(p,z) = -1 exactly."""
    n, d = 5, 8
    v = torch.randn(n, d, dtype=torch.float64)
    loss = simsiam_loss(v, v, v, v)
    assert loss.item() == pytest.approx(-1.0, abs=1e-9)


def test_simsiam_loss_zero_for_orthogonal_vectors():
    """p perp z -> cosine similarity 0 -> D(p,z) = 0 exactly."""
    n = 4
    basis = torch.eye(2 * n, dtype=torch.float64)
    p = basis[:n]  # first n basis vectors
    z = basis[n:]  # the other n, orthogonal to every row of p
    loss = simsiam_loss(p, z, p, z)
    assert loss.item() == pytest.approx(0.0, abs=1e-9)


def test_simsiam_loss_near_zero_at_random_init():
    """Random, uncorrelated p/z: cosine similarity concentrates near 0 in high-D, so the
    loss should read near 0 -- the init-sanity-check value train_simsiam.py looks for.
    Explicitly NOT nt_xent_loss's log(2N-1): a different loss, a different invariant."""
    torch.manual_seed(0)
    n, d = 64, 128
    p1, z1, p2, z2 = (torch.randn(n, d) for _ in range(4))
    loss = simsiam_loss(p1, z1, p2, z2)
    assert loss.item() == pytest.approx(0.0, abs=0.1)


def test_simsiam_loss_symmetric_in_the_two_branches(small_pz):
    p1, z1, p2, z2 = small_pz
    torch.testing.assert_close(simsiam_loss(p1, z1, p2, z2), simsiam_loss(p2, z2, p1, z1))


def test_stop_gradient_blocks_gradient_to_target_branch(small_pz):
    """The mechanism under study, tested directly: with stop_gradient=True, gradients
    must reach the predictor outputs (p1, p2) but must NOT reach the target/stopped
    branch (z1, z2). This is the single most important test in this module -- it is
    what makes the break-it experiment (stop_gradient=False) meaningful at all."""
    p1, z1, p2, z2 = (t.clone().requires_grad_(True) for t in small_pz)
    simsiam_loss(p1, z1, p2, z2, stop_gradient=True).backward()

    assert p1.grad is not None and p1.grad.abs().sum() > 0
    assert p2.grad is not None and p2.grad.abs().sum() > 0
    assert z1.grad is None, "stop_gradient=True must block gradient into z1"
    assert z2.grad is None, "stop_gradient=True must block gradient into z2"


def test_no_stop_gradient_lets_gradient_reach_every_branch(small_pz):
    """The break-it ablation, verified: stop_gradient=False must let gradients reach
    all four tensors, confirming the flag genuinely removes the asymmetry."""
    p1, z1, p2, z2 = (t.clone().requires_grad_(True) for t in small_pz)
    simsiam_loss(p1, z1, p2, z2, stop_gradient=False).backward()

    for name, t in [("p1", p1), ("z1", z1), ("p2", p2), ("z2", z2)]:
        assert t.grad is not None and t.grad.abs().sum() > 0, f"{name} got no gradient"


# --- vicreg_loss -------------------------------------------------------------------


def test_variance_term_hand_computed():
    """N=2 symmetric samples [c-d, c+d] have unbiased std = d*sqrt(2) (derived
    independently of the implementation: sum of squared deviations = 2d^2, divided by
    N-1=1). Build one dimension with std exactly 2 (hinge contributes 0, >= gamma) and
    one with std exactly 0.5 (hinge contributes 0.5, < gamma), and check the mean hinge
    against a value computed by hand."""
    gamma = 1.0
    d1 = 2.0 / math.sqrt(2)  # dimension 1: std = 2.0
    d2 = 0.5 / math.sqrt(2)  # dimension 2: std = 0.5
    z = torch.tensor([[-d1, -d2], [d1, d2]], dtype=torch.float64)

    result = _variance_term(z, gamma=gamma, eps=0.0)
    expected = (max(0.0, gamma - 2.0) + max(0.0, gamma - 0.5)) / 2  # (0 + 0.5) / 2
    assert result.item() == pytest.approx(expected, abs=1e-6)


def test_covariance_term_exact_for_perfectly_correlated_dims():
    """z2 = 2*z1 exactly: Cov(z1, 2*z1) = 2*Var(z1) by bilinearity of covariance, an
    independently-derived closed form. For a 2x2 covariance matrix the off-diagonal
    sum-of-squares is 2*cov12^2 (both symmetric entries); dividing by D=2 gives cov12^2."""
    z1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    z2 = 2 * z1
    z = torch.stack([z1, z2], dim=1)

    z1c = z1 - z1.mean()
    var1 = (z1c**2).sum() / (5 - 1)
    cov12 = 2 * var1  # Cov(z1, 2*z1) = 2*Var(z1)
    expected = cov12**2

    result = _covariance_term(z)
    assert result.item() == pytest.approx(expected.item(), rel=1e-9)


def test_covariance_term_smaller_for_independent_dims():
    torch.manual_seed(0)
    x = torch.randn(300, dtype=torch.float64)
    correlated = torch.stack([x, 2 * x], dim=1)
    independent = torch.randn(300, 2, dtype=torch.float64)
    assert _covariance_term(independent).item() < _covariance_term(correlated).item()


def test_variance_term_no_nan_at_exactly_zero_variance():
    """The numerical-stability case eps exists for: every sample identical (var=0
    exactly). sqrt'(0) is infinite without eps, which would poison the gradient."""
    z = torch.ones(10, 4, dtype=torch.float64, requires_grad=True)
    loss = _variance_term(z, gamma=1.0, eps=1e-4)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(z.grad).all()


def test_var_weight_zero_removes_the_variance_term():
    """The break-it mechanism, verified directly: with var_weight=0, the returned total
    must equal inv_weight*inv + cov_weight*cov exactly, even when the variance term
    itself is large -- proof it is genuinely excluded from the sum, not just small."""
    torch.manual_seed(0)
    # near-constant across the batch -> variance term is large if it's included at all
    z1 = 0.001 * torch.randn(50, 8, dtype=torch.float64)
    z2 = torch.randn(50, 8, dtype=torch.float64)

    loss, terms = vicreg_loss(z1, z2, inv_weight=25.0, var_weight=0.0, cov_weight=1.0)
    # z1 is near-collapsed (hinge ~1) and z2 is healthy (hinge ~0); the term averages
    # both branches, so ~0.5 is expected -- comfortably above "negligible."
    assert terms["var"].item() > 0.3, "sanity check: variance really is large here"
    expected = 25.0 * terms["inv"] + 1.0 * terms["cov"]
    assert loss.item() == pytest.approx(expected.item(), rel=1e-9)


def test_vicreg_terms_are_non_negative():
    torch.manual_seed(0)
    z1 = torch.randn(30, 6, dtype=torch.float64)
    z2 = torch.randn(30, 6, dtype=torch.float64)
    _, terms = vicreg_loss(z1, z2)
    assert terms["inv"].item() >= 0
    assert terms["var"].item() >= 0
    assert terms["cov"].item() >= 0


def test_vicreg_gradient_reaches_both_views():
    torch.manual_seed(0)
    z1 = torch.randn(30, 6, dtype=torch.float64, requires_grad=True)
    z2 = torch.randn(30, 6, dtype=torch.float64, requires_grad=True)
    loss, _ = vicreg_loss(z1, z2)
    loss.backward()
    assert z1.grad is not None and z1.grad.abs().sum() > 0
    assert z2.grad is not None and z2.grad.abs().sum() > 0
