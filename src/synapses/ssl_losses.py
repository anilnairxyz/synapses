"""Self-supervised losses for Phase 2.

Written by hand and unit-tested against independent reference computations, the same
treatment ``vae_loss`` got in Phase 1 -- these are the objects of study, not library calls.
Home for SimCLR's NT-Xent, SimSiam's symmetrised negative cosine, and VICReg's
variance-invariance-covariance loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """NT-Xent (SimCLR) contrastive loss over a batch of paired views.

    For each of the ``2N`` views (``z1`` and ``z2`` concatenated), the loss is a ``2N``-way
    classification: identify which of the other ``2N-1`` views is this view's positive
    (the other augmentation of the same source image). Implemented as cross-entropy over
    a cosine-similarity matrix, with the diagonal (self-similarity) masked out -- this
    routes through ``F.cross_entropy``'s internal log-sum-exp, which is the numerically
    stable way to compute the textbook ``-log(exp(sim_pos/tau) / sum_k exp(sim_k/tau))``
    formula (a naive ``exp(...) / sum(exp(...))`` can overflow).

    Args:
        z1, z2: ``(N, D)`` projector outputs for view 1 and view 2 of the same N images.
        temperature: tau in the formula above.

    Returns:
        A scalar: the mean of the ``2N`` per-anchor losses.
    """
    n = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2N, D)
    z = F.normalize(z, dim=1)

    sim = (z @ z.t()) / temperature  # (2N, 2N) cosine similarity / tau
    mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, float("-inf"))  # exclude self-similarity, k != i

    # Row i's positive is the other view of the same image: i -> i+N, and i+N -> i.
    pos_idx = (torch.arange(2 * n, device=z.device) + n) % (2 * n)
    return F.cross_entropy(sim, pos_idx)


def simsiam_loss(
    p1: torch.Tensor,
    z1: torch.Tensor,
    p2: torch.Tensor,
    z2: torch.Tensor,
    stop_gradient: bool = True,
) -> torch.Tensor:
    """SimSiam's symmetrised negative cosine similarity.

    ``D(p, z) = -cosine_sim(p, z)``, evaluated with ``z`` detached from the graph when
    ``stop_gradient`` is True. The loss is ``0.5 * (D(p1, z2) + D(p2, z1))``: each
    branch's *prediction* is matched against the *other* branch's *target*.

    ``stop_gradient`` is deliberately exposed as a flag, not hardcoded to True: this
    single boolean **is** the "remove the stop-gradient" break-it experiment. No
    negatives, no EMA, no queue -- collapse is avoided (when it is avoided) purely by
    this asymmetry between the predicted and the target branch, so being able to switch
    it off in one place is what makes the mechanism directly testable.

    Args:
        p1, p2: predictor outputs for view 1 and view 2, ``(N, D)``.
        z1, z2: projector outputs (pre-predictor) for view 1 and view 2, ``(N, D)``.
        stop_gradient: detach the target branch (the real SimSiam) if True; leave the
            graph intact (the ablation) if False.

    Returns:
        A scalar in ``[-1, 1]``.
    """

    def neg_cos_sim(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if stop_gradient:
            z = z.detach()
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p * z).sum(dim=1).mean()

    return 0.5 * (neg_cos_sim(p1, z2) + neg_cos_sim(p2, z1))


def _variance_term(z: torch.Tensor, gamma: float, eps: float) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0, unbiased=True) + eps)  # eps avoids a NaN grad at std=0
    return F.relu(gamma - std).mean()


def _covariance_term(z: torch.Tensor) -> torch.Tensor:
    n, d = z.shape
    z_c = z - z.mean(dim=0, keepdim=True)
    cov = (z_c.t() @ z_c) / (n - 1)
    off_diag_sq_sum = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return off_diag_sq_sum / d


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    inv_weight: float = 25.0,
    var_weight: float = 25.0,
    cov_weight: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """VICReg: Variance-Invariance-Covariance Regularisation.

    Three independent terms, no negatives, no predictor, no stop-gradient -- both
    branches are perfectly symmetric, and the entire anti-collapse mechanism lives in
    the variance term below.

    * **Invariance** -- plain MSE between the two views, ``F.mse_loss(z1, z2)``. Unlike
      SimCLR/SimSiam's cosine-based losses, this is *not* scale-invariant: shrinking z1
      and z2 together *would* reduce this term, so nothing here alone prevents a
      magnitude collapse -- that job belongs entirely to the variance term below.
    * **Variance** -- for each branch, hinge the per-dimension std across the batch
      against a floor ``gamma``: ``mean_j(max(0, gamma - std_j))``. This is the anti-
      collapse mechanism: a dimension whose variance drops below ``gamma`` is directly
      penalised, regardless of what the invariance term wants. ``var_weight=0`` removes
      it entirely -- the break-it experiment for this method, no separate flag needed.
    * **Covariance** -- for each branch, the off-diagonal squared entries of the batch
      covariance matrix, divided by the feature dimension. Penalises different
      dimensions for carrying redundant (correlated) information.

    Args:
        z1, z2: expander outputs for view 1 and view 2, ``(N, D)``. Unlike SimSiam's
            projector, VICReg's expander must end on a plain ``Linear`` with no
            batch norm -- see ``train_vicreg.py`` -- so these are genuinely
            unconstrained; a normalised final layer would satisfy the variance term
            structurally and make ``var_weight`` vacuous.
        gamma: the per-dimension standard-deviation floor.
        eps: numerical floor inside the variance term's square root.

    Returns:
        ``(loss, terms)`` -- the weighted scalar total, and a dict of the three
        unweighted term values (for logging each one separately; a single scalar would
        hide the variance term collapsing toward zero contribution under the ablation).
    """
    inv = F.mse_loss(z1, z2)
    var = 0.5 * (_variance_term(z1, gamma, eps) + _variance_term(z2, gamma, eps))
    cov = 0.5 * (_covariance_term(z1) + _covariance_term(z2))

    loss = inv_weight * inv + var_weight * var + cov_weight * cov
    return loss, {"inv": inv, "var": var, "cov": cov}
