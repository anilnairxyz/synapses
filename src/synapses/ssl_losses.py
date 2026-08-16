"""Self-supervised losses for Phase 2.

Written by hand and unit-tested against independent reference computations, the same
treatment ``vae_loss`` got in Phase 1 -- these are the objects of study, not library calls.
Home for SimCLR's NT-Xent and SimSiam's symmetrised negative cosine; VICReg's loss joins
them in Step 5.
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
