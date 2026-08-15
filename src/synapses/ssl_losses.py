"""Self-supervised losses for Phase 2.

Written by hand and unit-tested against independent reference computations, the same
treatment ``vae_loss`` got in Phase 1 -- these are the objects of study, not library calls.
Home for SimCLR's NT-Xent now; SimSiam's and VICReg's losses join it in Steps 4-5.
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
