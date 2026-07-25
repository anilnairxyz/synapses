"""Representation-quality metrics for the Step 5 sweep.

`kl_per_dim` and `active_units` read the encoder's posterior parameters; `linear_probe`
measures how much class information a representation carries. The probe protocol here is
the one reused for SimCLR in Phase 2, so a VAE latent and an SSL embedding land on one
comparable axis.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def kl_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Per-dimension KL to N(0, I), averaged over the batch. Shape ``(L,)``.

    Same closed form as :func:`synapses.losses.vae_loss` (tested to match), kept here so
    the sweep can compute it directly from encoded moments.
    """
    return (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())).mean(dim=0)


def active_units(mu: torch.Tensor, threshold: float = 0.01) -> int:
    """Number of latent dims the model actually uses (Burda et al.).

    A dim is "active" if the variance of its posterior mean across the dataset exceeds
    ``threshold`` — a collapsed dim outputs a near-constant mu and carries no information.
    """
    return int((mu.var(dim=0) > threshold).sum().item())


def linear_probe(
    train_x: torch.Tensor | np.ndarray,
    train_y: torch.Tensor | np.ndarray,
    test_x: torch.Tensor | np.ndarray,
    test_y: torch.Tensor | np.ndarray,
    standardize: bool = True,
    max_iter: int = 1000,
) -> float:
    """Fit a logistic-regression probe on ``train_x`` and return test accuracy.

    Works on any representation — VAE ``mu`` per latent dim, or raw pixels for the
    baseline. Standardizes features by default so convergence doesn't depend on scale.
    """
    def _np(t):
        return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)

    clf = LogisticRegression(max_iter=max_iter)
    model = make_pipeline(StandardScaler(), clf) if standardize else clf
    model.fit(_np(train_x), _np(train_y))
    return float(model.score(_np(test_x), _np(test_y)))
