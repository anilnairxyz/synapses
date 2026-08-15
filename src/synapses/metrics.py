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

from synapses.losses import vae_loss


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


@torch.no_grad()
def evaluate_vae(
    model,
    loader: torch.utils.data.DataLoader,
    device: torch.device | str,
) -> dict:
    """One pass over a dataset -> the VAE's core metrics.

    Recon and KL are reported at β=1 regardless of the model's training β, so the numbers
    are comparable across an L-sweep or a β-sweep. Returns recon, kl, elbo (all
    nats/image), the sorted-later per-dim KL, active-unit count, and the collected ``mu``
    and ``labels`` (for a downstream probe).
    """
    model.eval()
    recon_sum = kl_sum = 0.0
    n = 0
    mus: list[torch.Tensor] = []
    logvars: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True).view(x.size(0), -1)
        logits, mu, logvar = model(x)
        out = vae_loss(logits, x, mu, logvar, beta=1.0)
        bs = x.size(0)
        recon_sum += out.recon.item() * bs
        kl_sum += out.kl.item() * bs
        n += bs
        mus.append(mu.cpu())
        logvars.append(logvar.cpu())
        labels.append(y.cpu())

    mu_all = torch.cat(mus)
    logvar_all = torch.cat(logvars)
    recon, kl = recon_sum / n, kl_sum / n
    return {
        "recon": recon,
        "kl": kl,
        "elbo": -(recon + kl),
        "kl_per_dim": kl_per_dim(mu_all, logvar_all).tolist(),
        "active_units": active_units(mu_all),
        "mu": mu_all,
        "labels": torch.cat(labels),
    }


def linear_probe(
    train_x: torch.Tensor | np.ndarray,
    train_y: torch.Tensor | np.ndarray,
    test_x: torch.Tensor | np.ndarray,
    test_y: torch.Tensor | np.ndarray,
    standardize: bool = True,
    max_iter: int = 3000,
) -> float:
    """Fit a logistic-regression probe on ``train_x`` and return test accuracy.

    Works on any representation — VAE ``mu`` per latent dim, or raw pixels for the
    baseline. Standardizes features by default so convergence doesn't depend on scale.

    ``max_iter`` defaults higher than sklearn's own default (100): Phase 1's low-
    dimensional VAE latents converge well under 1000 iterations, but a 512-d ResNet
    representation (Phase 2) needs more before lbfgs actually settles, not just stops.
    """
    def _np(t):
        return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)

    clf = LogisticRegression(max_iter=max_iter)
    model = make_pipeline(StandardScaler(), clf) if standardize else clf
    model.fit(_np(train_x), _np(train_y))
    return float(model.score(_np(test_x), _np(test_y)))
