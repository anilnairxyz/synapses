"""The (beta-)VAE objective, assembled by hand.

``torch.distributions.kl_divergence`` would hand us the KL term for free. We write the
closed form out instead: it is the formula that reappears in every variational method
downstream, and ``tests/test_losses.py`` checks ours against the library version so the
derivation is verified numerically rather than by squinting.

**Normalization is the thing to get right.** The reconstruction term is *summed over the
784 pixels* and only then *averaged over the batch*. Using ``reduction="mean"`` would
silently divide it by 784, leaving the KL term ~3 orders of magnitude too strong: the
model posterior-collapses, and every beta measurement in Step 6 becomes meaningless
because beta is already effectively ~784.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class VAELoss:
    """All terms in nats per image, averaged over the batch.

    ``loss`` is the training objective (``recon + beta * kl``); the true ELBO is
    ``-(recon + kl)`` regardless of beta, which is what gets reported.
    """

    loss: torch.Tensor
    recon: torch.Tensor
    kl: torch.Tensor
    kl_per_dim: torch.Tensor

    @property
    def elbo(self) -> torch.Tensor:
        return -(self.recon + self.kl)


def vae_loss(
    logits: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> VAELoss:
    """Bernoulli reconstruction + Gaussian KL to a standard normal prior.

    Args:
        logits: decoder output, ``(B, D)``, pre-sigmoid.
        x: binarized target, ``(B, D)``, values in ``{0, 1}``.
        mu, logvar: encoder output, ``(B, L)``.
        beta: KL weight. 1.0 is the plain VAE; Step 6 sweeps it.

    Returns:
        A :class:`VAELoss`. ``kl_per_dim`` has shape ``(L,)`` — the rate carried by each
        latent dimension, and the most informative diagnostic in this phase.
    """
    # -log p(x|z), summed over pixels, averaged over the batch.
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=1).mean()

    # KL( N(mu, sigma^2) || N(0, I) ) = -0.5 * sum_j (1 + logvar_j - mu_j^2 - exp(logvar_j))
    # Kept per-dimension (mean over batch) before summing, so we can see the rate
    # allocation across dimensions in Step 5.
    kl_per_dim = (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())).mean(dim=0)
    kl = kl_per_dim.sum()

    return VAELoss(loss=recon + beta * kl, recon=recon, kl=kl, kl_per_dim=kl_per_dim)
