"""A small MLP VAE.

    Encoder                        Decoder
    784 -> 400 -> ReLU -> mu (L)   L -> 400 -> ReLU -> 784  (logits)
                       -> logvar (L)

Two deliberate choices:

*   The encoder predicts ``logvar``, never ``sigma``. Variance stays positive with no
    constraint or clamp, and the KL closed form is written directly in terms of it.
*   The decoder returns **logits**, not probabilities. Pairing them with
    ``binary_cross_entropy_with_logits`` fuses the sigmoid into the loss in a
    log-sum-exp-stable way. Use :meth:`MLPVAE.reconstruct` when you want images.
"""

from __future__ import annotations

import torch
from torch import nn

from synapses.run import load_checkpoint


class MLPVAE(nn.Module):
    """Gaussian-posterior, Bernoulli-likelihood VAE over flattened MNIST."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 400, latent_dim: int = 2) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``x`` of shape ``(B, 784)`` -> ``(mu, logvar)``, each ``(B, L)``."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """The reparameterization trick: ``z = mu + sigma * eps``, ``eps ~ N(0, I)``.

        Written out rather than sampled from ``torch.distributions`` so the gradient
        path through ``mu`` and ``sigma`` stays visible.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """``z`` of shape ``(B, L)`` -> Bernoulli **logits** of shape ``(B, 784)``."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> ``(logits, mu, logvar)``."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruction probabilities in ``[0, 1]``, for plotting from Step 2 on."""
        logits, _, _ = self.forward(x)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def sample(self, n: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """Decode ``n`` draws from the prior ``N(0, I)`` into probabilities."""
        z = torch.randn(n, self.latent_dim, device=device)
        return torch.sigmoid(self.decode(z))


def load_vae(ckpt_path: str, device: torch.device | str = "cpu") -> tuple[MLPVAE, dict]:
    """Rebuild a trained MLPVAE from a checkpoint written by ``save_checkpoint``.

    Reads the stored config to construct the model at the right shape, restores the
    weights, moves it to ``device``, and puts it in eval mode. Returns ``(model, config)``
    — every later Phase 1 step (manifold decode, class sampling, sweeps, feature
    emergence) reloads a trained model this way.
    """
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = MLPVAE(
        input_dim=cfg.get("input_dim", 784),
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg
