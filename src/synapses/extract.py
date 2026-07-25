"""Extract representations from a trained model over a dataset.

Right now this is VAE latent extraction for Phase 1 (Steps 2, 4, 5). The same concern
— run a frozen model over a loader and collect its internal representation — returns in
Phase 2 (SSL embeddings) and Phase 3 (residual-stream / MLP activations), so it lives in
its own module rather than in the model or the training script.
"""

from __future__ import annotations

import torch

from synapses.vae import MLPVAE


@torch.no_grad()
def encode_latents(
    model: MLPVAE,
    loader: torch.utils.data.DataLoader,
    device: torch.device | str,
    sample: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a whole dataset to its latent codes.

    Args:
        model: a trained :class:`MLPVAE` (already in eval mode).
        loader: yields ``(x, label)``; ``x`` may be flat ``(B, 784)`` or ``(B, 1, 28, 28)``.
        device: where to run the forward pass.
        sample: ``False`` returns the posterior means ``mu`` (less noise — the right
            choice for a scatter or a probe); ``True`` returns reparameterized samples
            ``z``, for when the stochasticity is the point.

    Returns:
        ``(codes, labels)`` on CPU, shapes ``(N, L)`` and ``(N,)``.
    """
    model.eval()
    codes: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True).view(x.size(0), -1)
        mu, logvar = model.encode(x)
        z = model.reparameterize(mu, logvar) if sample else mu
        codes.append(z.cpu())
        labels.append(y.cpu())

    return torch.cat(codes, dim=0), torch.cat(labels, dim=0)


def class_means(
    codes: torch.Tensor, labels: torch.Tensor, n_classes: int = 10
) -> torch.Tensor:
    """Per-class centroid of latent codes: ``μ̄_c`` for each ``c``.

    Args:
        codes: ``(N, L)`` latent codes.
        labels: ``(N,)`` integer class labels.

    Returns:
        ``(n_classes, L)``. A class with no samples yields a zero row.
    """
    means = torch.zeros(n_classes, codes.size(1), dtype=codes.dtype)
    for c in range(n_classes):
        m = labels == c
        if m.any():
            means[c] = codes[m].mean(dim=0)
    return means
