"""Plotting helpers for the exploration.

Currently a 2D latent scatter for Phase 1 Step 2; grows as later steps need it.

The 10-class palette below is a Paul Tol "bright"-derived set, tuned and validated with
the dataviz skill's ``validate_palette.js``. Ten categorical hues cannot all be made
colorblind-distinct — that is a hard constraint, not an oversight — so **identity is
carried by the digit label drawn at each cluster centroid**, and color only helps the
eye group points. The palette maximizes CVD separation subject to that (worst adjacent
pair ΔE 8.9, above the floor); the centroid labels are the authoritative channel.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

# digit -> hex. Fixed order, assigned by label, never cycled.
DIGIT_PALETTE = [
    "#4477AA",  # 0 blue
    "#EE6677",  # 1 red
    "#228833",  # 2 green
    "#CCBB44",  # 3 yellow
    "#66CCEE",  # 4 cyan
    "#AA3377",  # 5 purple
    "#774411",  # 6 brown
    "#EE8866",  # 7 orange
    "#77AADD",  # 8 light blue
    "#000000",  # 9 black
]


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def latent_scatter(
    mu: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    ax: plt.Axes | None = None,
    class_names: list[str] | None = None,
    prior_sigmas: tuple[float, ...] = (1.0, 2.0),
    annotate_centroids: bool = True,
    title: str | None = None,
    point_size: float = 4.0,
    alpha: float = 0.4,
) -> plt.Axes:
    """Scatter 2D latent means colored by class, with the N(0,I) prior overlaid.

    Args:
        mu: ``(N, 2)`` latent means.
        labels: ``(N,)`` integer class labels in ``[0, 9]``.
        prior_sigmas: radii (in std units) at which to draw the isotropic prior circles.
        annotate_centroids: draw each class's digit at its centroid (the identity channel).

    Returns:
        The matplotlib ``Axes`` the scatter was drawn on.
    """
    mu = _to_numpy(mu)
    labels = _to_numpy(labels).astype(int)
    if mu.ndim != 2 or mu.shape[1] != 2:
        raise ValueError(
            f"latent_scatter needs 2D latents, got shape {mu.shape}. "
            "For L>2, project first (t-SNE/UMAP) — that is Step 5's concern."
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    classes = np.unique(labels)
    for c in classes:
        m = labels == c
        name = class_names[c] if class_names is not None else str(c)
        ax.scatter(
            mu[m, 0], mu[m, 1],
            s=point_size, alpha=alpha, linewidths=0,
            color=DIGIT_PALETTE[c % len(DIGIT_PALETTE)], label=name,
        )

    # Prior overlay: N(0, I) is isotropic, so k-sigma contours are circles of radius k.
    for k in prior_sigmas:
        ax.add_patch(
            plt.Circle((0, 0), k, fill=False, linestyle="--", linewidth=1.0,
                       edgecolor="0.5", alpha=0.8, zorder=5)
        )
        ax.annotate(f"{k:g}σ", xy=(0, k), color="0.4", fontsize=8,
                    ha="center", va="bottom", zorder=6)

    # Secondary encoding: the digit at each centroid is the authoritative identity.
    if annotate_centroids:
        for c in classes:
            cx, cy = mu[labels == c].mean(axis=0)
            name = class_names[c] if class_names is not None else str(c)
            ax.text(
                cx, cy, name, fontsize=13, fontweight="bold",
                ha="center", va="center", color="black", zorder=10,
                path_effects=_halo(),
            )

    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linewidth=0.3, alpha=0.3)
    if title:
        ax.set_title(title)
    ax.legend(title="digit", markerscale=2, loc="best", framealpha=0.9, fontsize=8)
    return ax


def _halo():
    """A white outline so centroid labels stay legible over any cluster color."""
    import matplotlib.patheffects as pe

    return [pe.withStroke(linewidth=2.5, foreground="white")]
