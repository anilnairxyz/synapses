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


# Single accent color for one-measure bar charts (not the categorical palette).
ACCENT = "#4477AA"


def line_plot(
    x: list | np.ndarray,
    series: dict[str, list | np.ndarray],
    xlabel: str,
    ylabel: str,
    ax: plt.Axes | None = None,
    title: str | None = None,
    logx: bool = False,
    baselines: dict[str, float] | None = None,
) -> plt.Axes:
    """Multi-series line+marker plot on one axis.

    ``series`` maps name -> y-values (all against the shared ``x``). ``baselines`` maps
    name -> a constant, drawn as a horizontal dashed reference (e.g. the raw-pixel probe).
    A legend appears whenever there is more than one line in total.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    for i, (name, y) in enumerate(series.items()):
        ax.plot(x, _to_numpy(y), marker="o", linewidth=2,
                color=DIGIT_PALETTE[i % len(DIGIT_PALETTE)], label=name)
    if baselines:
        for name, val in baselines.items():
            ax.axhline(val, linestyle="--", linewidth=1.2, color="0.5", label=name)

    if logx:
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(v) for v in x])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linewidth=0.3, alpha=0.3)
    if len(series) + (len(baselines) if baselines else 0) > 1:
        ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    return ax


def rate_distortion_plot(
    distortion: list | np.ndarray,
    rate: list | np.ndarray,
    labels: list,
    ax: plt.Axes | None = None,
    title: str | None = None,
    iso_elbo: bool = True,
) -> plt.Axes:
    """The rate-distortion frontier: distortion (x) vs rate (y), one point per model.

    ``labels`` annotates each point (e.g. its β). With ``iso_elbo``, faint diagonals
    ``D + R = const`` mark equal-ELBO contours — lower-left is better ELBO, and you can
    read which β is ELBO-optimal off where the frontier touches the innermost diagonal.
    """
    d = _to_numpy(distortion).astype(float)
    r = _to_numpy(rate).astype(float)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    if iso_elbo:
        totals = d + r
        lo, hi = totals.min(), totals.max()
        xs = np.array([d.min(), d.max()])
        for c in np.linspace(lo, hi, 5):
            ax.plot(xs, c - xs, linestyle="--", linewidth=0.8, color="0.8", zorder=1)

    ax.plot(d, r, "-", color="0.6", linewidth=1.0, zorder=2)
    ax.scatter(d, r, s=40, color=ACCENT, zorder=3)
    for xi, yi, lab in zip(d, r, labels):
        ax.annotate(f" {lab}", (xi, yi), fontsize=9, color="0.2",
                    ha="left", va="bottom")

    ax.set_xlabel("distortion  D = −E[log p(x|z)]  (nats)")
    ax.set_ylabel("rate  R = E[KL]  (nats)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title)
    return ax


def kl_spectrum_bar(
    values: torch.Tensor | np.ndarray,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Per-dimension KL, sorted descending — the rate each latent dim carries.

    At large L most bars sit at ~zero: the model uses a handful of dims and discards the
    rest. This is the information bottleneck made visible.
    """
    vals = np.sort(_to_numpy(values).astype(float))[::-1]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.bar(np.arange(len(vals)), vals, color=ACCENT, width=1.0)
    ax.set_xlabel("latent dim (sorted)")
    ax.set_ylabel("KL (nats)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title)
    return ax


def image_grid(
    images: torch.Tensor | np.ndarray,
    nrow: int,
    ncol: int,
    row_labels: list[str] | None = None,
    image_shape: tuple[int, int] = (28, 28),
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Tile ``nrow*ncol`` images into one grayscale montage.

    ``images`` is ``(nrow*ncol, H*W)`` or ``(nrow*ncol, H, W)`` in row-major order
    (row 0 first). ``row_labels`` (length ``nrow``) annotates each row on the left —
    e.g. the target digit for Step 4 sample grids.
    """
    imgs = _to_numpy(images)
    h, w = image_shape
    imgs = imgs.reshape(nrow, ncol, h, w)
    tile = imgs.transpose(0, 2, 1, 3).reshape(nrow * h, ncol * w)

    if ax is None:
        _, ax = plt.subplots(figsize=(ncol * 0.6, nrow * 0.6))
    ax.imshow(tile, cmap="gray_r", origin="upper")
    ax.set_xticks([])
    if row_labels is not None:
        ax.set_yticks([h * (i + 0.5) for i in range(nrow)])
        ax.set_yticklabels(row_labels)
    else:
        ax.set_yticks([])
    if title:
        ax.set_title(title)
    return ax


def bar_by_class(
    values: torch.Tensor | np.ndarray,
    ylabel: str,
    ax: plt.Axes | None = None,
    title: str | None = None,
    as_percent: bool = False,
) -> plt.Axes:
    """Single-series bar over classes 0..N-1 (magnitude by category).

    One accent color (not the categorical palette — this encodes magnitude, not
    identity), values printed above each bar, recessive axes.
    """
    vals = _to_numpy(values).astype(float)
    n = len(vals)
    x = np.arange(n)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, vals, color=ACCENT, width=0.8)
    for xi, v in zip(x, vals):
        label = f"{v * 100:.0f}%" if as_percent else f"{v:g}"
        ax.text(xi, v, label, ha="center", va="bottom", fontsize=8, color="0.2")

    ax.set_xticks(x)
    ax.set_xlabel("digit")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(y=0.15)
    if title:
        ax.set_title(title)
    return ax


def latent_grid_coords(
    n: int = 20, prob_range: tuple[float, float] = (0.005, 0.995)
) -> torch.Tensor:
    """1D grid of ``n`` coordinates spaced by the standard-normal inverse CDF.

    Uniform spacing in ``z`` (e.g. ``linspace(-3, 3)``) oversamples the low-probability
    tails, where the model saw almost no data — so a grid built that way decodes to
    garbage in the corners. Spacing by the prior's inverse CDF instead gives each cell
    equal prior probability: points bunch near 0 and spread out in the tails, matching
    where the digits actually live. Returned ascending.
    """
    probs = torch.linspace(prob_range[0], prob_range[1], n)
    return torch.distributions.Normal(0.0, 1.0).icdf(probs)


def manifold_grid(
    model: MLPVAE,
    device: torch.device | str,
    n: int = 20,
    prob_range: tuple[float, float] = (0.005, 0.995),
    image_shape: tuple[int, int] = (28, 28),
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Decode an ``n x n`` inverse-CDF grid of latent points into one tiled image.

    Shows what the decoder generates at every point in the 2D latent space — the
    "manifold." Complements :func:`latent_scatter` (which shows where real data landed):
    same latent coordinates, same orientation, so the two read together.
    """
    if model.latent_dim != 2:
        raise ValueError(
            f"manifold_grid needs a 2D latent, got latent_dim={model.latent_dim}. "
            "Grid decoding only makes sense in 2D."
        )

    coords = latent_grid_coords(n, prob_range)  # ascending, shape (n,)
    # z1 across columns (ascending, left->right), z2 up rows. Put the highest z2 in row 0
    # so that with imshow's default origin="upper" it renders at the top — this keeps each
    # digit upright (origin="lower" would flip every glyph vertically) while still giving
    # a z2-increases-upward axis.
    z2, z1 = torch.meshgrid(coords.flip(0), coords, indexing="ij")
    z = torch.stack([z1.reshape(-1), z2.reshape(-1)], dim=1).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model.decode(z)).cpu()  # (n*n, 784)

    h, w = image_shape
    # Reshape to (n, n, h, w) then tile into (n*h, n*w).
    tile = probs.view(n, n, h, w).permute(0, 2, 1, 3).reshape(n * h, n * w).numpy()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    lo, hi = coords[0].item(), coords[-1].item()
    # origin="upper" (default): row 0 -> top. Row 0 holds the highest z2 (see meshgrid
    # above), and extent's top edge is `hi`, so z2 still increases upward and glyphs stay
    # upright. imshow places the highest-z2 row at the top edge, matching the y axis.
    ax.imshow(tile, cmap="gray_r", origin="upper", extent=[lo, hi, lo, hi])
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    if title:
        ax.set_title(title)
    return ax
