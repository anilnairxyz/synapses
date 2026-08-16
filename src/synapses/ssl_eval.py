"""Evaluation harness for Phase 2 self-supervised representations.

Every method (SimCLR, SimSiam, VICReg, ...) is scored the same way: freeze the backbone,
embed the labelled train/test splits, and report a linear probe
(:func:`synapses.metrics.linear_probe`, unchanged from Phase 1 -- the cross-phase
comparison axis) plus a k-NN accuracy as a cheap non-parametric complement.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier

from synapses.metrics import active_units, linear_probe


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


@torch.no_grad()
def embed_dataset(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a frozen backbone over a loader. Returns ``(features, labels)`` on CPU."""
    model.eval()
    feats: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        feats.append(model(x).cpu())
        labels.append(y.cpu())
    return torch.cat(feats), torch.cat(labels)


def knn_accuracy(
    train_x: torch.Tensor | np.ndarray,
    train_y: torch.Tensor | np.ndarray,
    test_x: torch.Tensor | np.ndarray,
    test_y: torch.Tensor | np.ndarray,
    k: int = 20,
) -> float:
    """k-NN test accuracy in feature space -- a cheap, non-parametric probe complement."""
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(_to_numpy(train_x), _to_numpy(train_y))
    return float(clf.score(_to_numpy(test_x), _to_numpy(test_y)))


def active_units_normalized(features: torch.Tensor, threshold: float = 0.01) -> int:
    """``active_units`` on L2-normalized features.

    A cosine-similarity-based loss (NT-Xent, SimSiam's negative cosine, ...) is
    completely invariant to the overall *magnitude* of the backbone's output -- only
    direction matters. With nothing anchoring that magnitude, weight decay can (and, in
    practice, does) shrink it steadily over hundreds of epochs, which drives every
    dimension's *raw* variance toward zero even while the representation's *direction*
    -- what the loss and the linear probe actually care about -- stays perfectly
    informative. Raw :func:`synapses.metrics.active_units` would misread that scale
    drift as collapse. Normalizing each row to unit length first removes the confound:
    the count now reflects genuine per-dimension informational collapse, in the same
    space the loss itself operates in, not an arbitrary absolute-magnitude threshold
    crossing.
    """
    return active_units(F.normalize(features, dim=1), threshold=threshold)


def evaluate_representation(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device | str,
    k: int = 20,
    max_iter: int = 3000,
    active_units_threshold: float = 0.01,
) -> dict:
    """Freeze the backbone, embed both splits once, report the headline metrics.

    ``active_units`` (reused from Phase 1's VAE analysis, already generic over any
    representation) counts how many feature dimensions actually vary across the test
    set -- a collapsed representation (every input mapping to ~the same vector) crashes
    this number toward zero even when the training loss looks fine. Computed on
    L2-normalized features (see :func:`active_units_normalized`) so a whole-vector
    magnitude drift -- expected under weight decay for any magnitude-invariant SSL loss
    -- can't be mistaken for collapse. Computed from the test embeddings already
    gathered here, no extra forward pass. The absolute threshold was tuned for
    VAE-latent scale and isn't necessarily calibrated for a 512-d CNN feature space --
    every use of this number here is a paired comparison (healthy vs. collapsed run,
    same threshold, same feature scale), not a claim about an absolute "healthy" count.
    """
    train_x, train_y = embed_dataset(model, train_loader, device)
    test_x, test_y = embed_dataset(model, test_loader, device)
    return {
        "linear_probe": linear_probe(train_x, train_y, test_x, test_y, max_iter=max_iter),
        "knn": knn_accuracy(train_x, train_y, test_x, test_y, k=k),
        "active_units": active_units_normalized(test_x, threshold=active_units_threshold),
    }


def project_2d(
    features: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray | None = None,
    method: str = "tsne",
    n_subsample: int = 3000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Project high-dimensional features to 2D for :func:`synapses.viz.latent_scatter`.

    t-SNE (scikit-learn, no extra dependency) by default; pass ``method="umap"`` to use
    ``umap-learn`` (the ``viz`` extra) instead. Subsamples to ``n_subsample`` points --
    t-SNE is roughly O(N^2), and the full 50k-image CIFAR-10 train set would be slow.
    Returns ``(embedding, labels)`` with ``labels`` subsampled identically to ``embedding``.
    """
    x = _to_numpy(features)
    y = _to_numpy(labels) if labels is not None else None

    if x.shape[0] > n_subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(x.shape[0], size=n_subsample, replace=False)
        x = x[idx]
        y = y[idx] if y is not None else None

    if method == "tsne":
        # sklearn's default perplexity (30) must be < n_samples; cap it for small inputs
        # so project_2d doesn't blow up when called on a small eval subset.
        perplexity = min(30, max(1, x.shape[0] - 1))
        embedding = TSNE(
            n_components=2, random_state=seed, init="pca", perplexity=perplexity
        ).fit_transform(x)
    elif method == "umap":
        import umap  # the `viz` extra

        embedding = umap.UMAP(n_components=2, random_state=seed).fit_transform(x)
    else:
        raise ValueError(f"unknown method {method!r}; use 'tsne' or 'umap'")

    return embedding, y
