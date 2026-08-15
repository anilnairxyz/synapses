"""Evaluation harness for Phase 2 self-supervised representations.

Every method (SimCLR, SimSiam, VICReg, ...) is scored the same way: freeze the backbone,
embed the labelled train/test splits, and report a linear probe
(:func:`synapses.metrics.linear_probe`, unchanged from Phase 1 -- the cross-phase
comparison axis) plus a k-NN accuracy as a cheap non-parametric complement.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier

from synapses.metrics import linear_probe


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


def evaluate_representation(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device | str,
    k: int = 20,
) -> dict:
    """Freeze the backbone, embed both splits once, report the two headline metrics."""
    train_x, train_y = embed_dataset(model, train_loader, device)
    test_x, test_y = embed_dataset(model, test_loader, device)
    return {
        "linear_probe": linear_probe(train_x, train_y, test_x, test_y),
        "knn": knn_accuracy(train_x, train_y, test_x, test_y, k=k),
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
