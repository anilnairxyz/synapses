"""CPU-only tests for the SSL evaluation harness."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from synapses.ssl_eval import embed_dataset, evaluate_representation, knn_accuracy, project_2d


class _Identity(torch.nn.Module):
    """Stands in for a trained backbone: features are handed straight through."""

    def forward(self, x):
        return x


def test_knn_accuracy_separable_blobs():
    torch.manual_seed(0)
    a = torch.randn(100, 4) + 5.0
    b = torch.randn(100, 4) - 5.0
    x = torch.cat([a, b])
    y = torch.cat([torch.zeros(100), torch.ones(100)]).long()
    acc = knn_accuracy(x[::2], y[::2], x[1::2], y[1::2], k=5)
    assert acc > 0.98


def test_embed_dataset_shapes():
    x = torch.randn(40, 8)
    y = torch.randint(0, 10, (40,))
    loader = DataLoader(TensorDataset(x, y), batch_size=16)
    feats, labels = embed_dataset(_Identity(), loader, device="cpu")
    assert feats.shape == (40, 8)
    assert labels.shape == (40,)


def test_evaluate_representation_keys_and_quality():
    torch.manual_seed(0)
    a = torch.randn(60, 4) + 5.0
    b = torch.randn(60, 4) - 5.0
    x = torch.cat([a, b])
    y = torch.cat([torch.zeros(60), torch.ones(60)]).long()
    train_loader = DataLoader(TensorDataset(x[::2], y[::2]), batch_size=16)
    test_loader = DataLoader(TensorDataset(x[1::2], y[1::2]), batch_size=16)

    out = evaluate_representation(_Identity(), train_loader, test_loader, device="cpu")
    assert set(out) == {"linear_probe", "knn"}
    assert out["linear_probe"] > 0.9
    assert out["knn"] > 0.9


def test_project_2d_shape_and_subsample():
    torch.manual_seed(0)
    x = torch.randn(200, 16)
    y = torch.randint(0, 5, (200,))
    emb, labels = project_2d(x, y, method="tsne", n_subsample=50, seed=0)
    assert emb.shape == (50, 2)
    assert labels.shape == (50,)


def test_project_2d_no_subsample_when_smaller_than_cap():
    torch.manual_seed(0)
    x = torch.randn(30, 8)
    emb, labels = project_2d(x, None, method="tsne", n_subsample=3000)
    assert emb.shape == (30, 2)
    assert labels is None
