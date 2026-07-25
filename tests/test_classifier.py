"""CPU-only tests for the Step 4 pieces: SmallCNN, classify, class_means, and the two
new viz helpers. No GPU, no data download.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from dataclasses import dataclass

import torch

from synapses.classifier import SmallCNN, classify, load_classifier
from synapses.extract import class_means
from synapses.run import save_checkpoint
from synapses.viz import bar_by_class, image_grid

BATCH = 8


@dataclass
class _Cfg:
    n_classes: int = 10


def test_smallcnn_forward_shape():
    model = SmallCNN().eval()
    x = torch.rand(BATCH, 1, 28, 28)
    assert model(x).shape == (BATCH, 10)


def test_load_classifier_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters())
    ckpt = tmp_path / "best.pt"
    save_checkpoint(ckpt, model, opt, epoch=1, config=_Cfg())

    loaded, cfg = load_classifier(str(ckpt), device="cpu")
    assert cfg["n_classes"] == 10
    assert loaded.training is False
    for (_, v1), (_, v2) in zip(model.state_dict().items(), loaded.state_dict().items()):
        torch.testing.assert_close(v1, v2)


def test_classify_accepts_both_shapes_and_ranges():
    model = SmallCNN().eval()
    flat = torch.rand(BATCH, 784)
    imgs = torch.rand(BATCH, 1, 28, 28)

    for x in (flat, imgs):
        preds, conf = classify(model, x, device="cpu")
        assert preds.shape == (BATCH,)
        assert conf.shape == (BATCH,)
        assert preds.min() >= 0 and preds.max() <= 9
        assert conf.min() >= 0.0 and conf.max() <= 1.0


def test_class_means_correctness():
    # class 0: rows at (0,0) and (2,0) -> mean (1,0); class 1: single row (5,5).
    codes = torch.tensor([[0.0, 0.0], [2.0, 0.0], [5.0, 5.0]])
    labels = torch.tensor([0, 0, 1])
    means = class_means(codes, labels, n_classes=3)
    assert means.shape == (3, 2)
    torch.testing.assert_close(means[0], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(means[1], torch.tensor([5.0, 5.0]))
    torch.testing.assert_close(means[2], torch.tensor([0.0, 0.0]))  # empty -> zeros


def test_image_grid_builds():
    imgs = torch.rand(6, 784)
    ax = image_grid(imgs, nrow=2, ncol=3, row_labels=["a", "b"])
    assert len(ax.images) == 1
    assert ax.images[0].get_array().shape == (2 * 28, 3 * 28)


def test_bar_by_class_builds():
    ax = bar_by_class([0.9, 0.5, 0.7], ylabel="hit rate", as_percent=True)
    assert len(ax.patches) == 3
