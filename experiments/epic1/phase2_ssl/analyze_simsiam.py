"""Phase 2 -- the per-class picture for a trained SimSiam representation.

Loads a trained checkpoint, embeds the CIFAR-10 test set through the frozen backbone, and
produces: (1) a t-SNE scatter coloured by true class -- directly comparable to
analyze_simclr.py's scatter, since both start from the same random-init floor
(check_substrate.py) and the same evaluation protocol; and (2) a per-class linear-probe
accuracy breakdown plus each class's top confusion, testing whether a completely
different anti-collapse mechanism (no negatives at all) rediscovers the same structure
SimCLR found -- the vehicle/animal split, and cat/dog as the hard pair -- or something else.

Usage:
    uv run python experiments/epic1/phase2_ssl/analyze_simsiam.py
    uv run python experiments/epic1/phase2_ssl/analyze_simsiam.py --run-id ... --epoch 128
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save a PNG, never open a window (works over SSH)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from synapses import get_device  # noqa: E402
from synapses.data import CIFAR10_CLASSES, cifar10_eval_loaders  # noqa: E402
from synapses.ssl_eval import embed_dataset, participation_ratio, project_2d  # noqa: E402
from synapses.viz import bar_by_class, latent_scatter  # noqa: E402
from train_simsiam import load_simsiam  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 -- SimSiam per-class analysis")
    p.add_argument("--run-id", type=str, default="phase2_simsiam_bs256_s0")
    p.add_argument("--epoch", type=int, default=200)
    p.add_argument("--ckpt", type=str, default=None, help="explicit checkpoint path")
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--out-root", type=str, default="outputs")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--n-scatter", type=int, default=5000, help="t-SNE subsample size")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    if args.ckpt is not None:
        ckpt_path = Path(args.ckpt)
        run_dir = ckpt_path.parent.parent
    else:
        run_dir = Path(args.out_root) / args.run_id
        ckpt_path = run_dir / "ckpt" / f"epoch_{args.epoch:04d}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Train first (train_simsiam.py) "
            "or pass an explicit --ckpt."
        )

    print(f"device     : {device}")
    print(f"checkpoint : {ckpt_path}")
    model, cfg = load_simsiam(str(ckpt_path), device=device)
    backbone = model.backbone  # only the representation is ever evaluated

    train_loader, test_loader = cifar10_eval_loaders(root=args.data_root, download=False)
    train_x, train_y = embed_dataset(backbone, train_loader, device)
    test_x, test_y = embed_dataset(backbone, test_loader, device)
    print(f"embedded   : {train_x.shape[0]} train / {test_x.shape[0]} test (512-d each)")

    # Effective rank of the feature covariance -- scale- and dimension-free, so it can't
    # be fooled by the magnitude drift a cosine-based loss + weight decay produces over
    # training. Recomputed here from the saved checkpoint -- no retraining needed.
    rank = participation_ratio(test_x)
    print(f"participation ratio (effective rank): {rank:.1f}/{test_x.shape[1]}")

    # Fit the same way synapses.metrics.linear_probe does, but keep the fitted classifier
    # so we can pull per-sample predictions (linear_probe only returns aggregate accuracy).
    # This overall number should closely match what training logged at this epoch.
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    clf.fit(train_x.numpy(), train_y.numpy())
    preds = clf.predict(test_x.numpy())
    overall_acc = float((preds == test_y.numpy()).mean())
    print(f"linear probe (overall): {overall_acc * 100:.1f}%")

    classes = list(CIFAR10_CLASSES)
    n_classes = len(classes)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(test_y.numpy(), preds):
        confusion[true, pred] += 1

    per_class_acc = confusion.diagonal() / confusion.sum(axis=1)
    print("\nper-class accuracy:")
    for i, c in enumerate(classes):
        row = confusion[i].copy()
        row[i] = 0  # exclude the diagonal (correct predictions) from "most confused with"
        top = int(row.argmax())
        top_frac = row[top] / confusion[i].sum()
        print(f"  {c:<10} {per_class_acc[i] * 100:5.1f}%   "
              f"most confused with: {classes[top]:<10} ({top_frac * 100:4.1f}%)")

    plots = run_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    ax = bar_by_class(
        per_class_acc, ylabel="linear-probe accuracy", as_percent=True,
        xlabel="class", xticklabels=classes,
        title=f"{args.run_id} · per-class probe accuracy (epoch {args.epoch})",
    )
    ax.figure.savefig(plots / "simsiam_per_class_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(ax.figure)

    emb, y = project_2d(test_x, test_y, method="tsne", n_subsample=args.n_scatter)
    ax = latent_scatter(
        emb, y, class_names=classes,
        title=f"{args.run_id} · trained backbone (t-SNE, epoch {args.epoch})",
    )
    ax.figure.savefig(plots / "simsiam_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(ax.figure)

    print(f"\nsaved      : {plots}/simsiam_per_class_accuracy.png, simsiam_scatter.png")


if __name__ == "__main__":
    main()
