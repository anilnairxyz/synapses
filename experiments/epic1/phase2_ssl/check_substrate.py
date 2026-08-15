"""Phase 2 Step 2 -- verify the shared SSL substrate end to end.

No training happens here: this builds a CifarResNet18 at random initialization and scores
it with the same evaluation harness every SSL method (Steps 3-5) will use. The resulting
probe/kNN numbers are the *floor* every trained method must clear -- a random CNN still
picks up crude colour/texture cues, so expect well above chance (10%) but far below a
trained representation (~80-90%+).

Usage:
    uv run python experiments/epic1/phase2_ssl/check_substrate.py
    uv run python experiments/epic1/phase2_ssl/check_substrate.py --subset 2000 --skip-scatter
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save a PNG, never open a window (works over SSH)
import matplotlib.pyplot as plt  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from synapses import get_device  # noqa: E402
from synapses.data import CIFAR10_CLASSES, cifar10_eval_loaders  # noqa: E402
from synapses.ssl_backbone import CifarResNet18  # noqa: E402
from synapses.ssl_eval import embed_dataset, evaluate_representation, project_2d  # noqa: E402
from synapses.viz import latent_scatter  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 Step 2 -- SSL substrate smoke test")
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--subset", type=int, default=5000, help="cap images embedded, for speed")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default="outputs/phase2_ssl/random_backbone_scatter.png")
    p.add_argument("--skip-scatter", action="store_true", help="skip the (slower) t-SNE plot")
    return p.parse_args()


def _capped(loader: DataLoader, n: int) -> DataLoader:
    """Cap a DataLoader's dataset to at most n examples, keeping its batch size."""
    ds = loader.dataset
    idx = list(range(min(n, len(ds))))
    return DataLoader(Subset(ds, idx), batch_size=loader.batch_size, shuffle=False)


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    print(f"device     : {device}")

    train_loader, test_loader = cifar10_eval_loaders(root=args.data_root, download=False)
    if args.subset:
        train_loader = _capped(train_loader, args.subset)
        test_loader = _capped(test_loader, args.subset)
    print(
        f"data       : {len(train_loader.dataset)} train / "
        f"{len(test_loader.dataset)} test (capped)"
    )

    model = CifarResNet18().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"backbone   : CifarResNet18 -- {n_params:,} params (random init, untrained)")

    metrics = evaluate_representation(model, train_loader, test_loader, device)
    print(f"linear probe : {metrics['linear_probe'] * 100:.1f}%")
    print(f"kNN (k=20)   : {metrics['knn'] * 100:.1f}%")
    print("(this is the floor every trained SSL method in Steps 3-5 must clear)")

    if not args.skip_scatter:
        feats, labels = embed_dataset(model, test_loader, device)
        emb, y = project_2d(feats, labels, method="tsne")
        ax = latent_scatter(
            emb, y, class_names=list(CIFAR10_CLASSES), title="random-init backbone (t-SNE)"
        )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(ax.figure)
        print(f"saved      : {out}")


if __name__ == "__main__":
    main()
