"""Phase 1 · Step 2 — see the latent space.

Load a trained VAE checkpoint, encode the MNIST test set to its latent means, and plot a
2D scatter colored by digit with the N(0,I) prior overlaid. See
``docs/epic1_phase1_plan_23-07-2026.md`` (Step 2) for what to read off it.

Usage (on the PC, RTX 4090):

    uv run python experiments/phase1_vae/visualize.py                 # defaults to the Step 1 run
    uv run python experiments/phase1_vae/visualize.py --epoch 8       # a mid-training checkpoint
    uv run python experiments/phase1_vae/visualize.py --ckpt path/to/epoch_0030.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save a PNG, never open a window (works over SSH)
import matplotlib.pyplot as plt  # noqa: E402

from synapses import get_device  # noqa: E402
from synapses.data import mnist_loaders  # noqa: E402
from synapses.extract import encode_latents  # noqa: E402
from synapses.vae import load_vae  # noqa: E402
from synapses.viz import latent_scatter  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 Step 2 — latent space scatter")
    p.add_argument("--run-id", type=str, default="phase1_L2_beta1_s0")
    p.add_argument("--epoch", type=int, default=30, help="checkpoint epoch to load")
    p.add_argument("--ckpt", type=str, default=None,
                   help="explicit checkpoint path (overrides run-id/epoch)")
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--out-root", type=str, default="outputs")
    p.add_argument("--out", type=str, default=None,
                   help="output PNG path (default: <run>/plots/latent_scatter.png)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=256)
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
            f"No checkpoint at {ckpt_path}. Train first (experiments/phase1_vae/train.py) "
            "or point --ckpt at an existing file."
        )

    print(f"device     : {device}")
    print(f"checkpoint : {ckpt_path}")
    model, cfg = load_vae(str(ckpt_path), device=device)

    latent_dim = cfg["latent_dim"]
    if latent_dim != 2:
        raise SystemExit(
            f"This run has latent_dim={latent_dim}, but the Step 2 scatter needs a 2D latent. "
            "Train an L=2 run, or wait for Step 5 (which projects L>2 with t-SNE/UMAP)."
        )

    _, test_loader = mnist_loaders(
        root=args.data_root, batch_size=args.batch_size, download=False
    )
    mu, labels = encode_latents(model, test_loader, device)
    print(f"encoded    : {mu.shape[0]} test images -> latent {tuple(mu.shape)}")

    # Cheap numeric read on the prior hole: aggregate-posterior spread vs the prior's 1.0.
    std = mu.std(dim=0)
    print(f"agg. posterior std per dim: [{std[0]:.2f}, {std[1]:.2f}]  (prior = 1.00)")

    ax = latent_scatter(mu, labels, title=f"{args.run_id} · epoch {args.epoch} · latent means")

    out = Path(args.out) if args.out else run_dir / "plots" / "latent_scatter.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
    print(f"saved      : {out}")


if __name__ == "__main__":
    main()
