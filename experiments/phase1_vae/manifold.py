"""Phase 1 · Step 3 — decode the manifold.

Load a trained 2D VAE, lay an inverse-CDF-spaced grid over the latent space, decode each
point, and tile the results into one image. You see digits morph smoothly into their
neighbors — the "manifold." See ``docs/epic1_phase1_plan_23-07-2026.md`` (Step 3).

Usage:

    uv run python experiments/phase1_vae/manifold.py            # defaults to the Step 1 run
    uv run python experiments/phase1_vae/manifold.py --n 30     # finer grid
    uv run python experiments/phase1_vae/manifold.py --ckpt path/to/epoch_0030.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save a PNG, never open a window (works over SSH)
import matplotlib.pyplot as plt  # noqa: E402

from synapses import get_device  # noqa: E402
from synapses.vae import load_vae  # noqa: E402
from synapses.viz import manifold_grid  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 Step 3 — decode the latent manifold")
    p.add_argument("--run-id", type=str, default="phase1_L2_beta1_s0")
    p.add_argument("--epoch", type=int, default=30, help="checkpoint epoch to load")
    p.add_argument("--ckpt", type=str, default=None,
                   help="explicit checkpoint path (overrides run-id/epoch)")
    p.add_argument("--n", type=int, default=20, help="grid resolution (n x n decodes)")
    p.add_argument("--out-root", type=str, default="outputs")
    p.add_argument("--out", type=str, default=None,
                   help="output PNG path (default: <run>/plots/manifold_grid.png)")
    p.add_argument("--device", type=str, default=None)
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

    if cfg["latent_dim"] != 2:
        raise SystemExit(
            f"This run has latent_dim={cfg['latent_dim']}, but the manifold grid needs a 2D "
            "latent. Train an L=2 run, or wait for Step 5 (which projects L>2)."
        )

    print(f"decoding   : {args.n} x {args.n} = {args.n ** 2} latent points")
    ax = manifold_grid(
        model, device, n=args.n,
        title=f"{args.run_id} · epoch {args.epoch} · latent manifold",
    )

    out = Path(args.out) if args.out else run_dir / "plots" / "manifold_grid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
    print(f"saved      : {out}")


if __name__ == "__main__":
    main()
