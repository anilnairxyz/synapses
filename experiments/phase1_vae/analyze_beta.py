"""Phase 1 · Step 6 — β sweep and the rate-distortion curve.

Loads the L=32 VAEs trained at β in {0, 0.1, 1, 5, 20} and plots the rate-distortion
frontier (R = E[KL], D = recon), active units vs β (posterior collapse), and
reconstructions (sharp at low β, blurry at high β). See
docs/epic1_phase1_plan_23-07-2026.md (Step 6).

Train the models first (L=32, β=1 already done in Step 5):
    uv run python experiments/phase1_vae/train.py --latent-dim 32 --beta 0
    uv run python experiments/phase1_vae/train.py --latent-dim 32 --beta 0.1
    uv run python experiments/phase1_vae/train.py --latent-dim 32 --beta 5
    uv run python experiments/phase1_vae/train.py --latent-dim 32 --beta 20
Then:
    uv run python experiments/phase1_vae/analyze_beta.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from synapses import get_device  # noqa: E402
from synapses.data import fixed_binarized_loader  # noqa: E402
from synapses.metrics import evaluate_vae  # noqa: E402
from synapses.vae import load_vae  # noqa: E402
from synapses.viz import image_grid, line_plot, rate_distortion_plot  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 Step 6 — β sweep / rate-distortion")
    p.add_argument("--betas", type=float, nargs="+", default=[0, 0.1, 1, 5, 20])
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epoch", type=int, default=30)
    p.add_argument("--n-recon", type=int, default=8)
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--out-root", type=str, default="outputs")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    print(f"device     : {device}")

    test_loader = fixed_binarized_loader(args.data_root, train=False, seed=1234)
    # A fixed set of images to reconstruct across every β (test loader is deterministic).
    fixed_x = next(iter(test_loader))[0][: args.n_recon]

    per_beta = {}
    recon_rows = [fixed_x]  # first row: the originals
    row_labels = ["orig"]
    for beta in args.betas:
        run_id = f"phase1_L{args.latent_dim}_beta{beta:g}_s{args.seed}"
        ckpt = Path(args.out_root) / run_id / "ckpt" / f"epoch_{args.epoch:04d}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Missing checkpoint for β={beta:g}: {ckpt}\n"
                f"Train it: uv run python experiments/phase1_vae/train.py "
                f"--latent-dim {args.latent_dim} --beta {beta:g}"
            )
        model, _ = load_vae(str(ckpt), device=device)
        m = evaluate_vae(model, test_loader, device)
        per_beta[beta] = {
            "recon": m["recon"], "kl": m["kl"], "elbo": m["elbo"],
            "active_units": m["active_units"],
        }
        recon_rows.append(model.reconstruct(fixed_x.to(device)).cpu())
        row_labels.append(f"β={beta:g}")
        print(f"β={beta:>5g}  D(recon) {m['recon']:7.2f}  R(kl) {m['kl']:6.2f}  "
              f"elbo {m['elbo']:8.2f}  active {m['active_units']:>3}/{args.latent_dim}")

    betas = list(per_beta.keys())
    out_dir = Path(args.out_root) / "phase1_beta_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Rate-distortion frontier.
    ax = rate_distortion_plot(
        [per_beta[b]["recon"] for b in betas],
        [per_beta[b]["kl"] for b in betas],
        labels=[f"β={b:g}" for b in betas],
        title=f"rate-distortion (L={args.latent_dim})",
    )
    ax.figure.savefig(out_dir / "rate_distortion.png", dpi=150, bbox_inches="tight")
    plt.close(ax.figure)

    # Active units + recon/KL vs β.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    line_plot(betas, {"active units": [per_beta[b]["active_units"] for b in betas]},
              xlabel="β", ylabel="active units", ax=axes[0], title="posterior collapse vs β")
    line_plot(betas,
              {"reconstruction": [per_beta[b]["recon"] for b in betas],
               "KL (rate)": [per_beta[b]["kl"] for b in betas]},
              xlabel="β", ylabel="nats / image", ax=axes[1], title="recon & rate vs β")
    fig.tight_layout()
    fig.savefig(out_dir / "beta_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Reconstructions: originals on top, then one row per β.
    grid = torch.cat(recon_rows, dim=0)
    ax = image_grid(grid, nrow=len(recon_rows), ncol=args.n_recon,
                    row_labels=row_labels, title=f"reconstructions across β (L={args.latent_dim})")
    ax.figure.savefig(out_dir / "reconstructions.png", dpi=150, bbox_inches="tight")
    plt.close(ax.figure)

    (out_dir / "step6_results.json").write_text(json.dumps({"per_beta": per_beta}, indent=2))
    print(f"saved      : {out_dir}/ (3 PNGs, step6_results.json)")


if __name__ == "__main__":
    main()
