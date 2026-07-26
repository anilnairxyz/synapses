"""Phase 1 · Step 5 — latent dimension sweep analysis.

Loads the trained VAEs at L in {2, 8, 32, 128}, and for each computes test ELBO (recon/KL
split), active units, per-dimension KL, and a linear-probe accuracy on mu — plus a
raw-pixel probe baseline. Produces the sweep summary and the KL-spectrum small multiples.
See docs/epic1_phase1_plan_23-07-2026.md (Step 5).

Train the models first (L=2 already done):
    uv run python experiments/epic1/phase1_vae/train.py --latent-dim 8
    uv run python experiments/epic1/phase1_vae/train.py --latent-dim 32
    uv run python experiments/epic1/phase1_vae/train.py --latent-dim 128
Then:
    uv run python experiments/epic1/phase1_vae/analyze_sweep.py
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
from synapses.extract import encode_moments  # noqa: E402
from synapses.metrics import evaluate_vae, linear_probe  # noqa: E402
from synapses.vae import load_vae  # noqa: E402
from synapses.viz import kl_spectrum_bar, line_plot  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 Step 5 — latent dimension sweep")
    p.add_argument("--latent-dims", type=int, nargs="+", default=[2, 8, 32, 128])
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epoch", type=int, default=30)
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--out-root", type=str, default="outputs")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    print(f"device     : {device}")

    test_loader = fixed_binarized_loader(args.data_root, train=False, seed=1234)
    train_loader = fixed_binarized_loader(args.data_root, train=True, seed=1234)

    # Raw-pixel probe baseline (independent of L, computed once).
    tr_x = torch.cat([x for x, _ in train_loader])
    tr_y = torch.cat([y for _, y in train_loader])
    te_x = torch.cat([x for x, _ in test_loader])
    te_y = torch.cat([y for _, y in test_loader])
    raw_probe = linear_probe(tr_x, tr_y, te_x, te_y)
    print(f"raw-pixel probe baseline: {raw_probe * 100:.1f}%")

    per_l = {}
    for L in args.latent_dims:
        run_id = f"phase1_L{L}_beta{args.beta:g}_s{args.seed}"
        ckpt = Path(args.out_root) / run_id / "ckpt" / f"epoch_{args.epoch:04d}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Missing checkpoint for L={L}: {ckpt}\n"
                f"Train it: uv run python experiments/epic1/phase1_vae/train.py --latent-dim {L}"
            )
        model, _ = load_vae(str(ckpt), device=device)
        m = evaluate_vae(model, test_loader, device)
        train_mu, _, train_labels = encode_moments(model, train_loader, device)
        probe = linear_probe(train_mu, train_labels, m["mu"], m["labels"])

        per_l[L] = {
            "recon": m["recon"], "kl": m["kl"], "elbo": m["elbo"],
            "active_units": m["active_units"], "kl_per_dim": m["kl_per_dim"],
            "probe_acc": probe,
        }
        print(f"L={L:>3}  elbo {m['elbo']:8.2f}  recon {m['recon']:7.2f}  kl {m['kl']:6.2f}  "
              f"active {m['active_units']:>3}/{L}  probe {probe * 100:5.1f}%")

    # --- Plots ---
    out_dir = Path(args.out_root) / "phase1_sweep"
    (out_dir).mkdir(parents=True, exist_ok=True)
    Ls = list(per_l.keys())

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    line_plot(
        Ls,
        {"reconstruction": [per_l[L]["recon"] for L in Ls],
         "KL (rate)": [per_l[L]["kl"] for L in Ls],
         "-ELBO (total)": [-per_l[L]["elbo"] for L in Ls]},
        xlabel="latent dim L", ylabel="nats / image", ax=axes[0], logx=True,
        title="rate-distortion vs L",
    )
    line_plot(
        Ls, {"active units": [per_l[L]["active_units"] for L in Ls]},
        xlabel="latent dim L", ylabel="active units", ax=axes[1], logx=True,
        title="active units vs L",
    )
    line_plot(
        Ls, {"VAE μ probe": [per_l[L]["probe_acc"] for L in Ls]},
        xlabel="latent dim L", ylabel="test accuracy", ax=axes[2], logx=True,
        baselines={"raw-pixel baseline": raw_probe}, title="linear probe vs L",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "sweep_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    ncol = 2
    nrow = (len(Ls) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4 * nrow), squeeze=False)
    for ax, L in zip(axes.ravel(), Ls):
        kl_spectrum_bar(per_l[L]["kl_per_dim"], ax=ax,
                        title=f"L={L} · {per_l[L]['active_units']} active")
    for ax in axes.ravel()[len(Ls):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "kl_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    results = {"raw_pixel_probe": raw_probe, "per_latent_dim": per_l}
    (out_dir / "step5_results.json").write_text(json.dumps(results, indent=2))
    print(f"saved      : {out_dir}/ (sweep_summary.png, kl_spectrum.png, step5_results.json)")


if __name__ == "__main__":
    main()
