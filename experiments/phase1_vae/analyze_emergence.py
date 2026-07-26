"""Phase 1 · Step 7 — feature emergence across training.

Loads a run's log-spaced checkpoint series and tracks how the representation forms over
training: the latent scatter condensing from noise into clusters (L=2), and metric curves
(active units, rate, probe accuracy) vs epoch. Warm-up for the Phase 3 SAE feature-
emergence study, which uses the same checkpoint series. See
docs/epic1_phase1_plan_23-07-2026.md (Step 7).

Usage:
    uv run python experiments/phase1_vae/analyze_emergence.py
    uv run python experiments/phase1_vae/analyze_emergence.py --run-id phase1_L32_beta1_s0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

from synapses import get_device  # noqa: E402
from synapses.data import fixed_binarized_loader  # noqa: E402
from synapses.extract import encode_latents, encode_moments  # noqa: E402
from synapses.metrics import evaluate_vae, linear_probe  # noqa: E402
from synapses.run import list_checkpoints  # noqa: E402
from synapses.vae import load_vae  # noqa: E402
from synapses.viz import latent_scatter, line_plot  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 Step 7 — feature emergence")
    p.add_argument("--run-id", type=str, default="phase1_L2_beta1_s0")
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--out-root", type=str, default="outputs")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    run_dir = Path(args.out_root) / args.run_id

    checkpoints = list_checkpoints(run_dir)
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints under {run_dir / 'ckpt'}. Train first: "
            f"experiments/phase1_vae/train.py (produces the log-spaced series)."
        )
    print(f"device     : {device}")
    print(f"run        : {args.run_id}  ({len(checkpoints)} checkpoints)")

    test_loader = fixed_binarized_loader(args.data_root, train=False, seed=1234)
    train_loader = fixed_binarized_loader(args.data_root, train=True, seed=1234)

    epochs, active, rate, probe = [], [], [], []
    scatters = []  # (epoch, mu, labels) for the L=2 small-multiples
    latent_dim = None
    for epoch, ckpt in checkpoints:
        model, cfg = load_vae(str(ckpt), device=device)
        latent_dim = cfg["latent_dim"]
        m = evaluate_vae(model, test_loader, device)
        train_mu, _, train_labels = encode_moments(model, train_loader, device)
        acc = linear_probe(train_mu, train_labels, m["mu"], m["labels"])

        epochs.append(epoch)
        active.append(m["active_units"])
        rate.append(m["kl"])
        probe.append(acc)
        if latent_dim == 2:
            mu, labels = encode_latents(model, test_loader, device)
            scatters.append((epoch, mu, labels))
        print(f"epoch {epoch:>3}  active {m['active_units']:>3}/{latent_dim}  "
              f"rate {m['kl']:6.2f}  probe {acc * 100:5.1f}%")

    plots = run_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    # Metric curves vs epoch.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    line_plot(epochs, {"active units": active}, xlabel="epoch", ylabel="active units",
              ax=axes[0], logx=True, title="active units vs epoch")
    line_plot(epochs, {"rate (KL)": rate}, xlabel="epoch", ylabel="nats / image",
              ax=axes[1], logx=True, title="rate vs epoch")
    line_plot(epochs, {"probe acc": probe}, xlabel="epoch", ylabel="test accuracy",
              ax=axes[2], logx=True, title="probe accuracy vs epoch")
    fig.tight_layout()
    fig.savefig(plots / "emergence_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Latent scatter small-multiples (L=2 only): clusters condensing over training.
    if scatters:
        ncol = 3
        nrow = (len(scatters) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5 * nrow), squeeze=False)
        for ax, (epoch, mu, labels) in zip(axes.ravel(), scatters):
            latent_scatter(mu, labels, ax=ax, annotate_centroids=True,
                           title=f"epoch {epoch}")
            ax.legend().remove()  # one shared story; keep the small multiples clean
        for ax in axes.ravel()[len(scatters):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(plots / "emergence_scatter.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved      : {plots}/emergence_scatter.png")
    else:
        print(f"  (latent_dim={latent_dim} > 2: skipping the scatter small-multiples)")

    results = {
        "run_id": args.run_id,
        "epochs": epochs,
        "active_units": active,
        "rate": rate,
        "probe_acc": probe,
    }
    (run_dir / "emergence_results.json").write_text(json.dumps(results, indent=2))
    print(f"saved      : {plots}/emergence_curves.png and {run_dir / 'emergence_results.json'}")


if __name__ == "__main__":
    main()
