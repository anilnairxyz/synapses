"""Phase 1 · Step 4 — sample near a class, and from the prior.

For each digit c: sample z ~ N(mu_c, spread*I) near its latent cluster centroid, decode,
and measure the hit rate (fraction the SmallCNN judge calls c). Then sample z ~ N(0,I)
and chart the digit histogram — the generation imbalance. See
docs/epic1_phase1_plan_23-07-2026.md (Step 4).

Usage (needs a trained VAE and a trained classifier):
    uv run python experiments/epic1/phase1_vae/train_classifier.py
    uv run python experiments/epic1/phase1_vae/sample.py
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
from synapses.classifier import classify, load_classifier  # noqa: E402
from synapses.data import mnist_loaders  # noqa: E402
from synapses.extract import class_means, encode_latents  # noqa: E402
from synapses.vae import load_vae  # noqa: E402
from synapses.viz import bar_by_class, image_grid  # noqa: E402

N_CLASSES = 10
GARBAGE_CONF = 0.9  # max-softmax below this counts as low-confidence ("ambiguous")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 Step 4 — class & prior sampling")
    p.add_argument("--run-id", type=str, default="phase1_L2_beta1_s0")
    p.add_argument("--epoch", type=int, default=30)
    p.add_argument("--ckpt", type=str, default=None, help="explicit VAE checkpoint path")
    p.add_argument("--classifier", type=str, default="outputs/mnist_cnn/ckpt/best.pt")
    p.add_argument("--n-per-class", type=int, default=500)
    p.add_argument("--spread", type=float, default=0.1)
    p.add_argument("--n-prior", type=int, default=5000)
    p.add_argument("--n-show", type=int, default=12, help="samples per row in the sample grid")
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--out-root", type=str, default="outputs")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def decode(model, z: torch.Tensor, device) -> torch.Tensor:
    """Latent codes -> image probabilities (N, 784)."""
    return torch.sigmoid(model.decode(z.to(device))).cpu()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    torch.manual_seed(args.seed)

    if args.ckpt is not None:
        ckpt_path = Path(args.ckpt)
        run_dir = ckpt_path.parent.parent
    else:
        run_dir = Path(args.out_root) / args.run_id
        ckpt_path = run_dir / "ckpt" / f"epoch_{args.epoch:04d}.pt"

    for path, what in [(ckpt_path, "VAE checkpoint"), (Path(args.classifier), "classifier")]:
        if not path.exists():
            raise FileNotFoundError(
                f"No {what} at {path}. Train first "
                "(train.py / train_classifier.py) or pass an explicit path."
            )

    print(f"device     : {device}")
    model, cfg = load_vae(str(ckpt_path), device=device)
    judge, _ = load_classifier(args.classifier, device=device)
    print(f"VAE        : {ckpt_path}  (L={cfg['latent_dim']})")
    print(f"classifier : {args.classifier}")

    # Encode the test set -> per-class latent centroids.
    _, test_loader = mnist_loaders(root=args.data_root, batch_size=256, download=False)
    mu, labels = encode_latents(model, test_loader, device)
    means = class_means(mu, labels, N_CLASSES)  # (10, L)

    # --- Targeted sampling: z ~ N(mu_c, spread*I) ---
    latent_dim = means.size(1)
    hit_rates = []
    grid_rows = []
    for c in range(N_CLASSES):
        eps = torch.randn(args.n_per_class, latent_dim)
        z = means[c].unsqueeze(0) + args.spread * eps
        imgs = decode(model, z, device)
        preds, _ = classify(judge, imgs, device)
        hit_rates.append((preds == c).float().mean().item())
        grid_rows.append(imgs[: args.n_show])
    for c, hr in enumerate(hit_rates):
        print(f"  digit {c}: hit rate {hr * 100:5.1f}%")
    print(f"  mean hit rate: {sum(hit_rates) / N_CLASSES * 100:.1f}%")

    # --- Prior baseline: z ~ N(0, I) ---
    z_prior = torch.randn(args.n_prior, latent_dim)
    prior_imgs = decode(model, z_prior, device)
    prior_preds, prior_conf = classify(judge, prior_imgs, device)
    prior_hist = torch.bincount(prior_preds, minlength=N_CLASSES).float()
    prior_frac = (prior_hist / prior_hist.sum())
    low_conf_frac = (prior_conf < GARBAGE_CONF).float().mean().item()
    print(f"  prior: mean confidence {prior_conf.mean() * 100:.1f}%, "
          f"low-confidence (<{GARBAGE_CONF:g}) fraction {low_conf_frac * 100:.1f}%")

    # --- Plots ---
    plots = run_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    ax = bar_by_class(hit_rates, "hit rate", as_percent=True,
                      title=f"{args.run_id} · targeted hit rate (spread={args.spread:g})")
    ax.figure.savefig(plots / "hit_rate.png", dpi=150, bbox_inches="tight")
    plt.close(ax.figure)

    grid = torch.cat(grid_rows, dim=0)
    ax = image_grid(grid, nrow=N_CLASSES, ncol=args.n_show,
                    row_labels=[str(c) for c in range(N_CLASSES)],
                    title=f"{args.run_id} · samples near each class mean")
    ax.figure.savefig(plots / "class_samples.png", dpi=150, bbox_inches="tight")
    plt.close(ax.figure)

    ax = bar_by_class(prior_frac, "fraction of prior samples", as_percent=True,
                      title=f"{args.run_id} · prior N(0,I) digit histogram")
    ax.figure.savefig(plots / "prior_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(ax.figure)

    results = {
        "run_id": args.run_id,
        "latent_dim": latent_dim,
        "spread": args.spread,
        "n_per_class": args.n_per_class,
        "hit_rate_per_class": hit_rates,
        "mean_hit_rate": sum(hit_rates) / N_CLASSES,
        "prior_histogram_fraction": prior_frac.tolist(),
        "prior_mean_confidence": prior_conf.mean().item(),
        "prior_low_confidence_fraction": low_conf_frac,
    }
    (run_dir / "step4_results.json").write_text(json.dumps(results, indent=2))
    print(f"saved      : {plots}/ (3 PNGs) and {run_dir / 'step4_results.json'}")


if __name__ == "__main__":
    main()
