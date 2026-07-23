"""Phase 1 · Step 1 — baseline MLP VAE on MNIST.

Trains, evaluates, logs one JSON line per epoch, and writes log-spaced checkpoints.
See ``docs/phase1_plan_23-07-2026.md`` for what this step is for.

Usage (on the PC, RTX 4090):

    uv run python experiments/phase1_vae/train.py --smoke
    uv run python experiments/phase1_vae/train.py --epochs 30

MNIST must already be on disk; this script will not download it.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field

import torch
from tqdm import tqdm

from synapses import get_device
from synapses.data import binarize, mnist_loaders
from synapses.losses import vae_loss
from synapses.run import (
    RunLogger,
    log_spaced_epochs,
    make_run_dir,
    save_checkpoint,
    set_seed,
)
from synapses.vae import MLPVAE

SMOKE_BATCHES = 20


@dataclass
class Phase1Config:
    latent_dim: int = 2
    hidden_dim: int = 400
    beta: float = 1.0
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 30
    seed: int = 0
    device: str | None = None
    data_root: str = "data"
    out_root: str = "outputs"
    num_workers: int = 2
    smoke: bool = False
    tags: dict = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        stem = f"phase1_L{self.latent_dim}_beta{self.beta:g}_s{self.seed}"
        return f"{stem}_smoke" if self.smoke else stem


def parse_args() -> Phase1Config:
    p = argparse.ArgumentParser(description="Phase 1 Step 1 — baseline MNIST VAE")
    d = Phase1Config()
    p.add_argument("--latent-dim", type=int, default=d.latent_dim)
    p.add_argument("--hidden-dim", type=int, default=d.hidden_dim)
    p.add_argument("--beta", type=float, default=d.beta)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", type=str, default=d.device, help="override device selection")
    p.add_argument("--data-root", type=str, default=d.data_root)
    p.add_argument("--out-root", type=str, default=d.out_root)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument(
        "--smoke",
        action="store_true",
        help=f"fast sanity run: 2 epochs, {SMOKE_BATCHES} batches each",
    )
    args = p.parse_args()

    cfg = Phase1Config(**vars(args))
    if cfg.smoke:
        cfg.epochs = 2
    return cfg


@torch.no_grad()
def evaluate(model: MLPVAE, loader, device: torch.device, max_batches: int | None = None) -> dict:
    """Eval pass over the fixed-binarized test set.

    Reconstruction and KL are reported at beta=1 regardless of the training beta — the
    reported ELBO is the true one, and beta only ever weights the training objective.
    """
    model.eval()
    n = 0
    recon_sum = 0.0
    kl_per_dim_sum = torch.zeros(model.latent_dim, device=device)
    mus = []

    for i, (x, _) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True).view(x.size(0), -1)
        logits, mu, logvar = model(x)
        out = vae_loss(logits, x, mu, logvar, beta=1.0)

        bs = x.size(0)
        n += bs
        recon_sum += out.recon.item() * bs
        kl_per_dim_sum += out.kl_per_dim.detach() * bs
        mus.append(mu.detach())

    recon = recon_sum / n
    kl_per_dim = (kl_per_dim_sum / n).cpu()
    kl = kl_per_dim.sum().item()

    # Active units (Burda et al.): dimensions whose posterior mean actually varies
    # across the dataset. Uninformative at L=2; the headline number in Step 5.
    mu_all = torch.cat(mus, dim=0)
    active_units = int((mu_all.var(dim=0) > 0.01).sum().item())

    return {
        "test_recon": recon,
        "test_kl": kl,
        "test_elbo": -(recon + kl),
        "kl_per_dim": kl_per_dim.tolist(),
        "active_units": active_units,
    }


def train_one_epoch(
    model: MLPVAE,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float,
    epoch: int,
    max_batches: int | None = None,
) -> dict:
    model.train()
    n = 0
    loss_sum = recon_sum = kl_sum = 0.0

    total = len(loader) if max_batches is None else min(max_batches, len(loader))
    bar = tqdm(loader, total=total, desc=f"epoch {epoch:>3}", leave=False)
    for i, (x, _) in enumerate(bar):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        x = binarize(x).view(x.size(0), -1)  # dynamic: resampled every epoch

        logits, mu, logvar = model(x)
        out = vae_loss(logits, x, mu, logvar, beta=beta)

        optimizer.zero_grad(set_to_none=True)
        out.loss.backward()
        optimizer.step()

        bs = x.size(0)
        n += bs
        loss_sum += out.loss.item() * bs
        recon_sum += out.recon.item() * bs
        kl_sum += out.kl.item() * bs
        bar.set_postfix(loss=f"{loss_sum / n:.1f}")

    return {
        "train_loss": loss_sum / n,
        "train_recon": recon_sum / n,
        "train_kl": kl_sum / n,
    }


def init_sanity_check(model: MLPVAE, loader, device: torch.device) -> None:
    """At init the decoder is near-uniform, so recon must be ~784 * ln2 = 543.4 nats.

    If this prints ~0.69 instead, the reconstruction term is being averaged over pixels
    rather than summed, the KL is ~784x too strong, and every downstream beta
    measurement would be meaningless. See ``synapses/losses.py``.
    """
    expected = 784 * math.log(2)
    stats = evaluate(model, loader, device, max_batches=4)
    recon, kl = stats["test_recon"], stats["test_kl"]
    print(f"init check : recon = {recon:7.2f} nats (expect ~{expected:.1f}), kl = {kl:.3f} (~0)")
    if recon < 10:
        print(
            "  !! recon is ~1/784 of the expected value — the loss is averaging over\n"
            "     pixels instead of summing. Fix before trusting anything downstream."
        )
    elif not 0.7 * expected < recon < 1.4 * expected:
        print(f"  !! recon is off the expected ~{expected:.0f}; check init and binarization.")


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    device = get_device(cfg.device)
    print(f"=== {cfg.run_id} ===")
    print(f"device     : {device}")
    if device.type != "cuda":
        print("  note: not running on CUDA — expected 'cuda' on the PC (RTX 4090).")

    train_loader, test_loader = mnist_loaders(
        root=cfg.data_root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        download=False,
        seed=cfg.seed,
    )
    print(f"data       : {len(train_loader.dataset)} train / {len(test_loader.dataset)} test")

    model = MLPVAE(784, cfg.hidden_dim, cfg.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"model      : MLPVAE(L={cfg.latent_dim}, hidden={cfg.hidden_dim}) "
        f"— {n_params:,} params"
    )

    init_sanity_check(model, test_loader, device)

    run_dir = make_run_dir(cfg.out_root, cfg.run_id)
    ckpt_epochs = log_spaced_epochs(cfg.epochs)
    max_batches = SMOKE_BATCHES if cfg.smoke else None

    # Freeze a held-out batch alongside the run so activations extracted in Phase 3 are
    # comparable across checkpoints.
    eval_x, eval_y = next(iter(test_loader))
    torch.save({"x": eval_x, "y": eval_y}, run_dir / "eval_batch.pt")

    print(f"run dir    : {run_dir}")
    print(f"checkpoints: epochs {sorted(ckpt_epochs)}\n")

    with RunLogger(run_dir, cfg) as logger:
        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            train_stats = train_one_epoch(
                model, train_loader, optimizer, device, cfg.beta, epoch, max_batches
            )
            test_stats = evaluate(model, test_loader, device, max_batches)
            seconds = time.time() - t0

            logger.log(
                epoch=epoch,
                **train_stats,
                **test_stats,
                lr=optimizer.param_groups[0]["lr"],
                seconds=seconds,
            )
            print(
                f"epoch {epoch:>3}/{cfg.epochs}  "
                f"train {train_stats['train_loss']:7.2f}  "
                f"test elbo {test_stats['test_elbo']:8.2f}  "
                f"recon {test_stats['test_recon']:7.2f}  "
                f"kl {test_stats['test_kl']:6.2f}  "
                f"au {test_stats['active_units']:>3}  "
                f"{seconds:5.1f}s"
            )

            if epoch in ckpt_epochs:
                save_checkpoint(
                    run_dir / "ckpt" / f"epoch_{epoch:04d}.pt", model, optimizer, epoch, cfg
                )

    print(f"\nDone. Metrics: {run_dir / 'metrics.jsonl'}")


if __name__ == "__main__":
    main()
