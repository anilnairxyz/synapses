"""Phase 2 · Step 3 -- SimCLR (contrastive) on CIFAR-10.

The sample-side answer to representation collapse: negatives (every other image in the
batch) make equal representations indistinguishable from the positive, so collapsing to a
constant vector scores badly on the loss itself. No stop-gradient, no EMA, no variance
term. See docs/epic1_phase2_plan_26-07-2026.md (Step 3).

Usage (on the PC, RTX 4090):

    uv run python experiments/epic1/phase2_ssl/train_simclr.py --smoke
    uv run python experiments/epic1/phase2_ssl/train_simclr.py

CIFAR-10 must already be on disk; this script will not download it.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from synapses import get_device
from synapses.data import cifar10_eval_loaders, cifar10_ssl_loader
from synapses.run import (
    RunLogger,
    cosine_warmup_lr,
    load_checkpoint,
    log_spaced_epochs,
    make_run_dir,
    save_checkpoint,
    set_seed,
)
from synapses.ssl_augment import TwoViewTransform, ssl_transform
from synapses.ssl_backbone import CifarResNet18
from synapses.ssl_eval import evaluate_representation
from synapses.ssl_losses import nt_xent_loss

SMOKE_BATCHES = 5
SMOKE_PROBE_SUBSET = 500  # cap probe/kNN eval data under --smoke -- the full-scale fit
                          # (50k x 512, sklearn, CPU-bound) defeats the point of a smoke test


@dataclass
class SimCLRConfig:
    batch_size: int = 256
    epochs: int = 200
    lr: float = 0.5  # SGD, standard SimCLR-CIFAR scale at this batch size
    momentum: float = 0.9
    weight_decay: float = 1e-4
    temperature: float = 0.5
    warmup_epochs: int = 10
    projector_hidden: int = 512
    projector_out: int = 128
    aug_strength: str = "strong"  # feeds Step 6's augmentation ablation later
    seed: int = 0
    device: str | None = None
    data_root: str = "data"
    out_root: str = "outputs"
    num_workers: int = 4
    smoke: bool = False

    @property
    def run_id(self) -> str:
        stem = f"phase2_simclr_bs{self.batch_size}_t{self.temperature:g}_s{self.seed}"
        return f"{stem}_smoke" if self.smoke else stem


def parse_args() -> SimCLRConfig:
    p = argparse.ArgumentParser(description="Phase 2 Step 3 -- SimCLR on CIFAR-10")
    d = SimCLRConfig()
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--momentum", type=float, default=d.momentum)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--temperature", type=float, default=d.temperature)
    p.add_argument("--warmup-epochs", type=int, default=d.warmup_epochs)
    p.add_argument("--aug-strength", type=str, default=d.aug_strength,
                    choices=["strong", "weak", "minimal"])
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", type=str, default=d.device)
    p.add_argument("--data-root", type=str, default=d.data_root)
    p.add_argument("--out-root", type=str, default=d.out_root)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument(
        "--smoke", action="store_true",
        help=f"fast sanity run: 2 epochs, {SMOKE_BATCHES} batches each",
    )
    args = p.parse_args()

    cfg = SimCLRConfig(**vars(args))
    if cfg.smoke:
        cfg.epochs = 2
    return cfg


def _capped(loader: DataLoader, n: int) -> DataLoader:
    """Cap a DataLoader's dataset to at most n examples, keeping its batch size."""
    ds = loader.dataset
    idx = list(range(min(n, len(ds))))
    return DataLoader(Subset(ds, idx), batch_size=loader.batch_size, shuffle=False)


def build_projector(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """Standard SimCLR projector: Linear -> BN -> ReLU -> Linear. Discarded at eval time;
    only the backbone's output is ever probed."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


class SimCLRModel(nn.Module):
    """Backbone + projector bundled so one checkpoint covers both (mirrors MLPVAE
    bundling encoder+decoder in Phase 1)."""

    def __init__(self, backbone: CifarResNet18, projector: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.projector = projector

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        z = self.projector(h)
        return h, z


def load_simclr(ckpt_path: str, device: torch.device | str = "cpu") -> tuple[SimCLRModel, dict]:
    """Rebuild a trained SimCLRModel from a checkpoint written by save_checkpoint.

    Mirrors synapses.vae.load_vae's pattern: read the stored config to reconstruct the
    architecture, restore weights, move to device, eval mode. Returns (model, config).
    """
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    backbone = CifarResNet18()
    projector = build_projector(backbone.out_dim, cfg["projector_hidden"], cfg["projector_out"])
    model = SimCLRModel(backbone, projector)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg


def init_sanity_check(model: SimCLRModel, loader, device: torch.device, temperature: float) -> None:
    """At init, similarities concentrate near 0, so NT-Xent should read close to
    log(2N-1) -- the uniform-over-negatives value. See tests/test_ssl_losses.py for the
    same invariant checked in isolation."""
    model.eval()
    (v1, v2), _ = next(iter(loader))
    n = v1.size(0)
    with torch.no_grad():
        _, z1 = model(v1.to(device))
        _, z2 = model(v2.to(device))
        loss = nt_xent_loss(z1, z2, temperature=temperature)
    expected = math.log(2 * n - 1)
    print(f"init check : NT-Xent = {loss.item():.3f} (expect ~{expected:.3f})")
    if abs(loss.item() - expected) > 1.0:
        print("  !! far from log(2N-1) -- check the projector/masking/temperature wiring.")
    model.train()


def train_one_epoch(
    model: SimCLRModel,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    temperature: float,
    epoch: int,
    max_batches: int | None = None,
) -> float:
    model.train()
    loss_sum = n = 0
    total = len(loader) if max_batches is None else min(max_batches, len(loader))
    bar = tqdm(loader, total=total, desc=f"epoch {epoch:>3}", leave=False)
    for i, ((v1, v2), _) in enumerate(bar):  # label discarded -- never touches training
        if max_batches is not None and i >= max_batches:
            break
        v1, v2 = v1.to(device, non_blocking=True), v2.to(device, non_blocking=True)
        _, z1 = model(v1)
        _, z2 = model(v2)
        loss = nt_xent_loss(z1, z2, temperature=temperature)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        bs = v1.size(0)
        loss_sum += loss.item() * bs
        n += bs
        bar.set_postfix(loss=f"{loss_sum / n:.3f}")
    return loss_sum / n


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    device = get_device(cfg.device)
    print(f"=== {cfg.run_id} ===")
    print(f"device     : {device}")
    if device.type != "cuda":
        print("  note: not running on CUDA -- expected 'cuda' on the PC (RTX 4090).")

    transform = TwoViewTransform(ssl_transform(cfg.aug_strength))
    train_loader = cifar10_ssl_loader(
        root=cfg.data_root, transform=transform, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, download=False, seed=cfg.seed,
    )
    probe_train_loader, probe_test_loader = cifar10_eval_loaders(
        root=cfg.data_root, batch_size=cfg.batch_size, download=False,
    )
    if cfg.smoke:
        # Full-scale probe fit (50k x 512, sklearn lbfgs, CPU-bound) is slow and pointless
        # for a sanity check -- cap it so --smoke stays fast while still exercising the
        # real evaluate_representation() code path.
        probe_train_loader = _capped(probe_train_loader, SMOKE_PROBE_SUBSET)
        probe_test_loader = _capped(probe_test_loader, SMOKE_PROBE_SUBSET)
    print(f"data       : {len(train_loader.dataset)} train (SSL) / "
          f"{len(probe_test_loader.dataset)} test (probe), aug={cfg.aug_strength}")

    backbone = CifarResNet18()
    projector = build_projector(backbone.out_dim, cfg.projector_hidden, cfg.projector_out)
    model = SimCLRModel(backbone, projector).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model      : SimCLRModel (ResNet-18 + projector) -- {n_params:,} params")

    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay,
    )
    max_batches = SMOKE_BATCHES if cfg.smoke else None
    steps_per_epoch = max_batches if max_batches is not None else len(train_loader)
    scheduler = cosine_warmup_lr(
        optimizer,
        warmup_steps=cfg.warmup_epochs * steps_per_epoch,
        total_steps=cfg.epochs * steps_per_epoch,
    )

    init_sanity_check(model, train_loader, device, cfg.temperature)

    run_dir = make_run_dir(cfg.out_root, cfg.run_id)
    ckpt_epochs = log_spaced_epochs(cfg.epochs)
    print(f"run dir    : {run_dir}")
    print(f"checkpoints: epochs {sorted(ckpt_epochs)} (probe/kNN evaluated at these too)\n")

    with RunLogger(run_dir, cfg) as logger:
        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, device,
                cfg.temperature, epoch, max_batches,
            )
            seconds = time.time() - t0

            log_kwargs = {"epoch": epoch, "train_nt_xent": train_loss,
                          "lr": optimizer.param_groups[0]["lr"], "seconds": seconds}
            line = (f"epoch {epoch:>3}/{cfg.epochs}  nt_xent {train_loss:7.3f}  "
                   f"lr {optimizer.param_groups[0]['lr']:.4f}  {seconds:5.1f}s")

            if epoch in ckpt_epochs:
                # Evaluation is not cheap (fitting a probe on 50k x 512 features on CPU
                # via sklearn), so it only runs at the same log-spaced epochs already
                # being checkpointed. Timed separately so it's clear this -- not a hang
                # -- is why the gap to the next printed line is longer than `seconds`.
                t_eval = time.time()
                metrics = evaluate_representation(
                    model.backbone, probe_train_loader, probe_test_loader, device,
                )
                eval_seconds = time.time() - t_eval
                log_kwargs.update(
                    linear_probe=metrics["linear_probe"], knn=metrics["knn"],
                    eval_seconds=eval_seconds,
                )
                line += (f"  probe {metrics['linear_probe'] * 100:5.1f}%  "
                        f"knn {metrics['knn'] * 100:5.1f}%  (eval {eval_seconds:.1f}s)")
                save_checkpoint(run_dir / "ckpt" / f"epoch_{epoch:04d}.pt",
                               model, optimizer, epoch, cfg)

            logger.log(**log_kwargs)
            print(line)

    print(f"\nDone. Metrics: {run_dir / 'metrics.jsonl'}")


if __name__ == "__main__":
    main()
