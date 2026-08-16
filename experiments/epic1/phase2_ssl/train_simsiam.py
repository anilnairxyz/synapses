"""Phase 2 · Step 4 -- SimSiam (non-contrastive) on CIFAR-10.

The optimisation-asymmetry answer to representation collapse: a predictor on one branch
plus stop-gradient on the target branch, no negatives, no EMA, no queue. Unlike NT-Xent,
SimSiam's loss is NOT diagnostic of collapse -- a healthy run and a collapsed run both
drive the loss toward -1, so every checkpoint also tracks `participation_ratio` (the
effective rank of the feature covariance -- scale- and dimension-free, see ssl_eval.py)
on the frozen backbone's features to tell them apart. See
docs/epic1_phase2_plan_26-07-2026.md (Step 4).

Usage (on the PC, RTX 4090):

    uv run python experiments/epic1/phase2_ssl/train_simsiam.py --smoke
    uv run python experiments/epic1/phase2_ssl/train_simsiam.py                       # healthy
    uv run python experiments/epic1/phase2_ssl/train_simsiam.py --epochs 20 --no-stop-gradient
    uv run python experiments/epic1/phase2_ssl/train_simsiam.py --epochs 20 --no-predictor

CIFAR-10 must already be on disk; this script will not download it.
"""

from __future__ import annotations

import argparse
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
from synapses.ssl_losses import simsiam_loss

SMOKE_BATCHES = 5
SMOKE_PROBE_SUBSET = 500


@dataclass
class SimSiamConfig:
    batch_size: int = 256  # matched to SimCLR by design, not a SimSiam requirement
    epochs: int = 200
    lr: float = 0.5
    momentum: float = 0.9
    weight_decay: float = 1e-4
    projector_hidden: int = 512
    projector_out: int = 128  # matches SimCLR's z-dim, for comparable embeddings
    predictor_hidden: int = 64  # a bottleneck, scaled down from the paper's ratio
    warmup_epochs: int = 10
    aug_strength: str = "strong"
    no_stop_gradient: bool = False  # the signature break-it flag
    no_predictor: bool = False  # the second ablation
    seed: int = 0
    device: str | None = None
    data_root: str = "data"
    out_root: str = "outputs"
    num_workers: int = 4
    smoke: bool = False

    @property
    def run_id(self) -> str:
        stem = f"phase2_simsiam_bs{self.batch_size}_s{self.seed}"
        if self.no_stop_gradient:
            stem += "_nostopgrad"
        if self.no_predictor:
            stem += "_nopred"
        return f"{stem}_smoke" if self.smoke else stem


def parse_args() -> SimSiamConfig:
    p = argparse.ArgumentParser(description="Phase 2 Step 4 -- SimSiam on CIFAR-10")
    d = SimSiamConfig()
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--momentum", type=float, default=d.momentum)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--projector-hidden", type=int, default=d.projector_hidden)
    p.add_argument("--projector-out", type=int, default=d.projector_out)
    p.add_argument("--predictor-hidden", type=int, default=d.predictor_hidden)
    p.add_argument("--warmup-epochs", type=int, default=d.warmup_epochs)
    p.add_argument("--aug-strength", type=str, default=d.aug_strength,
                    choices=["strong", "weak", "minimal"])
    p.add_argument("--no-stop-gradient", action="store_true",
                    help="break-it ablation: remove the stop-gradient (expect collapse)")
    p.add_argument("--no-predictor", action="store_true",
                    help="break-it ablation: drop the predictor head (expect collapse)")
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

    cfg = SimSiamConfig(**vars(args))
    if cfg.smoke:
        cfg.epochs = 2
    return cfg


def _capped(loader: DataLoader, n: int) -> DataLoader:
    """Cap a DataLoader's dataset to at most n examples, keeping its batch size."""
    ds = loader.dataset
    idx = list(range(min(n, len(ds))))
    return DataLoader(Subset(ds, idx), batch_size=loader.batch_size, shuffle=False)


def build_projector(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """SimSiam's 3-layer projector: Linear-BN-ReLU x2, then Linear-BN(no affine). The
    final batch norm with no learnable affine parameters is a specific detail from the
    original paper's own implementation, known to matter for stability -- cheap to keep.
    Deliberately not shared with SimCLR's 2-layer projector; the depths genuinely differ."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
        nn.BatchNorm1d(out_dim, affine=False),
    )


def build_predictor(dim: int, hidden_dim: int) -> nn.Sequential:
    """The 2-layer bottleneck predictor: dim -> hidden -> dim, plain linear output."""
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, dim),
    )


class SimSiamModel(nn.Module):
    """Backbone + projector + predictor bundled so one checkpoint covers all three.

    The predictor-ablation trick lives entirely at construction time: pass
    ``nn.Identity()`` as ``predictor`` and ``forward`` degrades to ``p = z`` with no
    special-casing here -- see ``main()``.
    """

    def __init__(self, backbone: CifarResNet18, projector: nn.Module, predictor: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.projector = projector
        self.predictor = predictor

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        z = self.projector(h)
        p = self.predictor(z)
        return h, z, p


def load_simsiam(ckpt_path: str, device: torch.device | str = "cpu") -> tuple[SimSiamModel, dict]:
    """Rebuild a trained SimSiamModel from a checkpoint, mirroring load_simclr /
    load_vae. Reconstructs the predictor as nn.Identity() if the checkpoint's stored
    config has no_predictor=True, so a deliberately-broken ablation checkpoint loads
    back with its actual (missing) predictor, not a freshly-initialized real one."""
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    backbone = CifarResNet18()
    projector = build_projector(backbone.out_dim, cfg["projector_hidden"], cfg["projector_out"])
    predictor: nn.Module
    if cfg.get("no_predictor", False):
        predictor = nn.Identity()
    else:
        predictor = build_predictor(cfg["projector_out"], cfg["predictor_hidden"])
    model = SimSiamModel(backbone, projector, predictor)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg


def init_sanity_check(model: SimSiamModel, loader, device: torch.device) -> None:
    """At init, p and z are near-random unit vectors, so cosine similarity concentrates
    near 0 and the loss should read close to 0 -- NOT nt_xent_loss's log(2N-1). A
    different loss, a different invariant; see tests/test_ssl_losses.py.

    Exception: with the predictor ablated (p = z identically), this invariant does not
    hold. Untrained deep CNNs tend to produce highly-correlated outputs for different
    random inputs (a known signal-propagation artifact, unrelated to SimSiam), so with
    no predictor to break the symmetry, the loss can already sit close to its collapsed
    value (-1) before a single gradient step -- expected for that ablation, not a bug.
    """
    model.eval()
    (v1, v2), _ = next(iter(loader))
    with torch.no_grad():
        _, z1, p1 = model(v1.to(device))
        _, z2, p2 = model(v2.to(device))
        loss = simsiam_loss(p1, z1, p2, z2)

    if isinstance(model.predictor, nn.Identity):
        print(f"init check : SimSiam loss = {loss.item():.3f} "
              "(no predictor -> p=z; starting near the collapsed value here is expected, "
              "not a bug -- see docstring)")
    else:
        print(f"init check : SimSiam loss = {loss.item():.3f} (expect ~0.0)")
        if abs(loss.item()) > 0.3:
            print("  !! far from 0 -- check the predictor/projector wiring.")
    model.train()


def train_one_epoch(
    model: SimSiamModel,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    stop_gradient: bool,
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
        _, z1, p1 = model(v1)
        _, z2, p2 = model(v2)
        loss = simsiam_loss(p1, z1, p2, z2, stop_gradient=stop_gradient)

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
    if cfg.no_stop_gradient or cfg.no_predictor:
        print(f"  ablation   : no_stop_gradient={cfg.no_stop_gradient} "
              f"no_predictor={cfg.no_predictor} -- collapse expected")

    transform = TwoViewTransform(ssl_transform(cfg.aug_strength))
    train_loader = cifar10_ssl_loader(
        root=cfg.data_root, transform=transform, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, download=False, seed=cfg.seed,
    )
    probe_train_loader, probe_test_loader = cifar10_eval_loaders(
        root=cfg.data_root, batch_size=cfg.batch_size, download=False,
    )
    if cfg.smoke:
        probe_train_loader = _capped(probe_train_loader, SMOKE_PROBE_SUBSET)
        probe_test_loader = _capped(probe_test_loader, SMOKE_PROBE_SUBSET)
    print(f"data       : {len(train_loader.dataset)} train (SSL) / "
          f"{len(probe_test_loader.dataset)} test (probe), aug={cfg.aug_strength}")

    backbone = CifarResNet18()
    projector = build_projector(backbone.out_dim, cfg.projector_hidden, cfg.projector_out)
    predictor = nn.Identity() if cfg.no_predictor else build_predictor(
        cfg.projector_out, cfg.predictor_hidden
    )
    model = SimSiamModel(backbone, projector, predictor).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model      : SimSiamModel (ResNet-18 + projector + predictor) -- {n_params:,} params")

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

    init_sanity_check(model, train_loader, device)

    run_dir = make_run_dir(cfg.out_root, cfg.run_id)
    ckpt_epochs = log_spaced_epochs(cfg.epochs)
    print(f"run dir    : {run_dir}")
    print(f"checkpoints: epochs {sorted(ckpt_epochs)} (probe/kNN/rank at these too)\n")

    stop_gradient = not cfg.no_stop_gradient
    with RunLogger(run_dir, cfg) as logger:
        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, device,
                stop_gradient, epoch, max_batches,
            )
            seconds = time.time() - t0

            log_kwargs = {"epoch": epoch, "train_loss": train_loss,
                          "lr": optimizer.param_groups[0]["lr"], "seconds": seconds}
            line = (f"epoch {epoch:>3}/{cfg.epochs}  loss {train_loss:7.3f}  "
                   f"lr {optimizer.param_groups[0]['lr']:.4f}  {seconds:5.1f}s")

            if epoch in ckpt_epochs:
                t_eval = time.time()
                metrics = evaluate_representation(
                    model.backbone, probe_train_loader, probe_test_loader, device,
                )
                eval_seconds = time.time() - t_eval
                log_kwargs.update(
                    linear_probe=metrics["linear_probe"], knn=metrics["knn"],
                    participation_ratio=metrics["participation_ratio"],
                    eval_seconds=eval_seconds,
                )
                # participation_ratio is the diagnostic that matters here: loss alone
                # can't distinguish a healthy run from a collapsed one.
                line += (f"  probe {metrics['linear_probe'] * 100:5.1f}%  "
                        f"knn {metrics['knn'] * 100:5.1f}%  "
                        f"rank {metrics['participation_ratio']:6.1f}/{backbone.out_dim}  "
                        f"(eval {eval_seconds:.1f}s)")
                save_checkpoint(run_dir / "ckpt" / f"epoch_{epoch:04d}.pt",
                               model, optimizer, epoch, cfg)

            logger.log(**log_kwargs)
            print(line)

    print(f"\nDone. Metrics: {run_dir / 'metrics.jsonl'}")


if __name__ == "__main__":
    main()
