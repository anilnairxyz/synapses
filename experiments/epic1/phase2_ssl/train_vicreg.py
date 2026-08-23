"""Phase 2 · Step 5 -- VICReg (regularisation) on CIFAR-10.

The regularisation answer to representation collapse: no negatives, no predictor, no
stop-gradient -- both branches are perfectly symmetric, and the whole anti-collapse
mechanism lives in one explicit term of the loss (the variance hinge). Unlike SimSiam,
the break-it experiment needs no special flag: the variance term already has a
continuous weight, so `--var-weight 0.0` *is* the ablation. See
docs/epic1_phase2_plan_26-07-2026.md (Step 5).

Usage (on the PC, RTX 4090):

    uv run python experiments/epic1/phase2_ssl/train_vicreg.py --smoke
    uv run python experiments/epic1/phase2_ssl/train_vicreg.py                      # healthy
    uv run python experiments/epic1/phase2_ssl/train_vicreg.py --epochs 20 --var-weight 0.0

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
from synapses.ssl_losses import vicreg_loss

SMOKE_BATCHES = 5
SMOKE_PROBE_SUBSET = 500


@dataclass
class VICRegConfig:
    batch_size: int = 256  # matched to SimCLR/SimSiam by design
    epochs: int = 200
    lr: float = 0.5
    momentum: float = 0.9
    weight_decay: float = 1e-4
    expander_hidden: int = 2048  # a wide expander, not a bottleneck -- see build_expander
    expander_out: int = 2048
    inv_weight: float = 25.0
    var_weight: float = 25.0  # --var-weight 0.0 *is* the break-it experiment
    cov_weight: float = 1.0
    gamma: float = 1.0
    warmup_epochs: int = 10
    aug_strength: str = "strong"
    seed: int = 0
    device: str | None = None
    data_root: str = "data"
    out_root: str = "outputs"
    num_workers: int = 4
    smoke: bool = False

    @property
    def run_id(self) -> str:
        stem = f"phase2_vicreg_bs{self.batch_size}_var{self.var_weight:g}_s{self.seed}"
        return f"{stem}_smoke" if self.smoke else stem


def parse_args() -> VICRegConfig:
    p = argparse.ArgumentParser(description="Phase 2 Step 5 -- VICReg on CIFAR-10")
    d = VICRegConfig()
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--momentum", type=float, default=d.momentum)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--expander-hidden", type=int, default=d.expander_hidden)
    p.add_argument("--expander-out", type=int, default=d.expander_out)
    p.add_argument("--inv-weight", type=float, default=d.inv_weight)
    p.add_argument("--var-weight", type=float, default=d.var_weight,
                    help="0.0 is the break-it ablation: drops the anti-collapse term")
    p.add_argument("--cov-weight", type=float, default=d.cov_weight)
    p.add_argument("--gamma", type=float, default=d.gamma)
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

    cfg = VICRegConfig(**vars(args))
    if cfg.smoke:
        cfg.epochs = 2
    return cfg


def _capped(loader: DataLoader, n: int) -> DataLoader:
    """Cap a DataLoader's dataset to at most n examples, keeping its batch size."""
    ds = loader.dataset
    idx = list(range(min(n, len(ds))))
    return DataLoader(Subset(ds, idx), batch_size=loader.batch_size, shuffle=False)


def build_expander(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """VICReg's wide expander: Linear-BN-ReLU x2, then a *plain* final Linear -- no
    batch norm, no activation. This is correctness-critical, not stylistic: if the
    final layer were normalised (as SimSiam's projector deliberately is), that
    normalisation would satisfy the variance term structurally regardless of what the
    loss is doing, making var_weight vacuous. The loss must act on genuinely
    unconstrained outputs."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


class VICRegModel(nn.Module):
    """Backbone + expander only -- no predictor, unlike SimSiam. forward(x) -> (h, z),
    the same shape as SimCLRModel. model.backbone is the only thing ever probed."""

    def __init__(self, backbone: CifarResNet18, expander: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.expander = expander

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        z = self.expander(h)
        return h, z


def load_vicreg(ckpt_path: str, device: torch.device | str = "cpu") -> tuple[VICRegModel, dict]:
    """Rebuild a trained VICRegModel from a checkpoint, mirroring load_simclr/load_simsiam."""
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    backbone = CifarResNet18()
    expander = build_expander(backbone.out_dim, cfg["expander_hidden"], cfg["expander_out"])
    model = VICRegModel(backbone, expander)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg


def init_sanity_check(
    model: VICRegModel, loader, device: torch.device, inv_w: float, var_w: float,
    cov_w: float, gamma: float,
) -> None:
    """Unlike NT-Xent (log(2N-1)) or SimSiam's loss (~0), VICReg's MSE-based invariance
    term has no universal theoretical value at init -- it depends on the untrained
    network's raw output scale, which isn't pinned to anything. So this check doesn't
    assert a specific number (that would be manufactured, not principled); it prints
    all three terms separately and confirms they're finite and non-negative -- a real,
    if weaker, sanity check that's honest about what can and can't be predicted here."""
    model.eval()
    (v1, v2), _ = next(iter(loader))
    with torch.no_grad():
        _, z1 = model(v1.to(device))
        _, z2 = model(v2.to(device))
        loss, terms = vicreg_loss(z1, z2, inv_w, var_w, cov_w, gamma)
    print(f"init check : loss {loss.item():.3f}  "
          f"inv {terms['inv'].item():.3f}  var {terms['var'].item():.3f}  "
          f"cov {terms['cov'].item():.3f}  (no universal target here -- checking finiteness)")
    for name, t in terms.items():
        if not torch.isfinite(t) or t.item() < 0:
            print(f"  !! {name} is non-finite or negative -- check the expander/loss wiring.")
    model.train()


def train_one_epoch(
    model: VICRegModel,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    inv_w: float, var_w: float, cov_w: float, gamma: float,
    epoch: int,
    max_batches: int | None = None,
) -> dict:
    model.train()
    sums = {"loss": 0.0, "inv": 0.0, "var": 0.0, "cov": 0.0}
    n = 0
    total = len(loader) if max_batches is None else min(max_batches, len(loader))
    bar = tqdm(loader, total=total, desc=f"epoch {epoch:>3}", leave=False)
    for i, ((v1, v2), _) in enumerate(bar):  # label discarded -- never touches training
        if max_batches is not None and i >= max_batches:
            break
        v1, v2 = v1.to(device, non_blocking=True), v2.to(device, non_blocking=True)
        _, z1 = model(v1)
        _, z2 = model(v2)
        loss, terms = vicreg_loss(z1, z2, inv_w, var_w, cov_w, gamma)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        bs = v1.size(0)
        sums["loss"] += loss.item() * bs
        for k in ("inv", "var", "cov"):
            sums[k] += terms[k].item() * bs
        n += bs
        bar.set_postfix(loss=f"{sums['loss'] / n:.3f}", var=f"{sums['var'] / n:.3f}")
    return {k: v / n for k, v in sums.items()}


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    device = get_device(cfg.device)
    print(f"=== {cfg.run_id} ===")
    print(f"device     : {device}")
    if device.type != "cuda":
        print("  note: not running on CUDA -- expected 'cuda' on the PC (RTX 4090).")
    if cfg.var_weight == 0.0:
        print("  ablation   : var_weight=0.0 -- collapse expected")

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
    expander = build_expander(backbone.out_dim, cfg.expander_hidden, cfg.expander_out)
    model = VICRegModel(backbone, expander).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model      : VICRegModel (ResNet-18 + expander) -- {n_params:,} params")

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

    init_sanity_check(model, train_loader, device, cfg.inv_weight, cfg.var_weight,
                      cfg.cov_weight, cfg.gamma)

    run_dir = make_run_dir(cfg.out_root, cfg.run_id)
    ckpt_epochs = log_spaced_epochs(cfg.epochs)
    print(f"run dir    : {run_dir}")
    print(f"checkpoints: epochs {sorted(ckpt_epochs)} (probe/kNN/rank at these too)\n")

    with RunLogger(run_dir, cfg) as logger:
        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            train_stats = train_one_epoch(
                model, train_loader, optimizer, scheduler, device,
                cfg.inv_weight, cfg.var_weight, cfg.cov_weight, cfg.gamma,
                epoch, max_batches,
            )
            seconds = time.time() - t0

            log_kwargs = {"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()},
                          "lr": optimizer.param_groups[0]["lr"], "seconds": seconds}
            line = (f"epoch {epoch:>3}/{cfg.epochs}  loss {train_stats['loss']:7.3f}  "
                   f"inv {train_stats['inv']:6.3f}  var {train_stats['var']:6.3f}  "
                   f"cov {train_stats['cov']:6.3f}  "
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
