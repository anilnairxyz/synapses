"""Train the SmallCNN judge for Step 4.

An independent MNIST classifier (~99%) used to score generated samples in sample.py. It
sees plain [0,1] images (ToTensor, not binarized) to match the VAE decoder's probability
outputs. Saves the best-by-test-accuracy checkpoint to outputs/mnist_cnn/ckpt/best.pt.

Usage:
    uv run python experiments/epic1/phase1_vae/train_classifier.py
    uv run python experiments/epic1/phase1_vae/train_classifier.py --epochs 8
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

from synapses import get_device
from synapses.classifier import SmallCNN
from synapses.run import RunLogger, make_run_dir, save_checkpoint, set_seed

DOWNLOAD_HINT = (
    "MNIST not found under {root!r}. Fetch it once with:\n"
    "    uv run python -c \"from torchvision.datasets import MNIST; "
    "MNIST('{root}', train=True, download=True); MNIST('{root}', train=False, download=True)\""
)


@dataclass
class ClassifierConfig:
    epochs: int = 5
    lr: float = 1e-3
    batch_size: int = 128
    seed: int = 0
    n_classes: int = 10
    device: str | None = None
    data_root: str = "data"
    out_root: str = "outputs"
    run_id: str = "mnist_cnn"


def parse_args() -> ClassifierConfig:
    p = argparse.ArgumentParser(description="Train the SmallCNN judge (Step 4)")
    d = ClassifierConfig()
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", type=str, default=d.device)
    p.add_argument("--data-root", type=str, default=d.data_root)
    p.add_argument("--out-root", type=str, default=d.out_root)
    return ClassifierConfig(**vars(p.parse_args()))


def loaders(root: str, batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Plain [0,1] MNIST, shape (B,1,28,28). Own loaders (not the binarized VAE ones)."""
    try:
        train = MNIST(root, train=True, download=False, transform=transforms.ToTensor())
        test = MNIST(root, train=False, download=False, transform=transforms.ToTensor())
    except RuntimeError as exc:
        raise RuntimeError(DOWNLOAD_HINT.format(root=root)) from exc
    pin = torch.cuda.is_available()
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin),
    )


@torch.no_grad()
def evaluate(model: SmallCNN, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"=== {cfg.run_id} ===\ndevice     : {device}")
    if device.type != "cuda":
        print("  note: not running on CUDA (fine on the Mac; expected 'cuda' on the PC).")

    train_loader, test_loader = loaders(cfg.data_root, cfg.batch_size)
    model = SmallCNN(cfg.n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    run_dir = make_run_dir(cfg.out_root, cfg.run_id)
    best_acc = 0.0
    best_path = run_dir / "ckpt" / "best.pt"

    with RunLogger(run_dir, cfg) as logger:
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            t0 = time.time()
            loss_sum = n = 0
            for x, y in tqdm(train_loader, desc=f"epoch {epoch:>2}", leave=False):
                x, y = x.to(device), y.to(device)
                loss = F.cross_entropy(model(x), y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * y.size(0)
                n += y.size(0)

            acc = evaluate(model, test_loader, device)
            secs = time.time() - t0
            logger.log(epoch=epoch, train_loss=loss_sum / n, test_acc=acc, seconds=secs)
            marker = ""
            if acc > best_acc:
                best_acc = acc
                save_checkpoint(best_path, model, optimizer, epoch, cfg)
                marker = "  <- best"
            print(f"epoch {epoch:>2}/{cfg.epochs}  loss {loss_sum / n:.4f}  "
                  f"test acc {acc * 100:.2f}%  {secs:.1f}s{marker}")

    print(f"\nBest test accuracy: {best_acc * 100:.2f}%\nsaved: {best_path}")


if __name__ == "__main__":
    main()
