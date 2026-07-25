"""A small MNIST CNN — the independent judge for Step 4.

Step 4 measures how often a sample drawn near a digit's latent cluster actually decodes
to that digit. To trust that number we need a classifier that is *independent* of the VAE
— so a low hit rate blames the generator, not the judge. This is a plain conv net
(~99% on MNIST), trained once and reused.

It sees images in **[0,1]** (trained on plain ``ToTensor`` MNIST), matching the decoder's
probability outputs (sigmoid), so there's no train/judge distribution mismatch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from synapses.run import load_checkpoint


class SmallCNN(nn.Module):
    """conv(1->32) -> conv(32->64) -> pool -> fc(128) -> fc(10)."""

    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` of shape ``(B, 1, 28, 28)`` in [0,1] -> logits ``(B, n_classes)``."""
        return self.classifier(self.features(x))


def load_classifier(ckpt_path: str, device: torch.device | str = "cpu") -> tuple[SmallCNN, dict]:
    """Rebuild a trained SmallCNN from a checkpoint. Returns ``(model.eval(), config)``."""
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {}) or {}
    model = SmallCNN(n_classes=cfg.get("n_classes", 10))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg


def _as_images(images: torch.Tensor) -> torch.Tensor:
    """Accept ``(N, 784)`` or ``(N, 1, 28, 28)`` (or ``(N, 28, 28)``) -> ``(N, 1, 28, 28)``."""
    if images.dim() == 2:  # (N, 784)
        return images.view(-1, 1, 28, 28)
    if images.dim() == 3:  # (N, 28, 28)
        return images.unsqueeze(1)
    return images


@torch.no_grad()
def classify(
    model: SmallCNN,
    images: torch.Tensor,
    device: torch.device | str,
    batch_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classify a batch of images.

    Returns ``(preds, confidences)`` on CPU: argmax labels and the max-softmax
    probability of each (the latter used as a garbage proxy — a blurry between-digit
    sample gets low confidence).
    """
    model.eval()
    x = _as_images(images)
    preds: list[torch.Tensor] = []
    confs: list[torch.Tensor] = []
    for i in range(0, x.size(0), batch_size):
        chunk = x[i : i + batch_size].to(device, non_blocking=True)
        probs = F.softmax(model(chunk), dim=1)
        conf, pred = probs.max(dim=1)
        preds.append(pred.cpu())
        confs.append(conf.cpu())
    return torch.cat(preds), torch.cat(confs)
