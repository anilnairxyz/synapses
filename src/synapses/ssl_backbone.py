"""CIFAR-adapted ResNet-18 backbone for Phase 2 self-supervised learning.

The standard ImageNet ResNet-18 stem (7x7 stride-2 conv + stride-2 maxpool) throws away
most of a 32x32 CIFAR image before the first residual block ever sees it. This swaps the
stem for a single 3x3 stride-1 conv and drops the maxpool -- the adaptation used by every
CIFAR SimCLR/SimSiam/VICReg reference implementation -- so spatial resolution starts at
32x32 and only shrinks through the four stages' internal strides (32->16->8->4).

Built from torchvision's BasicBlock directly, so this is the standard ResNet-18 residual
structure (four stages of two blocks, channels 64/128/256/512) without pulling in an
external CIFAR-resnet package -- no new dependency.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models.resnet import BasicBlock


def _make_stage(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
    """Two BasicBlocks; the first carries the stage's stride and any channel change."""
    downsample = None
    if stride != 1 or in_channels != out_channels:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    return nn.Sequential(
        BasicBlock(in_channels, out_channels, stride=stride, downsample=downsample),
        BasicBlock(out_channels, out_channels, stride=1),
    )


class CifarResNet18(nn.Module):
    """ResNet-18 with a CIFAR stem. ``forward`` maps ``(B, 3, 32, 32) -> (B, out_dim)``.

    This ``out_dim``-vector is *the representation* for every SSL method in this phase --
    the convention (established by SimCLR) of evaluating the backbone, not the projection
    head, holds throughout.
    """

    def __init__(self, out_dim: int = 512) -> None:
        super().__init__()
        self.out_dim = out_dim

        # CIFAR stem: 3x3 stride-1, no maxpool -- keeps full 32x32 resolution into block 1.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = _make_stage(64, 64, stride=1)  # 32x32
        self.layer2 = _make_stage(64, 128, stride=2)  # 16x16
        self.layer3 = _make_stage(128, 256, stride=2)  # 8x8
        self.layer4 = _make_stage(256, out_dim, stride=2)  # 4x4
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        return self.pool(h).flatten(1)
