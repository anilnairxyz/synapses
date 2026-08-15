"""Two-view augmentation pipeline for CIFAR-10 self-supervised learning.

Every method in Phase 2 (SimCLR, SimSiam, VICReg) trains on two independently augmented
views of the same image -- the composition of that pipeline *is* the method's inductive
bias (crop -> some scale/translation invariance, colour jitter -> colour invariance, ...).
Built once here, at three strengths, so Step 6's augmentation ablation swaps a transform
rather than reimplementing this three times.
"""

from __future__ import annotations

from torchvision import transforms

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

_STRENGTHS = ("strong", "weak", "minimal")


def ssl_transform(strength: str = "strong") -> transforms.Compose:
    """The two-view training transform, at one of three strengths.

    * ``"strong"``  -- the full SimCLR-for-CIFAR recipe: crop, flip, colour jitter,
      grayscale, blur.
    * ``"weak"``    -- crop + flip only.
    * ``"minimal"`` -- flip only.
    """
    if strength not in _STRENGTHS:
        raise ValueError(f"unknown strength {strength!r}; use one of {_STRENGTHS}")

    ops: list = []
    if strength != "minimal":
        ops.append(transforms.RandomResizedCrop(32, scale=(0.2, 1.0)))
    ops.append(transforms.RandomHorizontalFlip())
    if strength == "strong":
        ops.append(transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8))
        ops.append(transforms.RandomGrayscale(p=0.2))
        # A small kernel: at 32x32 blur has limited effect, kept for recipe completeness
        # and so Step 6's ablation can toggle it off.
        ops.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)))
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(CIFAR_MEAN, CIFAR_STD))
    return transforms.Compose(ops)


def eval_transform() -> transforms.Compose:
    """Deterministic transform for probe/kNN embedding: no augmentation."""
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    )


class TwoViewTransform:
    """Wraps a base transform; each call returns two independently sampled views.

    This object has no notion of "SimCLR" vs "SimSiam" -- it is purely "give me two
    views," and every method's data loader uses it the same way.
    """

    def __init__(self, base_transform: transforms.Compose) -> None:
        self.base_transform = base_transform

    def __call__(self, img):
        return self.base_transform(img), self.base_transform(img)
