"""MNIST data pipeline for Phase 1.

Binarization is asymmetric on purpose:

*   **Train — dynamic.** ``x ~ Bernoulli(pixel)`` resampled every epoch (Burda et al.),
    which makes the Bernoulli likelihood exact and acts as mild regularization. The
    caller applies :func:`binarize` per batch; ``train_loader`` yields raw ``[0, 1]``
    pixels.
*   **Test — fixed.** Binarized once with a dedicated seed and cached. Resampling the
    test set each epoch would inject sampling noise into the eval metric, making the
    epoch-to-epoch curve jittery and the Step 5/6 cross-run comparisons noisier than
    the effects being measured.

Nothing here downloads anything: ``download=False`` is the default and missing data
raises with the exact command to run.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST

DOWNLOAD_HINT = (
    "MNIST not found under {root!r}. Fetch it once with:\n\n"
    "    uv run python -c \"from torchvision.datasets import MNIST; \"\\\n"
    "        \"MNIST('{root}', train=True, download=True); \"\\\n"
    "        \"MNIST('{root}', train=False, download=True)\"\n"
)


def binarize(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample ``x ~ Bernoulli(pixel)``, turning ``[0, 1]`` greyscale into ``{0, 1}``."""
    return torch.bernoulli(x, generator=generator)


def _load_split(root: str, train: bool, download: bool) -> MNIST:
    try:
        return MNIST(root, train=train, download=download, transform=transforms.ToTensor())
    except RuntimeError as exc:  # torchvision raises this when the raw files are absent
        raise RuntimeError(DOWNLOAD_HINT.format(root=root)) from exc


def mnist_loaders(
    root: str = "data",
    batch_size: int = 128,
    num_workers: int = 2,
    download: bool = False,
    seed: int = 0,
    test_binarize_seed: int = 1234,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build the Phase 1 MNIST loaders.

    Returns ``(train_loader, test_loader)`` where:

    *   ``train_loader`` yields **continuous** ``[0, 1]`` images of shape ``(B, 1, 28, 28)``.
        Apply :func:`binarize` to each batch inside the training loop.
    *   ``test_loader`` yields **already-binarized** flat images of shape ``(B, 784)``,
        fixed for the lifetime of the process.

    Set ``download=True`` only if you want this call to fetch the dataset; the default
    refuses and tells you what to run instead.
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_set = _load_split(root, train=True, download=download)
    test_set = _load_split(root, train=False, download=download)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )

    # Fixed binarization of the whole test split, done once. 10k x 784 floats is ~31 MB.
    test_x = torch.stack([img for img, _ in test_set]).view(len(test_set), -1)
    test_y = torch.tensor([label for _, label in test_set])
    fixed_gen = torch.Generator().manual_seed(test_binarize_seed)
    test_x = binarize(test_x, generator=fixed_gen)

    test_loader = DataLoader(
        TensorDataset(test_x, test_y),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # already in memory
        pin_memory=pin_memory,
    )
    return train_loader, test_loader
