"""Device selection that works on both machines: CUDA on the PC (RTX 4090),
MPS on the Mac (M3 Pro), CPU as a fallback."""

from __future__ import annotations

import platform

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Pick the best available device.

    Order: explicit ``prefer`` > CUDA > MPS (Apple Silicon) > CPU.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_report() -> str:
    """A human-readable summary of the compute environment."""
    lines = [
        f"Python      : {platform.python_version()}",
        f"Platform    : {platform.system()} {platform.machine()}",
        f"PyTorch     : {torch.__version__}",
        f"CUDA build  : {torch.version.cuda or 'n/a'}",
        f"CUDA avail  : {torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        lines.append(f"GPU         : {torch.cuda.get_device_name(0)}")
    lines.append(f"MPS avail   : {torch.backends.mps.is_available()}")
    lines.append(f"Selected    : {get_device()}")
    return "\n".join(lines)
