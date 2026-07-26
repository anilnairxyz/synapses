"""Experiment plumbing: seeding, run directories, JSONL metrics, checkpoints.

A run lives in one directory::

    outputs/phase1_L2_beta1.0_s0/
      config.json          the exact config that produced the run
      metrics.jsonl        one JSON object per epoch
      eval_batch.pt        frozen held-out test images, for Step 7 comparability
      ckpt/epoch_0001.pt   log-spaced checkpoints

Checkpoints carry model + optimizer + epoch + RNG state + config, so a run is
reproducible from any point. ``outputs/`` is gitignored — these stay on the PC.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU + all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict[str, Any]:
    """Capture RNG state for every generator we seed, for exact resumption."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def log_spaced_epochs(n_epochs: int) -> set[int]:
    """``{1, 2, 4, 8, 16, ...}`` plus the final epoch.

    Feature formation is fast early and slow late, so uniform checkpoint spacing wastes
    disk on the boring end. Step 7 (feature emergence across training) consumes these.
    """
    epochs = set()
    e = 1
    while e <= n_epochs:
        epochs.add(e)
        e *= 2
    epochs.add(n_epochs)
    return epochs


def list_checkpoints(run_dir: str | Path) -> list[tuple[int, Path]]:
    """Enumerate a run's saved checkpoints as ``(epoch, path)``, sorted by epoch.

    Looks for ``<run_dir>/ckpt/epoch_*.pt``. This is the checkpoint series the
    longitudinal feature-emergence analysis (Step 7) consumes.
    """
    ckpt_dir = Path(run_dir) / "ckpt"
    out: list[tuple[int, Path]] = []
    for p in ckpt_dir.glob("epoch_*.pt"):
        try:
            epoch = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        out.append((epoch, p))
    return sorted(out, key=lambda t: t[0])


def _to_plain(obj: Any) -> Any:
    """Make configs and metrics JSON-serializable."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_plain(v) for k, v in asdict(obj).items()}
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def make_run_dir(root: str | Path, run_id: str) -> Path:
    """Create ``<root>/<run_id>/ckpt/`` and return the run directory."""
    run_dir = Path(root) / run_id
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    return run_dir


class RunLogger:
    """Writes ``config.json`` once, then appends one JSON line per :meth:`log` call.

    ``metrics.jsonl`` is *truncated* on construction. A re-run of the same ``run_id``
    replaces its metrics rather than appending to the previous run's — otherwise the
    Step 5/6 sweep plots would silently read two interleaved runs as one.
    """

    def __init__(self, run_dir: str | Path, config: Any) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(json.dumps(_to_plain(config), indent=2))
        self._path = self.run_dir / "metrics.jsonl"
        self._fh = self._path.open("w")

    def log(self, **metrics: Any) -> None:
        self._fh.write(json.dumps(_to_plain(metrics)) + "\n")
        self._fh.flush()  # survive an interrupted run

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: Any,
) -> None:
    """Save model + optimizer + epoch + RNG state + config."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": _to_plain(config),
            "rng_state": rng_state(),
        },
        Path(path),
    )


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint, optionally restoring ``model`` and ``optimizer`` in place."""
    # weights_only=False: our checkpoints hold the config dict and RNG state, not just
    # tensors, and torch >= 2.6 defaults this to True.
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
    if model is not None:
        model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
