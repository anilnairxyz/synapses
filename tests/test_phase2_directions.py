"""E4 Part A pieces on synthetic states: a readout finds a planted direction; raw-space weights."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/epic2/phase2_confidence_gap"))
pytest.importorskip("inspect_ai")
import directions as e4  # noqa: E402


def test_readout_recovers_planted_direction():
    rng = np.random.default_rng(0)
    n, d = 300, 64
    scale = rng.uniform(0.5, 5, size=d)  # anisotropic features, as in real states
    X = rng.normal(size=(n, d)) * scale
    w = rng.normal(size=d)
    w /= np.linalg.norm(w)
    y = X @ w + rng.normal(scale=0.5, size=n)
    groups = np.repeat(np.arange(n // 3), 3)
    oof, w_hat = e4.readout(X, y, groups)
    assert np.corrcoef(oof, y)[0, 1] > 0.8  # out of fold, still readable
    assert e4.cosine(w_hat, w) > 0.8  # the direction is found, in the raw space
    assert abs(np.linalg.norm(w_hat) - 1) < 1e-9


def test_boot_questions_keeps_questions_whole():
    qids = np.array(["a", "a", "b", "b", "b", "c"])
    for idx in e4.boot_questions(qids, 0)[:20]:
        picked = qids[idx]
        for q in set(picked):
            assert (picked == q).sum() % (qids == q).sum() == 0
