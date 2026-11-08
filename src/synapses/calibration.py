"""Calibration and discrimination metrics for confidence-elicitation evals (Epic 2).

All functions take ``p`` (stated confidence in [0, 1]) and ``y`` (1 = correct, 0 = not)
as 1-D array-likes of equal length. Pure numpy; no eval-framework dependency, so the
same code scores Inspect logs, hand-built runs, or later phases.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata


def _arrays(p, y, unit: bool = True) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if p.shape != y.shape or p.ndim != 1:
        raise ValueError(f"p and y must be 1-D and equal length, got {p.shape} vs {y.shape}")
    if unit and np.any((p < 0) | (p > 1)):
        raise ValueError("confidences must lie in [0, 1]")
    return p, y


def brier(p, y) -> float:
    """Mean squared error between confidence and outcome. 0 is perfect; 0.25 is p=0.5 always."""
    p, y = _arrays(p, y)
    return float(np.mean((p - y) ** 2))


@dataclass(frozen=True)
class Reliability:
    """Per-bin reliability table: bin edges, counts, mean confidence and accuracy per bin."""

    edges: np.ndarray
    count: np.ndarray
    confidence: np.ndarray  # NaN for empty bins
    accuracy: np.ndarray  # NaN for empty bins


def reliability(p, y, n_bins: int = 10) -> Reliability:
    """Equal-width bins on [0, 1]; the last bin is closed so p=1.0 lands in it."""
    p, y = _arrays(p, y)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    count = np.bincount(idx, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        conf = np.bincount(idx, weights=p, minlength=n_bins) / count
        acc = np.bincount(idx, weights=y, minlength=n_bins) / count
    return Reliability(edges, count, conf, acc)


def ece(p, y, n_bins: int = 10) -> float:
    """Expected calibration error: count-weighted mean |accuracy - confidence| over bins."""
    r = reliability(p, y, n_bins)
    m = r.count > 0
    return float(np.sum(r.count[m] * np.abs(r.accuracy[m] - r.confidence[m])) / r.count.sum())


@dataclass(frozen=True)
class MurphyDecomposition:
    """Brier = reliability - resolution + uncertainty (exact when confidences take bin values).

    reliability: calibration error (lower is better).
    resolution:  how much binned accuracy varies away from the base rate (higher is better;
                 this is the discrimination part).
    uncertainty: base-rate variance ybar * (1 - ybar); a property of the data, not the model.
    """

    reliability: float
    resolution: float
    uncertainty: float


def murphy(p, y, n_bins: int = 10) -> MurphyDecomposition:
    p, y = _arrays(p, y)
    r = reliability(p, y, n_bins)
    m = r.count > 0
    n = len(p)
    ybar = y.mean()
    rel = np.sum(r.count[m] * (r.confidence[m] - r.accuracy[m]) ** 2) / n
    res = np.sum(r.count[m] * (r.accuracy[m] - ybar) ** 2) / n
    return MurphyDecomposition(float(rel), float(res), float(ybar * (1 - ybar)))


def auroc(p, y) -> float:
    """Probability a random correct answer has higher confidence than a random incorrect one.

    Ties count half (Mann-Whitney U). Returns NaN if only one class is present.
    0.5 = confidence carries no information about correctness. Only the ranking of ``p``
    matters, so any score works (log-probabilities, counts), not just values in [0, 1].
    """
    p, y = _arrays(p, y, unit=False)
    n_pos = int((y == 1).sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(p)  # average ranks handle ties
    u = ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def bootstrap_ci(stat, p, y, n_boot: int = 2000, level: float = 0.95,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval for ``stat(p, y)``, resampling (p, y) pairs.

    Resamples where the statistic is undefined (e.g. AUROC with one class) are dropped.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(p), size=(n_boot, len(p)))
    vals = np.array([stat(p[i], y[i]) for i in idx])
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    a = (1 - level) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


def selective_curve(p, y) -> tuple[np.ndarray, np.ndarray]:
    """Answer only the most-confident fraction of questions.

    Returns (coverage, accuracy) arrays, sorted by decreasing confidence: entry k is the
    accuracy over the top-(k+1) most confident answers. Ties are broken stably.
    """
    p, y = _arrays(p, y)
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    k = np.arange(1, len(ys) + 1)
    return k / len(ys), np.cumsum(ys) / k


def penalised_score(p, y, t: float, abstained=None) -> float:
    """Kalai et al. (2025) threshold scoring, averaged per question.

    Answer if p >= t (and the model did not explicitly abstain): +1 if correct,
    -t/(1-t) if wrong; otherwise 0. Under this rule, answering is worthwhile exactly when
    true P(correct) > t, so a calibrated model that abstains below t maximises it.
    """
    if not 0 <= t < 1:
        raise ValueError("t must be in [0, 1)")
    p, y = _arrays(p, y)
    ab = np.zeros_like(p, dtype=bool) if abstained is None else np.asarray(abstained, dtype=bool)
    answer = (p >= t) & ~ab
    penalty = t / (1 - t)
    s = np.where(answer, np.where(y == 1, 1.0, -penalty), 0.0)
    return float(s.mean())
