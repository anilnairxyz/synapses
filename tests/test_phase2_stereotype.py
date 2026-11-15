"""E1 statistics: within-stratum Spearman ignores between-stratum trends; the null centres on 0."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/epic2/phase2_confidence_gap"))
pytest.importorskip("sklearn")
import analyze_stereotype as e1  # noqa: E402


def test_between_strata_trend_ignored():
    # x and y both rise with the stratum, but inside each stratum they are opposed.
    strata = np.repeat([0, 1, 2], 10)
    x = strata * 10 + np.tile(np.arange(10), 3)
    y = strata * 10 - np.tile(np.arange(10), 3)
    assert e1.within_spearman(x, y, strata) == pytest.approx(-1.0)
    assert e1.within_spearman(x, x, strata) == pytest.approx(1.0)


def test_ties_and_constant_strata():
    strata = np.repeat([0, 1], 6)
    conf = np.array([0.9] * 6 + [0.5, 0.5, 0.9, 0.9, 0.95, 0.95])
    assert e1.varying_strata(conf, strata).tolist() == [False] * 6 + [True] * 6
    r = e1.centred_ranks(conf, strata)
    assert np.allclose(r[:6], 0) and r[6] == r[7] < 0 < r[10] == r[11]


def test_null_centres_on_zero():
    rng = np.random.default_rng(0)
    strata = np.repeat(np.arange(10), 30)
    x = rng.normal(size=300)
    vals = [e1.within_spearman(rng.permutation(np.repeat([0.5, 0.8, 0.9], 100)), x, strata)
            for _ in range(300)]
    assert abs(np.mean(vals)) < 0.02


def test_win_rate_and_within_auroc():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/epic2/phase2_confidence_gap"))
    import analyze_swaps as an

    c = np.array([0.9, 0.5, 0.7, np.nan])
    w = np.array([0.1, 0.5, 0.8, 0.2])
    assert an.win_rate(c, w) == pytest.approx((1 + 0.5 + 0) / 3)  # tie half, NaN pair dropped
    assert an.win_rate(np.ones(5), np.ones(5)) == 0.5  # all tied: no separation
    # within strata: a score that only tracks the stratum separates nothing
    s = np.array([0.1, 0.1, 0.9, 0.9])
    y = np.array([1, 0, 1, 0])
    assert an.within_auroc(s, y, np.array(["a", "a", "b", "b"])) == 0.5
