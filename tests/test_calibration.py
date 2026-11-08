import numpy as np
import pytest

from synapses.calibration import (
    auroc,
    bootstrap_ci,
    brier,
    ece,
    murphy,
    penalised_score,
    reliability,
    selective_curve,
)


def test_brier_known_values():
    assert brier([1, 0], [1, 0]) == 0
    assert brier([0.5, 0.5], [1, 0]) == pytest.approx(0.25)
    assert brier([1, 1], [0, 0]) == 1


def test_input_validation():
    with pytest.raises(ValueError):
        brier([0.5], [1, 0])
    with pytest.raises(ValueError):
        brier([1.2], [1])


def test_reliability_bins_and_p_equal_one():
    r = reliability([0.05, 0.95, 1.0, 1.0], [0, 1, 1, 0], n_bins=10)
    assert r.count[0] == 1 and r.count[-1] == 3
    assert r.accuracy[-1] == pytest.approx(2 / 3)
    assert np.isnan(r.accuracy[5])


def test_ece_perfectly_calibrated_is_zero():
    # In each bin, accuracy equals the confidence exactly.
    p = [0.25] * 4 + [0.75] * 4
    y = [1, 0, 0, 0] + [1, 1, 1, 0]
    assert ece(p, y, n_bins=4) == pytest.approx(0)


def test_ece_overconfident():
    assert ece([0.9] * 10, [1] * 5 + [0] * 5) == pytest.approx(0.4)


def test_murphy_sums_to_brier_on_bin_valued_confidences():
    rng = np.random.default_rng(0)
    p = rng.choice([0.05, 0.35, 0.65, 0.95], size=500)
    y = (rng.random(500) < p).astype(int)
    d = murphy(p, y, n_bins=10)
    assert d.reliability - d.resolution + d.uncertainty == pytest.approx(brier(p, y))


def test_auroc_extremes_and_ties():
    assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1
    assert auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0
    assert auroc([0.5] * 4, [1, 0, 1, 0]) == pytest.approx(0.5)
    assert np.isnan(auroc([0.5, 0.6], [1, 1]))
    assert auroc([-0.1, -0.2, -3.0, -5.0], [1, 1, 0, 0]) == 1  # any score, e.g. logprobs
    assert auroc([4, 2, 2, 0], [1, 1, 0, 0]) == pytest.approx(0.875)  # counts, with a tie


def test_bootstrap_ci():
    rng = np.random.default_rng(0)
    p = rng.random(300)
    y = (rng.random(300) < p).astype(int)
    lo, hi = bootstrap_ci(auroc, p, y, n_boot=500)
    assert lo < auroc(p, y) < hi and hi - lo < 0.2
    assert all(np.isnan(bootstrap_ci(auroc, [0.5, 0.6], [1, 1], n_boot=50)))


def test_auroc_matches_sklearn():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(1)
    p = np.round(rng.random(200), 1)  # many ties
    y = (rng.random(200) < p).astype(int)
    assert auroc(p, y) == pytest.approx(roc_auc_score(y, p))


def test_selective_curve():
    cov, acc = selective_curve([0.9, 0.1, 0.5], [1, 0, 0])
    np.testing.assert_allclose(cov, [1 / 3, 2 / 3, 1])
    np.testing.assert_allclose(acc, [1, 0.5, 1 / 3])


def test_penalised_score():
    p = [0.9, 0.9, 0.3]
    y = [1, 0, 1]
    # t=0.5: penalty 1. Answers first two: +1 -1, abstains third -> 0/3
    assert penalised_score(p, y, 0.5) == pytest.approx(0)
    # t=0: plain accuracy
    assert penalised_score(p, y, 0.0) == pytest.approx(2 / 3)
    # explicit abstention overrides confidence
    assert penalised_score(p, y, 0.5, abstained=[False, True, False]) == pytest.approx(1 / 3)
