"""Row logic of the Step 3 loader (analyze.sample_rows) on hand-made samples."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/epic2/phase1_confidence"))
import analyze  # noqa: E402


def sample(q, calls, labels, value, fb=None):
    meta = {"labels": labels, "fb": fb or []}
    return SimpleNamespace(id=q["id"], metadata={"q": q, "calls": calls},
                           scores={"confidence_scorer": SimpleNamespace(value=value, metadata=meta)})


def by_method(rows):
    return {r["method"]: r for r in rows}


def test_answerable_rows():
    q = {"id": "rec-1", "category": "recall", "level": 3, "meta": {}}
    calls = {"P": {"answer": "X", "lp": {"sum": -0.5, "mean": -0.1, "min": -0.4}},
             "A": {"answer": "Y", "confidence": 0.9, "text": '```json\n{"answer": "Y"}```'},
             "D": {"answer": "X", "confidence": 0.6, "text": '6 x 7 = 42. {"confidence": 0.6}'},
             "C": {t: {"answer": a} for t, a in zip(analyze.THRESHOLDS, ["X", "X", "IDK", "IDK"])}}
    labels = {"P": "CORRECT", "A": "INCORRECT", "D": "CORRECT", "C0.0": "CORRECT",
              "C0.5": "CORRECT", "C0.75": "ABSTAINS", "C0.9": "ABSTAINS"}
    r = by_method(analyze.sample_rows("x", "m", sample(q, calls, labels, {"b_agree": 0.7})))
    assert r["P"]["correct"] == 1 and r["P"]["abstained"] == 0 and np.isnan(r["P"]["conf"])
    assert r["A"]["correct"] == 0 and r["A"]["conf"] == 0.9  # A is scored on its own answer
    assert r["B"]["correct"] == 1 and r["B"]["conf"] == 0.7  # B and E on P's answer
    assert r["E"]["conf"] == -0.5 and r["E_min"]["conf"] == -0.4
    assert r["A"]["preamble"] is False and r["D"]["preamble"] is True  # fences don't count
    assert r["C"]["conf"] == 2 and r["C"]["correct"] == 1  # answers at t=0 and .5; scored at t=0
    assert r["C0.9"]["abstained"] == 1 and r["C0.9"]["correct"] == 0


def test_missing_label_and_behaviour_kind():
    q = {"id": "ill-1", "category": "ill_posed", "level": 1, "meta": {"type": "false_premise"}}
    calls = {"P": {"answer": "never"}, "A": {"answer": "1999", "confidence": 0.8},
             "C": {t: {"answer": "never"} for t in analyze.THRESHOLDS}}
    labels = {"P": "CORRECTS_PREMISE", "A": "ANSWERS", "C0.0": "CORRECTS_PREMISE",
              "C0.5": "UNPARSED", "C0.75": "ABSTAINS", "C0.9": "ABSTAINS"}
    r = by_method(analyze.sample_rows("x", "m", sample(q, calls, labels, {})))
    assert r["P"]["kind"] == "false_premise" and r["P"]["correct"] == 1
    assert r["A"]["correct"] == 0 and r["A"]["abstained"] == 0
    assert np.isnan(r["C0.5"]["correct"]) and np.isnan(r["C"]["conf"])  # a gap: no implied value
    assert "B" not in r and "E" not in r  # no samples, no logprobs


def test_forecast_rows():
    q = {"id": "fc-1", "category": "forecastable", "level": 1, "meta": {"outcome": 1}}
    r = by_method(analyze.sample_rows("x", "m", sample(q, {"F": {"confidence": 0.3}}, {}, {},
                                                       fb=[0.2, None, 0.4])))
    assert r["F"]["conf"] == 0.3 and r["F"]["correct"] == 1
    assert abs(r["FB"]["conf"] - 0.3) < 1e-9


def test_regression_recovers_category_slopes():
    """Confidence predicts correctness within each cell for recall, not for perception."""
    import pandas as pd

    rng = np.random.default_rng(0)
    rows = []
    for config in ("m1", "m2"):
        for kind in analyze.ANSWERABLE:
            for level in (1, 2, 3):
                for i in range(80):
                    conf = rng.random()
                    base = {1: 0.8, 2: 0.5, 3: 0.2}[level]
                    p = base + (0.4 * (conf - 0.5) if kind == "recall" else 0.0)
                    rows.append({"config": config, "qid": f"{kind}-{level}-{i}", "kind": kind,
                                 "level": level, "method": "A", "abstained": 0.0, "conf": conf,
                                 "correct": float(rng.random() < p)})
    res = analyze.regression(pd.DataFrame(rows), methods=("A",), n_boot=50).set_index("term")
    assert res.loc["recall", "lo"] > 0  # informative
    assert res.loc["perception", "lo"] < 0 < res.loc["perception", "hi"]  # no signal
    assert res.loc["perception - recall", "hi"] < 0


def test_hard_easy_sensitivity():
    """Flat confidence: sensitivity 0 and overconfidence grows with difficulty; confidence that
    equals each level's accuracy: sensitivity 1 and no overconfidence."""
    import pandas as pd

    rows = []
    accs = {1: 1.0, 2: 0.75, 3: 0.5, 4: 0.25, 5: 0.0}
    for config, flat in (("flat", True), ("tracks", False)):
        for level, acc in accs.items():
            for i in range(20):
                rows.append({"config": config, "kind": "computation", "level": level,
                             "method": "A", "abstained": 0.0, "correct": float(i < 20 * acc),
                             "conf": 0.95 if flat else acc})
    _, s = analyze.hard_easy(pd.DataFrame(rows))
    s = s.set_index("config")
    assert abs(s.loc["flat", "sensitivity"]) < 1e-9 and abs(s.loc["tracks", "sensitivity"] - 1) < 1e-9
    assert abs(s.loc["flat", "over_hardest"] - 0.95) < 1e-9 and abs(s.loc["tracks", "over_hardest"]) < 1e-9


def _rows(**cols):
    import pandas as pd

    n = max(len(v) for v in cols.values() if isinstance(v, list))
    return pd.DataFrame({k: v if isinstance(v, list) else [v] * n for k, v in cols.items()})


def test_d_vs_a_drops_preamble_pairs():
    import pandas as pd

    rows = []
    for i in range(30):
        right = float(i % 2)
        for m, conf in (("A", 0.9), ("D", 0.2 + 0.6 * right)):  # D informative, A flat
            rows.append({"config": "m", "qid": f"q{i}", "kind": "recall", "level": 1,
                         "method": m, "conf": conf, "correct": right, "abstained": 0.0,
                         "preamble": m == "A" and i < 4})
    r = analyze.d_vs_a(pd.DataFrame(rows), n_boot=50).iloc[0]
    assert r.n == 26 and r.dropped_preamble == 4
    assert r.auroc_within_A == 0.5 and r.auroc_within_D == 1.0 and r.diff_lo > 0


def test_kalai_penalised_score():
    # At t = 0.75 (penalty 3): one right answer, one wrong, two abstentions -> (1 - 3) / 4.
    rows = _rows(config="m", qid=["a", "b", "c", "d"] * 2, kind="recall", level=1,
                 method=["C0.75"] * 4 + ["C0.0"] * 4,
                 correct=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                 abstained=[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    k = analyze.kalai(rows).set_index("t")
    assert k.loc[0.75, "score"] == -0.5 and k.loc[0.75, "answer_rate"] == 0.5
    assert k.loc[0.75, "score_t0_answers"] == (1 - 3 - 3 + 1) / 4  # answering everything


def test_forecast_baselines():
    rows = _rows(config="m", qid=["a", "b", "c", "d"], kind="forecastable", level=1,
                 method="F", conf=[0.9, 0.1, 0.9, 0.1], correct=[1.0, 0.0, 1.0, 0.0],
                 crowd=[0.5, 0.5, 0.5, 0.5], spread=np.nan)
    f = analyze.forecasts(rows).set_index("config")
    assert abs(f.loc["m", "brier"] - 0.01) < 1e-9
    assert f.loc["crowd", "brier"] == 0.25 and f.loc["base rate", "brier"] == 0.25
