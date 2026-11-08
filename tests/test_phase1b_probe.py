"""Phase 1B probe pipeline on synthetic states (no model, no logs)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/epic2/phase1_confidence"))
import probe  # noqa: E402

KINDS = ("recall", "computation", "perception")
META = {"recall": {"log10_pop": 3.0, "relation": "director"},
        "computation": {"a": 12, "b": 34, "digits": [2, 2]},
        "perception": {"length": 6, "zipf": 4.0, "family": "count"}}


def fake_split(rng, n_per_kind, n_layers=4, hidden=12, signal_layer=2, behaviour=0):
    """States where, at ``signal_layer`` and position 'gen', feature 0 carries the label."""
    rows, states = [], []
    cats = [k for k in KINDS for _ in range(n_per_kind)] + ["unknowable"] * behaviour
    for i, cat in enumerate(cats):
        level = int(rng.integers(1, 6)) if cat in KINDS else None
        y = int(rng.random() < 0.4) if cat in KINDS else 0
        x = rng.normal(size=(n_layers, 4, hidden))
        x[signal_layer, 1, 0] += 2.5 * (2 * y - 1)
        states.append(x)
        rows.append({"qid": f"{cat}-{i}", "category": cat, "level": level, "answer": "a",
                     "n_prompt": 10, "n_answer": 2, "y": y, "abstained": 0, "label": "",
                     "meta": META.get(cat, {})})
    return np.array(states, dtype=np.float16), pd.DataFrame(rows)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    rng = np.random.default_rng(0)
    splits = {"train": fake_split(rng, 150), "test": fake_split(rng, 60, behaviour=10)}

    def load(config, split):
        return splits[split]

    monkeypatch.setattr(probe, "load", load)
    return splits


def test_metadata_features():
    rows = pd.DataFrame({"meta": [META["computation"]] * 3})
    f = probe.metadata_features(rows, "computation")
    assert list(f.columns) == ["da", "db", "log_a", "log_b"] and len(f) == 3


def test_run_config_finds_the_signal(fake):
    layers, scores = probe.run_config("fake", mlp=False, n_jobs=1)
    gen = layers[(layers.position == "gen") & (layers.train_kind == "pooled")]
    assert gen.set_index("layer").val_within.idxmax() == 2
    lr = scores[(scores.probe == "logreg") & (scores.position == "gen")]
    assert set(lr.layer) == {2}
    from synapses.calibration import auroc
    for kind in ("recall", "pooled"):
        s = lr[(lr.train_kind == kind) & lr.category.isin(KINDS)]
        assert auroc(s.score, s.y) > 0.85
    sh = scores[(scores.probe == "shuffled") & (scores.position == "gen")
                & (scores.train_kind == "pooled") & scores.category.isin(KINDS)]
    assert auroc(sh.score, sh.y) < 0.75
    # Every probe scores every test question, behaviour items included (cross-kind / abstention).
    assert lr.groupby("train_kind").size().eq(190).all()
    meta = scores[scores.probe == "meta"]
    assert meta.groupby("train_kind").score.count().eq(60).all()


def test_oof_scores_every_training_question(fake):
    _, scores = probe.run_config("fake", mlp=False, n_jobs=1)
    chosen = (scores[scores.probe == "logreg"].drop_duplicates(["train_kind", "position"])
              [["train_kind", "position", "layer", "C"]])
    oof = probe.run_oof("fake", chosen)
    from synapses.calibration import auroc
    g = oof[(oof.probe == "logreg") & (oof.position == "gen") & (oof.train_kind == "pooled")]
    assert len(g) == 450 and auroc(g.score, g.y) > 0.85
    assert oof[oof.probe == "meta"].groupby("train_kind").size().eq(150).all()
