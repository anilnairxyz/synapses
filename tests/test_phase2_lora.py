"""E5 data preparation (no model): balanced inputs shared by all variants, targets per variant."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/epic2/phase2_confidence_gap"))
pytest.importorskip("inspect_ai")
import lora_confidence as lc  # noqa: E402


@pytest.fixture(scope="module")
def bal():
    return lc.training_set("qwen3-1.7b", None)


def test_balanced(bal):
    assert bal.correct.mean() == pytest.approx(0.5, abs=1e-9)
    assert set(bal.split) == {"train"} and set(bal.source) == {"own"}


def test_targets(bal):
    rows, reps = lc.targets(bal, "per_answer")
    assert all(r == lc.reply(str(c)) for r, c in zip(reps, rows.correct, strict=True))
    rows, reps = lc.targets(bal, "lin")
    assert all(r.startswith('{"confidence": ') and r[-4:-1].replace(".", "").isdigit() for r in reps)
    assert len(set(reps)) > 3  # graded
    rows, reps = lc.targets(bal, "shuffled")
    assert rows.shuffled.mean() == pytest.approx(rows.correct.mean())
    assert (rows.shuffled != rows.correct).mean() > 0.3
    rows, _ = lc.targets(bal, "loko_recall")
    assert "recall" not in set(rows.category)


def test_candidates_parse():
    import swap_task as st

    for c in lc.CANDIDATES:
        assert st.parse_confidence(lc.reply(c)) == float(c)


def test_test_set_has_all_sources():
    t = lc.test_set("qwen3-1.7b", None)
    assert set(t.source) == {"own", "correct", "wrong"} and set(t.split) == {"test"}
