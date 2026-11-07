import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/epic2/phase1_confidence"))
from questions import (  # noqa: E402
    computation,
    forecastable,
    ill_posed,
    perception,
    recall,
    unknowable,
)
from questions.schema import Question  # noqa: E402


def test_schema_validation():
    with pytest.raises(ValueError):
        Question("x", "recall", "answerable", 1, "s", "q?", "short_text", [])
    with pytest.raises(ValueError):
        Question("x", "unknowable", "behaviour", None, "s", "q?", "none", [])
    with pytest.raises(ValueError):
        Question("x", "recall", "answerable", 6, "s", "q?", "short_text", ["a"])


def test_computation_gold_digits_and_determinism():
    qs = computation.build(per_level=10, seed=0)
    assert len(qs) == 50
    for q in qs:
        a, b = q.meta["a"], q.meta["b"]
        assert q.gold_answers == [str(a * b)]
        assert [len(str(a)), len(str(b))] == q.meta["digits"] == list(computation.LEVELS[q.level])
        assert a % 10 and b % 10
    assert [q.question for q in qs] == [q.question for q in computation.build(10, 0)]
    assert [q.question for q in qs] != [q.question for q in computation.build(10, 1)]


def test_perception_gold_lengths_and_balance():
    qs = perception.build(per_level=9, seed=0)
    assert len(qs) == 45
    for q in qs:
        w, fam, gold = q.meta["word"], q.meta["family"], q.gold_answers[0]
        lo, hi = perception.LEVELS[q.level]
        assert lo <= len(w) <= hi
        if fam == "reverse":
            assert gold == w[::-1]
        elif fam == "count":
            letter = q.question.split("'")[1]
            assert int(gold) == w.count(letter) == max(w.count(c) for c in set(w))
        else:
            n = int("".join(ch for ch in q.question.split("the ")[1] if ch.isdigit()))
            assert gold == w[n - 1] and n >= len(w) / 3
    for lvl in perception.LEVELS:
        fams = [q.meta["family"] for q in qs if q.level == lvl]
        assert fams.count("count") == fams.count("nth") == fams.count("reverse") == 3
    with pytest.raises(ValueError):
        perception.build(per_level=10, seed=0)


def test_recall_helpers():
    assert [recall.level_of(n) for n in (150_000, 100_000, 99_999, 1_000, 100, 99, 2)] == [
        1, 1, 2, 3, 4, 5, 5,
    ]
    assert recall._given_away("Gordes", "Canton of Gordes")
    assert recall._given_away("Kambarka", "Kambarsky District")
    assert not recall._given_away("Karel Zeman", "Ludmila Zeman")
    assert recall._leaks("Piano Concerto (Tan Dun)", "Piano Concerto", ["Tan Dun"])
    assert not recall._leaks("The List (2012 film)", "The List", ["Sean Hogan"])


def test_recall_build():
    pytest.importorskip("datasets")
    qs = recall.build(per_level=10, seed=0)  # downloads PopQA on first use
    assert len(qs) == 50
    assert len({q.meta["subject"] for q in qs}) == 50
    for q in qs:
        assert recall.level_of(q.meta["s_pop"]) == q.level
        assert q.gold_answers and q.answer_type == "short_text"
        assert not any(recall._given_away(a, q.meta["subject"]) for a in q.gold_answers)
    for lvl in range(1, 6):
        rels = [q.meta["relation"] for q in qs if q.level == lvl]
        assert all(rels.count(r) == 2 for r in recall.RELATIONS)
    assert [q.question for q in qs] == [q.question for q in recall.build(per_level=10, seed=0)]
    with pytest.raises(ValueError):
        recall.build(per_level=12, seed=0)


def test_ill_posed_build():
    pytest.importorskip("datasets")
    qs = ill_posed.build(seed=0, per_level=4, n_ambiguous=8)
    fp = [q for q in qs if q.meta["type"] == "false_premise"]
    amb = [q for q in qs if q.meta["type"] == "ambiguous"]
    assert len(fp) == 20 and len(amb) == 8
    assert {q.level for q in fp} == {1, 2, 3, 4, 5}
    for q in fp:
        assert q.right_behaviour == "correct_premise" and q.kind == "behaviour"
        assert q.meta["false_person"] in q.question and q.meta["subject"] in q.question
        assert q.meta["false_person"].lower() != q.meta["true_answer"].lower()
    for q in amb:
        assert q.right_behaviour == "clarify" and q.level is None
        r = q.meta["readings"]
        assert len({x["title"] for x in r}) == len(r) >= 2
        assert not {a.lower() for a in r[0]["answers"]} & {a.lower() for a in r[1]["answers"]}
        assert q.meta["pop_ratio"] <= ill_posed.MAX_RATIO
    assert [q.question for q in qs] == [
        q.question for q in ill_posed.build(seed=0, per_level=4, n_ambiguous=8)
    ]


def test_unknowable_build():
    qs = unknowable.build(seed=0)
    assert len(qs) == 75 and len({q.id for q in qs}) == 75
    assert all(q.right_behaviour == "abstain" and q.level is None for q in qs)
    lifespans = {name: (born, died) for name, born, died in unknowable.PEOPLE}
    for q in qs:
        if q.meta["type"] == "unrecorded":
            born, died = lifespans[q.meta["person"]]
            assert born + 25 <= int(q.meta["date"].split()[-1]) < died
        if q.meta["type"] == "far_future":
            assert 2070 <= q.meta["year"] <= 2153
            if "World Cup" in q.question:
                assert q.meta["year"] % 4 == 2
    assert [q.question for q in qs] == [q.question for q in unknowable.build(seed=0)]


def test_forecastable_family_groups_candidates():
    a = forecastable._family("Will Lula win the 2026 Brazilian presidential election?")
    b = forecastable._family("Will Renan Santos win the 2026 Brazilian presidential election?")
    assert a == b
    assert a != forecastable._family("Will Russia test a nuclear weapon by September 30 2026?")


def test_forecastable_build():
    try:
        qs = forecastable.build(seed=0)  # downloads ForecastBench files on first use
    except OSError:
        pytest.skip("ForecastBench files not cached and no network")
    assert len(qs) == sum(forecastable.SOURCE_QUOTA.values())
    lo, hi = forecastable.CROWD_RANGE
    for q in qs:
        m = q.meta
        assert q.right_behaviour == "probability" and m["outcome"] in (0, 1)
        assert lo <= m["crowd"] <= hi and m["resolved_on"] > m["as_of"]
        assert m["resolved_on"] >= forecastable.FREEZE_FROM
    assert len({forecastable._family(q.question) for q in qs}) == len(qs)
    assert [q.id for q in qs] == [q.id for q in forecastable.build(seed=0)]
