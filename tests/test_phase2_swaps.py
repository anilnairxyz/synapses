"""E2 wrong-answer builders and formatting: always wrong, never the own answer, plausible."""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/epic2/phase2_confidence_gap"))
pytest.importorskip("inspect_ai")
import build_swaps as bs  # noqa: E402
import phase1  # noqa: E402


def q_count(word="parry", letter="r", gold="2"):
    return {"category": "perception", "answer_type": "integer", "gold_answers": [gold],
            "question": f'How many times does the letter \'{letter}\' appear in the word "{word}"?',
            "meta": {"family": "count", "word": word}}


def q_nth(word="tumor", n=4):
    return {"category": "perception", "answer_type": "letter", "gold_answers": [word[n - 1]],
            "question": f'What is the {n}th letter of the word "{word}"?', "meta": {"family": "nth", "word": word}}


def q_rev(word="spray"):
    return {"category": "perception", "answer_type": "exact_string", "gold_answers": [word[::-1]],
            "question": f'Spell the word "{word}" backwards.', "meta": {"family": "reverse", "word": word}}


def q_comp(a=385, b=774):
    return {"category": "computation", "answer_type": "integer", "gold_answers": [str(a * b)],
            "question": f"What is {a} × {b}?", "meta": {"a": a, "b": b}}


@pytest.mark.parametrize("seed", range(50))
def test_wrong_computation(seed):
    gold = str(385 * 774)
    own = str(385 * 774 + 1000)
    w = bs.wrong_computation(gold, own, random.Random(seed))
    assert w != gold and w != own and len(w) == len(gold)
    assert w[0] == gold[0] and w[-1] == gold[-1]  # a middle digit changed
    assert sum(x != y for x, y in zip(w, gold, strict=True)) == 1


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("make", [q_count, q_nth, q_rev])
def test_wrong_perception(seed, make):
    q = make()
    gold = q["gold_answers"][0]
    w = bs.wrong_perception(q, None, random.Random(seed))
    assert w != gold
    assert phase1.ct.grade_fixed(q, w) == "INCORRECT"
    if q["meta"]["family"] == "reverse":
        assert sorted(w) == sorted(gold)  # same letters, two swapped
    if q["meta"]["family"] == "nth":
        assert w in q["meta"]["word"]


def test_wrong_perception_avoids_own():
    q = q_count(gold="2")
    for seed in range(30):
        assert bs.wrong_perception(q, "1", random.Random(seed)) == "3"


def test_wrong_recall_prefers_other_models_names():
    q = {"gold_answers": ["George Lucas"], "category": "recall"}
    others = ["More information is needed.", "The director of Star Wars was Steven Spielberg.",
              "George Lucas", "I don't know"]
    v, origin = bs.wrong_recall(q, "Ridley Scott", others, ["Peter Jackson"], random.Random(0))
    assert (v, origin) == ("Steven Spielberg", "other_model")
    v, origin = bs.wrong_recall(q, None, ["George Lucas."], ["Peter Jackson"], random.Random(0))
    assert (v, origin) == ("Peter Jackson", "other_question")


def test_render_keeps_the_models_format():
    q = q_nth()
    assert bs.render(q, 'The 4th letter of the word "tumor" is "r".', "o", False) == (
        'The 4th letter of the word "tumor" is "o".', "substituted")
    assert bs.render(q, "R", "o", False) == ("O", "bare")  # case follows the own answer
    qc = q_count()
    assert bs.render(qc, "The letter 'r' appears 2 times in \"parry\".", "3", False)[0] == (
        "The letter 'r' appears 3 times in \"parry\".")
    assert bs.render(qc, "Once.", "2", False)[0] == "Twice."
    comp = q_comp(58, 78)
    assert bs.render(comp, "58 × 78 = 4524.", "4534", False) == ("58 × 78 = 4534.", "substituted")
    assert bs.render(comp, "33,635", "4534", True) == ("4,534", "bare")
    rec = {"category": "recall", "answer_type": "short_text", "meta": {}}
    assert bs.render(rec, "The director of Escape to Victory was John Ford.", "John Huston", False) == (
        "The director of Escape to Victory was John Huston.", "substituted")
    assert bs.render(rec, "Ian Fleming.", "Roald Dahl", False) == ("Roald Dahl.", "bare")


def test_render_falls_back_when_too_long():
    assert bs.render(q_comp(), "x" * 500, "123", False) == ("123", "bare")


def test_ptrue_shots_and_block():
    pytest.importorskip("sklearn")
    import extract_answers as ea
    import pandas as pd

    rows = pd.DataFrame([{"split": "train", "source": "own", "qid": f"{k}-{i}", "category": k,
                          "correct": i % 2, "text": f"ans{i}"}
                         for k in ("recall", "computation", "perception") for i in range(20)])
    qs = {r: {"question": f"Q {r}?"} for r in rows.qid}
    shots = ea.choose_shots(rows, qs)
    assert len(shots) == ea.K_SHOTS and sum(s["ok"] for s in shots) == ea.K_SHOTS // 2
    assert len({s["qid"] for s in shots}) == ea.K_SHOTS
    assert shots == ea.choose_shots(rows, qs)  # deterministic
    text = ea.shots_text(shots)
    assert text.count("The proposed answer is: (A)") == ea.K_SHOTS // 2
    assert text.count("The proposed answer is: (B)") == ea.K_SHOTS // 2
    b = ea.block("What is 2 × 3?", "6")
    assert b.startswith("Question: What is 2 × 3?\nProposed Answer: 6\nIs the proposed answer:")
    assert b.endswith("(A) True\n(B) False\nThe proposed answer is:")


def test_parse_confidence():
    import swap_task as st

    assert st.parse_confidence('{"confidence": 0.85}') == 0.85
    assert st.parse_confidence("0.9") == 0.9
    assert st.parse_confidence("90%") == 0.9
    assert st.parse_confidence("1.") == 1.0
    assert st.parse_confidence("The answer is 4524, I am sure.") is None
