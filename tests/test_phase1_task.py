"""Tests for the Phase 1 Confidence Audit task (Step 2): parsing, grading, and an end-to-end
run on a scripted mock model, with and without the judge."""

import json
import sys
from pathlib import Path

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput, get_model
from inspect_ai.model._model_output import Logprob, Logprobs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/epic2/phase1_confidence"))
import confidence_task as ct  # noqa: E402


def q(**kw):
    base = {"id": "x", "category": "computation", "kind": "answerable", "level": 1,
            "source": "s", "question": "What is 85 × 98?", "answer_type": "integer",
            "gold_answers": ["8330"], "right_behaviour": None, "meta": {}}
    return base | kw


def test_parsing_helpers():
    assert ct.parse_json('Sure. {"answer": "x", "confidence": 0.7}') == {"answer": "x", "confidence": 0.7}
    assert ct.parse_json("no json") is None
    assert ct.parse_prob("85%") == 0.85 and ct.parse_prob(0.3) == 0.3
    assert ct.parse_prob("high") is None and ct.parse_prob(150) is None
    assert ct.strip_think("<think>hmm</think>\n391") == ("391", False)
    assert ct.strip_think("<think>still going") == ("", True)
    assert ct.looks_abstain("I don't know.") and not ct.looks_abstain("Rob Zombie")
    assert "loses 3 points" in ct.kalai_rule(0.75) and "loses 1 point," in ct.kalai_rule(0.5)


def test_fixed_grading():
    assert ct.grade_fixed(q(), "85 × 98 = 8,330") == "CORRECT"
    assert ct.grade_fixed(q(), "8331") == "INCORRECT"
    assert ct.grade_fixed(q(), "I don't know") == "ABSTAINS"
    assert ct.grade_fixed(q(), "8, 330") == "CORRECT"  # digit group split by a space
    count = q(category="perception", gold_answers=["2"])
    assert ct.grade_fixed(count, "twice") == "CORRECT"
    assert ct.grade_fixed(count, "Two times.") == "CORRECT"
    assert ct.grade_fixed(count, 'The letter \'i\' appears 2 times in "corinthians".') == "CORRECT"
    assert ct.grade_fixed(q(category="perception", gold_answers=["0"]),
                          "The letter 'i' does not appear in it.") == "CORRECT"
    letter = q(category="perception", answer_type="letter", gold_answers=["o"])
    assert ct.grade_fixed(letter, "The 4th letter is 'o'.") == "CORRECT"
    assert ct.grade_fixed(letter, "u") == "INCORRECT"
    rev = q(category="perception", answer_type="exact_string", gold_answers=["yarps"])
    assert ct.grade_fixed(rev, "Yarps") == "CORRECT"
    assert ct.grade_fixed(rev, '"spray" backwards is "yarps"') == "CORRECT"
    assert ct.grade_fixed(rev, "yrasp") == "INCORRECT"
    rec = q(category="recall", answer_type="short_text", gold_answers=["Rob Zombie", "Robert Cummings"])
    assert ct.grade_recall_exact(rec, "rob zombie.") == "CORRECT"
    assert ct.grade_recall_exact(rec, "Directed by Rob Zombie") is None  # left to the judge


def test_judge_label_parsing():
    labels = ("CORRECTS_PREMISE", "ANSWERS", "ABSTAINS")
    assert ct.first_label("ANSWERS", labels) == "ANSWERS"
    assert ct.first_label("ANSWERS (it never questions CORRECTS_PREMISE)", labels) == "ANSWERS"
    assert ct.first_label("no idea", labels) is None
    assert ct._BARE_YEAR.match("1885") and ct._BARE_YEAR.match("In 1885.")
    assert not ct._BARE_YEAR.match("Blake did not write it; Schwartz did in 2003.")


def test_refusal_rule():
    for a in ("Unknown", "Unclear", "Not applicable", "N/A", "None", "Not available",
              "The question cannot be answered as asked.", "The question cannot be answered.",
              "The question cannot be answered with the given information.", "Cannot determine",
              "Cannot be determined.", "Cannot be predicted", "I cannot answer",
              "I cannot confirm the year.", "Unable to confirm", "The question is incorrect.",
              "Question is unclear", "Unanswerable", "Not publicly known.", "It is impossible to determine"):
        assert ct.is_refusal(a), a
    for a in ("Never", "The question is ambiguous.", "Woody Allen did not direct it.",
              "The question cannot be answered as asked. Kafka did not write it.", "1885"):
        assert not ct.is_refusal(a), a
    fp = q(category="ill_posed", kind="behaviour", meta={"type": "false_premise"})
    assert ct.rule_label(fp, "false_premise", "Unknown") == "ABSTAINS"
    assert ct.rule_label(fp, "false_premise", "In 1885.") == "ANSWERS"
    assert ct.rule_label(fp, "false_premise", "Never") is None  # left to the judge
    rec = q(category="recall", gold_answers=["Christophe Beck"])
    assert ct.rule_label(rec, "recall", "Not applicable") == "ABSTAINS"
    assert ct.rule_label(rec, "recall", "Question is ambiguous") == "ABSTAINS"  # names no one
    amb = q(category="ill_posed", kind="behaviour", meta={"type": "ambiguous"})
    assert ct.rule_label(amb, "ambiguous", "The question is ambiguous.") is None  # judge: CLARIFIES


def test_spread_covers_levels():
    rows = [{"id": f"{lv}-{i}", "level": lv} for lv in (1, 2, 3, 4, 5) for i in range(30)]
    picked = ct._spread(rows, 10)
    assert len(picked) == 10 and {r["level"] for r in picked} == {1, 2, 3, 4, 5}
    assert ct._spread(rows[:3], 10) == sorted(rows[:3], key=lambda r: r["id"])


def test_answer_logprobs_skips_thinking():
    toks = [Logprob(token=t, logprob=lp) for t, lp in
            [("<think>", -0.1), ("x", -0.2), ("</think>", -0.1), ("\n", -0.5), ("3", -0.1), ("91", -0.3)]]
    out = ModelOutput(model="m", choices=[ChatCompletionChoice(
        message=ChatMessageAssistant(content="391"), logprobs=Logprobs(content=toks))])
    lp = ct.answer_logprobs(out)
    assert lp["n"] == 2 and lp["first"] == -0.1 and lp["sum"] == pytest.approx(-0.4)


# --------------------------------------------------------------------------- end to end

ITEMS = [
    q(id="comp-1"),
    q(id="rec-1", category="recall", answer_type="short_text", question="Who directed Halloween II?",
      gold_answers=["Rob Zombie"]),
    q(id="unk-1", category="unknowable", kind="behaviour", level=None, answer_type="none",
      question="How many steps did Newton take on 1 May 1700?", gold_answers=[],
      right_behaviour="abstain", meta={"type": "unrecorded"}),
    q(id="fc-1", category="forecastable", kind="behaviour", level=None, answer_type="probability",
      question="Will X happen?", gold_answers=[], right_behaviour="probability",
      meta={"outcome": 1, "as_of": "2026-06-20", "background": "b", "resolution_criteria": "r"}),
]


def scripted_model(prompt_text: str) -> str:
    """Answers the way a decent model would, keyed on the prompt."""
    if "resolves YES" in prompt_text:
        return '{"probability": 0.7}'
    ans = ("8330" if "85 × 98" in prompt_text else
           "Directed by Rob Zombie" if "Halloween" in prompt_text else "No record of that survives.")
    if "confidence first" in prompt_text:
        return json.dumps({"confidence": 0.8, "answer": ans})
    if "JSON object" in prompt_text:
        return json.dumps({"answer": ans, "confidence": 0.9})
    if "loses 9 points" in prompt_text and "Halloween" in prompt_text:
        return "I don't know"
    return ans


def judge_model(prompt_text: str) -> str:
    if "JSON list" in prompt_text:
        return "[true, true]"
    return "ABSTAINS" if "Nobody knows" in prompt_text else "CORRECT"


def _mock(fn):
    def call(messages, tools, tool_choice, config):
        return ModelOutput.from_content(model="mock", content=fn(messages[-1].text))
    return get_model("mockllm/model", custom_outputs=call, memoize=False)


@pytest.mark.parametrize("judge", [False, True])
def test_end_to_end(tmp_path, judge):
    path = tmp_path / "questions.jsonl"
    path.write_text("\n".join(json.dumps(i) for i in ITEMS))
    task = ct.confidence_audit(questions=str(path), k_samples=2, judge=judge)
    roles = {"grader": _mock(judge_model)} if judge else None
    [log] = inspect_eval(task, model=_mock(scripted_model), model_roles=roles,
                         log_dir=str(tmp_path / "logs"), display="none")
    assert log.status == "success"
    s = {x.id: x.scores["confidence_scorer"].value for x in log.samples}

    comp = s["comp-1"]
    assert comp["p_ok"] == 1 and comp["a_conf"] == 0.9 and comp["d_conf"] == 0.8
    assert comp["b_agree"] == 1.0 and comp["ungraded"] == 0 and comp["unparsed"] == 0
    assert all(comp[f"c{t}_answered"] == 1 for t in ct.THRESHOLDS)

    assert s["fc-1"]["f_prob"] == 0.7 and s["fc-1"]["f_brier"] == pytest.approx(0.09)

    rec, unk = s["rec-1"], s["unk-1"]
    assert rec["c0.9_answered"] == 0  # "I don't know" needs no judge
    if judge:
        assert rec["p_ok"] == 1 and rec["ungraded"] == 0 and rec["b_agree"] == 1.0
        assert unk["p_ok"] == 1 and unk["p_abstain"] == 1
    else:
        assert "p_ok" not in rec and rec["ungraded"] == 6  # P, A, D and C at t = 0, 0.5, 0.75
        assert "p_ok" not in unk
