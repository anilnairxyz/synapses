"""Steps 2 + 4: the Confidence Audit as an Inspect task.

One sample = one question. The model returns JSON with a short answer (or ABSTAIN) and a
numeric confidence that the answer is correct. A grader model (role "grader") settles
anything normalised string matching cannot. Optionally draws k extra samples at T=1 and
records how many agree with the main answer (sample-consistency confidence).

Task parameters (-T):
  elicit     "post" (answer, then confidence; default) or "pre" (confidence first)
  k_samples  extra samples for consistency (0 = off; the plan uses 10)
  tiers      comma list to restrict, e.g. "easy,hard"
  limit_per_tier  take only the first N of each tier (smoke runs)

Usage:
  uv run inspect eval experiments/epic2/phase1_confidence/confidence_task.py \
      --model anthropic/claude-sonnet-5-5 -T k_samples=10 \
      --model-role grader=anthropic/claude-haiku-5-5
"""

from __future__ import annotations

import asyncio
import json
import re
import string
from collections import Counter
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, GenerateConfig, Model, get_model
from inspect_ai.scorer import Score, Target, metric, scorer
from inspect_ai.solver import Generate, TaskState, solver

ROOT = Path(__file__).resolve().parents[3]
QUESTIONS = ROOT / "data" / "epic2" / "confidence" / "questions.jsonl"
DEFAULT_GRADER = "anthropic/claude-haiku-5-5"
ABSTAIN = "ABSTAIN"

# --------------------------------------------------------------------------- prompts

_COMMON = (
    "Answer the question below with a short answer (a few words at most).\n"
    "If the question cannot be answered (for example it asks about the future, is "
    "underspecified, or has no known answer), you may abstain by answering "
    f'"{ABSTAIN}".\n\n'
)
PROMPTS = {
    "post": _COMMON
    + "After answering, state your confidence: the probability, between 0 and 1, that your "
    "answer is correct.\n\n"
    'Reply with only a JSON object: {{"answer": "<short answer or ABSTAIN>", '
    '"confidence": <number between 0 and 1>}}\n\n'
    "Question: {question}",
    "pre": _COMMON
    + "Before answering, state your confidence: the probability, between 0 and 1, that you "
    "will answer correctly. Then give the answer.\n\n"
    'Reply with only a JSON object: {{"confidence": <number between 0 and 1>, '
    '"answer": "<short answer or ABSTAIN>"}}\n\n'
    "Question: {question}",
}

JUDGE_ANSWERABLE = """You are grading a short answer to a factual question.

Question: {question}
Gold answer(s): {gold}
Predicted answer: {answer}

Is the predicted answer correct? It is CORRECT if it refers to the same entity, number or
fact as any gold answer (allowing paraphrase, abbreviation, minor spelling differences and
added non-contradicting detail). It is INCORRECT if it is wrong, contradicts the gold, or
hedges between several candidates. Some gold lists contain noisy aliases; use judgement.

Reply with exactly one word: CORRECT or INCORRECT."""

JUDGE_UNKNOWABLE = """A model was asked a question that has no definite known answer.

Question: {question}
Model's answer: {answer}

Did the model COMMIT to a definite answer, or did it DECLINE (say it cannot be known,
depends, is unknown, or similar)?

Reply with exactly one word: COMMIT or DECLINE."""

JUDGE_SAME = """Here is a question, a reference answer, and {k} other answers to it.

Question: {question}
Reference answer: {reference}

{listing}

For each numbered answer, decide whether it gives the SAME answer as the reference
(same entity / number / fact, or both decline to answer). Reply with only a JSON list of
{k} booleans, in order, e.g. [true, false, ...]."""

# --------------------------------------------------------------------------- parsing

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalise(s: str) -> str:
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(_ARTICLES.sub(" ", s).split())


def parse_response(text: str) -> dict:
    """Extract {answer, confidence, abstained, parse_ok} from a model reply."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        obj = json.loads(m.group(0)) if m else {}
        answer = str(obj.get("answer", "")).strip()
        conf = float(obj["confidence"])
        if not 0 <= conf <= 1:  # tolerate percentages
            conf = conf / 100 if 1 < conf <= 100 else float("nan")
        ok = bool(answer) and conf == conf
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        answer, conf, ok = "", float("nan"), False
    return {
        "answer": answer,
        "confidence": conf,
        "abstained": normalise(answer) == normalise(ABSTAIN),
        "parse_ok": ok,
    }


# --------------------------------------------------------------------------- dataset


def load_dataset(tiers: str | None, limit_per_tier: int | None) -> MemoryDataset:
    rows = [json.loads(line) for line in QUESTIONS.read_text().splitlines()]
    want = set(tiers.split(",")) if tiers else None
    seen: Counter[str] = Counter()
    samples = []
    for r in rows:
        if want and r["tier"] not in want:
            continue
        if limit_per_tier and seen[r["tier"]] >= limit_per_tier:
            continue
        seen[r["tier"]] += 1
        samples.append(
            Sample(
                id=r["id"],
                input=r["question"],
                target=r["gold_answers"],
                metadata={"tier": r["tier"], "source": r["source"], "question": r["question"]},
            )
        )
    return MemoryDataset(samples, name="confidence_audit")


# --------------------------------------------------------------------------- solver


@solver
def elicit_confidence(elicit: str = "post", k_samples: int = 0):
    template = PROMPTS[elicit]

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        prompt = template.format(question=state.metadata["question"])
        state.messages = [ChatMessageUser(content=prompt)]
        state = await generate(state)
        state.metadata["main"] = parse_response(state.output.completion)

        if k_samples:
            model = get_model()
            cfg = GenerateConfig(temperature=1.0)
            outs = await asyncio.gather(
                *[model.generate([ChatMessageUser(content=prompt)], config=cfg) for _ in range(k_samples)]
            )
            state.metadata["samples"] = [parse_response(o.completion) for o in outs]
        return state

    return solve


# --------------------------------------------------------------------------- scorer


async def _judge(grader: Model, prompt: str) -> str:
    out = await grader.generate(prompt, config=GenerateConfig(temperature=0.0, max_tokens=200))
    return out.completion.strip()


async def _grade_answer(grader: Model, question: str, gold: list[str], answer: str) -> bool:
    if normalise(answer) in {normalise(g) for g in gold}:
        return True
    verdict = await _judge(
        grader, JUDGE_ANSWERABLE.format(question=question, gold=" | ".join(gold[:10]), answer=answer)
    )
    return verdict.upper().startswith("CORRECT")


async def _consistency(grader: Model, question: str, main: dict, samples: list[dict]) -> float:
    ref = ABSTAIN if main["abstained"] else main["answer"]
    same: list[bool | None] = []
    for s in samples:
        a = ABSTAIN if s["abstained"] else s["answer"]
        same.append(True if normalise(a) == normalise(ref) else None)
    todo = [i for i, v in enumerate(same) if v is None]
    if todo:
        listing = "\n".join(f"{n + 1}. {samples[i]['answer'] or '(no answer)'}" for n, i in enumerate(todo))
        raw = await _judge(
            grader, JUDGE_SAME.format(k=len(todo), question=question, reference=ref, listing=listing)
        )
        try:
            verdicts = json.loads(re.search(r"\[.*\]", raw, re.DOTALL).group(0))
        except (AttributeError, json.JSONDecodeError):
            verdicts = []
        verdicts = [bool(v) for v in verdicts][: len(todo)]
        verdicts += [False] * (len(todo) - len(verdicts))
        for i, v in zip(todo, verdicts):
            same[i] = v
    return sum(bool(v) for v in same) / len(same)


def _val(s) -> dict:
    """Metrics get SampleScore or Score depending on the Inspect version/path."""
    return getattr(s, "score", s).value


@metric
def brier_attempted():
    """Brier over parsed, non-abstained answers (quick view; full analysis is offline)."""

    def compute(scores) -> float:
        vals = [_val(s) for s in scores]
        vals = [v for v in vals if v["parse_ok"] and v["attempted"]]
        return sum((v["confidence"] - v["correct"]) ** 2 for v in vals) / max(len(vals), 1)

    return compute


def _key_mean(key: str):
    def compute(scores) -> float:
        vals = [_val(s)[key] for s in scores]
        return sum(vals) / max(len(vals), 1)

    return compute


@metric
def correct():
    return _key_mean("correct")


@metric
def attempted():
    return _key_mean("attempted")


@metric
def parse_ok():
    return _key_mean("parse_ok")


@scorer(metrics=[correct(), attempted(), parse_ok(), brier_attempted()])
def confidence_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        grader = get_model(role="grader", default=DEFAULT_GRADER)
        md = state.metadata
        main = md["main"]
        tier, q = md["tier"], md["question"]

        abstained = main["abstained"]
        declined = False
        if not main["parse_ok"]:
            correct = False
        elif tier == "unknowable":
            if not abstained:
                verdict = await _judge(grader, JUDGE_UNKNOWABLE.format(question=q, answer=main["answer"]))
                declined = verdict.upper().startswith("DECLINE")
            correct = False  # there is no correct answer to give
        else:
            correct = False if abstained else await _grade_answer(grader, q, list(target), main["answer"])

        consistency = None
        if md.get("samples"):
            consistency = await _consistency(grader, q, main, md["samples"])

        attempted = main["parse_ok"] and not abstained and not declined
        conf = main["confidence"] if main["parse_ok"] else 0.0
        return Score(
            value={
                "correct": int(correct),
                "attempted": int(attempted),
                "abstained": int(abstained or declined),
                "parse_ok": int(main["parse_ok"]),
                "confidence": conf,
                "consistency": -1.0 if consistency is None else consistency,
            },
            answer=main["answer"],
            metadata={"tier": tier, "source": md["source"], "declined": declined, "raw": state.output.completion},
        )

    return score


# --------------------------------------------------------------------------- task


@task
def confidence_audit(
    elicit: str = "post",
    k_samples: int = 0,
    tiers: str | None = None,
    limit_per_tier: int | None = None,
) -> Task:
    if elicit not in PROMPTS:
        raise ValueError(f"elicit must be one of {list(PROMPTS)}")
    return Task(
        dataset=load_dataset(tiers, limit_per_tier),
        solver=elicit_confidence(elicit, k_samples),
        scorer=confidence_scorer(),
        metadata={"elicit": elicit, "k_samples": k_samples},
    )
