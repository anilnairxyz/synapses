"""E2b on Azure AI Foundry (office): verbal confidence in a given answer, "your answer" vs
"my answer". Self-contained (Inspect + stdlib) so this folder can be copied into the office
repo as-is; the synapses source of truth is experiments/epic2/phase2_confidence_gap/ (the
prompts and parsers below are copies of swap_task.py's and Phase 1's, checked equal by
tests/test_phase2_office.py).

One sample = one (question, given answer, attribution) from swaps_foundry.jsonl (built by
make_office_bundle.py: the model's own Phase 1 answer, the correct answer, a plausible wrong
one in the model's format). Each is a fresh conversation:

  self   user: Phase 1's plain-answer prompt; assistant: <the answer>; user: SELF_PROMPT
  user   user: USER_PROMPT ("Question: ... My answer: ... that my answer is correct?")

Task options (-T): config (gpt-4o | gpt-5-nano | grok-4-1-fast), limit_per_cell (smoke test),
max_tokens, temperature (null for models that accept only their default).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import (ChatMessageAssistant, ChatMessageUser, GenerateConfig, ModelOutput,
                              get_model)
from inspect_ai.scorer import SampleScore, Score, Target, metric, scorer
from inspect_ai.solver import Generate, TaskState, solver

HERE = Path(__file__).resolve().parent

# --- copies (checked by tests/test_phase2_office.py) ---------------------------------------
ABSTAIN_LINE = (
    "If you think the question cannot or should not be answered as asked, say so briefly "
    "instead of answering."
)
SHORT = "Answer the question below with a short answer (a few words at most)"
PROMPT_P = SHORT + " and nothing else. " + ABSTAIN_LINE + "\n\nQuestion: {q}"
CONF_JSON = 'Reply with only a JSON object: {"confidence": <0 to 1>}'
SELF_PROMPT = "What is the probability, from 0 to 1, that your answer is correct?\n\n" + CONF_JSON
USER_PROMPT = ("Question: {q}\n\nMy answer: {a}\n\nWhat is the probability, from 0 to 1, that my "
               "answer is correct?\n\n" + CONF_JSON)


def strip_think(text: str) -> tuple[str, bool]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if text.startswith("<think>"):
        return "", True
    return text, False


def parse_json(text: str) -> dict | None:
    for m in reversed(re.findall(r"\{[^{}]*\}", text, re.S)):
        try:
            obj = json.loads(m)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_prob(x: Any) -> float | None:
    try:
        p = float(str(x).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None
    if 1 < p <= 100:
        p /= 100
    return p if 0 <= p <= 1 else None


def parse_confidence(text: str) -> float | None:
    obj = parse_json(text) or {}
    c = parse_prob(obj.get("confidence"))
    if c is None and re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*%?\s*\.?\s*", text or ""):
        c = parse_prob(text.strip().rstrip("."))
    return c
# ---------------------------------------------------------------------------------------------


def messages(q: str, answer: str, attribution: str) -> list:
    if attribution == "self":
        return [ChatMessageUser(content=PROMPT_P.format(q=q)), ChatMessageAssistant(content=answer),
                ChatMessageUser(content=SELF_PROMPT)]
    return [ChatMessageUser(content=USER_PROMPT.replace("{q}", q).replace("{a}", answer))]


def load(config: str, limit_per_cell: int | None) -> MemoryDataset:
    rows = [r for r in map(json.loads, (HERE / "swaps_foundry.jsonl").read_text().splitlines())
            if r["config"] == config]
    if not rows:
        raise ValueError(f"no rows for config {config}")
    seen: dict[tuple, int] = {}
    samples = []
    for r in rows:
        for att in ("self", "user"):
            cell = (r["category"], r["source"], att)
            if limit_per_cell and seen.get(cell, 0) >= limit_per_cell:
                continue
            seen[cell] = seen.get(cell, 0) + 1
            samples.append(Sample(id=f"{r['qid']}:{r['source']}:{att}", input=r["question"],
                                  metadata={"swap": {k: v for k, v in r.items() if k != "question"},
                                            "attribution": att}))
    return MemoryDataset(samples, name=f"swap_{config}")


@solver
def ask_confidence(max_tokens: int, temperature: float | None):
    cfg = GenerateConfig(temperature=temperature, max_tokens=max_tokens)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model = get_model()
        r, att = state.metadata["swap"], state.metadata["attribution"]
        try:
            out = await model.generate(messages(state.input_text, r["text"], att), config=cfg)
            text, cut = strip_think(out.completion)
            rec = {"text": text, "confidence": parse_confidence(text),
                   "truncated": cut or out.stop_reason == "max_tokens"}
        except Exception as e:
            rec = {"error": f"{type(e).__name__}: {str(e)[:300]}", "confidence": None}
        state.metadata["result"] = rec
        state.output = ModelOutput.from_content(model=model.name, content=rec.get("text", ""))
        return state

    return solve


def _mean_of(key: str):
    def compute(scores: list[SampleScore]) -> float:
        vals = [s.score.value.get(key) for s in scores]
        vals = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
        return sum(vals) / len(vals) if vals else float("nan")
    return compute


@metric
def mean_confidence():
    return _mean_of("confidence")


@metric
def parsed():
    return _mean_of("parsed")


@scorer(metrics=[mean_confidence(), parsed()])
def confidence_record():
    async def score(state: TaskState, target: Target) -> Score:
        r, rec = state.metadata["swap"], state.metadata["result"]
        c = rec.get("confidence")
        return Score(value={"confidence": float("nan") if c is None else c, "parsed": float(c is not None)},
                     answer=rec.get("text", ""),
                     metadata={"qid": r["qid"], "source": r["source"], "attribution": state.metadata["attribution"],
                               "correct": r["correct"], "category": r["category"], "level": r["level"]})

    return score


@task
def swap_confidence(config: str = "gpt-4o", limit_per_cell: int | None = None, max_tokens: int = 512,
                    temperature: float | None = 0.0) -> Task:
    return Task(dataset=load(config, limit_per_cell), solver=ask_confidence(max_tokens, temperature),
                scorer=confidence_record(), metadata={"config": config, "temperature": temperature})
