"""E2 (and E3's post-hoc and user-framing readouts): verbal confidence in a given answer.

One sample = one (question, given answer, attribution) from swaps.jsonl (build_swaps.py),
test split, one instruct config. Each is a fresh conversation, temperature 0, thinking off:

  self   the answer sits in the assistant turn, as if the model had given it:
           user       Phase 1's plain-answer prompt (PROMPT_P) with the question
           assistant  <the given answer>
           user       SELF_PROMPT ("... that your answer is correct?")
  user   the user gives the answer as theirs (Sanz-Guerrero et al.'s framing):
           user       USER_PROMPT ("Question: ... My answer: ... that my answer is correct?")

Sources: own (the model's Phase 1 answer, verbatim), correct, wrong (built by construction,
in the model's own format). The confidence is parsed with Phase 1's JSON and probability
parsers. P(True) and the hidden states for the same answers come from extract_answers.py.

Task options (-T): config (qwen3-8b | qwen3-1.7b), limit_per_cell (smoke test: N questions
per kind x source x attribution), max_tokens (default 64).

    uv run --extra evals inspect eval experiments/epic2/phase2_confidence_gap/swap_task.py \\
        --model openai-api/local/Qwen/Qwen3-8B -T config=qwen3-8b
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import (ChatMessageAssistant, ChatMessageUser, GenerateConfig, ModelOutput,
                              get_model)
from inspect_ai.scorer import SampleScore, Score, Target, metric, scorer
from inspect_ai.solver import Generate, TaskState, solver

sys.path.insert(0, str(Path(__file__).resolve().parent))  # Inspect loads this file by path
import phase1  # noqa: E402

ct = phase1.ct  # bound while loading: Inspect restores sys.path before the task runs

CONF_JSON = 'Reply with only a JSON object: {"confidence": <0 to 1>}'
SELF_PROMPT = "What is the probability, from 0 to 1, that your answer is correct?\n\n" + CONF_JSON
USER_PROMPT = ("Question: {q}\n\nMy answer: {a}\n\nWhat is the probability, from 0 to 1, that my "
               "answer is correct?\n\n" + CONF_JSON)
ATTRIBUTIONS = ("self", "user")
INSTRUCT = ("qwen3-8b", "qwen3-1.7b")


def messages(q: str, answer: str, attribution: str) -> list:
    if attribution == "self":
        return [ChatMessageUser(content=ct.PROMPT_P.format(q=q)),
                ChatMessageAssistant(content=answer),
                ChatMessageUser(content=SELF_PROMPT)]
    if attribution == "user":
        return [ChatMessageUser(content=USER_PROMPT.replace("{q}", q).replace("{a}", answer))]
    raise ValueError(attribution)


def load(config: str, limit_per_cell: int | None) -> MemoryDataset:
    if config not in INSTRUCT:
        raise ValueError(f"config must be one of {INSTRUCT} (the base model gets P(True) only)")
    qs = {q["id"]: q for q in map(json.loads, phase1.PHASE1_QUESTIONS.read_text().splitlines())}
    rows = [r for r in map(json.loads, (phase1.DATA / "swaps.jsonl").read_text().splitlines())
            if r["config"] == config and r["split"] == "test"]
    seen: dict[tuple, int] = {}
    samples = []
    for r in rows:
        for att in ATTRIBUTIONS:
            cell = (r["category"], r["source"], att)
            if limit_per_cell and seen.get(cell, 0) >= limit_per_cell:
                continue
            seen[cell] = seen.get(cell, 0) + 1
            samples.append(Sample(id=f"{r['qid']}:{r['source']}:{att}", input=qs[r["qid"]]["question"],
                                  metadata={"swap": r, "attribution": att}))
    return MemoryDataset(samples, name=f"swap_{config}")


def parse_confidence(text: str) -> float | None:
    """The JSON's "confidence" (Phase 1's parsers); failing that, a reply that is just a
    number ("0.9", "90%"): the instruct models sometimes skip the JSON wrapper."""
    obj = ct.parse_json(text) or {}
    c = ct.parse_prob(obj.get("confidence"))
    if c is None and re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*%?\s*\.?\s*", text or ""):
        c = ct.parse_prob(text.strip().rstrip("."))
    return c


@solver
def ask_confidence(max_tokens: int):
    cfg = GenerateConfig(temperature=0.0, max_tokens=max_tokens,
                         extra_body={"chat_template_kwargs": {"enable_thinking": False}})

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model = get_model()
        r, att = state.metadata["swap"], state.metadata["attribution"]
        rec: dict[str, Any] = {}
        try:
            out = await model.generate(messages(state.input_text, r["text"], att), config=cfg)
            text, cut = ct.strip_think(out.completion)
            rec = {"text": text, "confidence": parse_confidence(text),
                   "truncated": cut or out.stop_reason == "max_tokens"}
        except Exception as e:  # keep the run going; the record shows the error
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
    """Mean of the parsed confidences (unparsed replies left out)."""
    return _mean_of("confidence")


@metric
def parsed():
    """Share of replies with a confidence that parsed."""
    return _mean_of("parsed")


@scorer(metrics=[mean_confidence(), parsed()])
def confidence_record():
    """Value: the parsed confidence (NaN if unparsed) and whether it parsed; the given
    answer's correctness and the cell go into the score metadata for the analysis."""

    async def score(state: TaskState, target: Target) -> Score:
        r, rec = state.metadata["swap"], state.metadata["result"]
        c = rec.get("confidence")
        return Score(value={"confidence": float("nan") if c is None else c, "parsed": float(c is not None)},
                     answer=rec.get("text", ""),
                     metadata={"qid": r["qid"], "source": r["source"], "attribution": state.metadata["attribution"],
                               "correct": r["correct"], "category": r["category"], "level": r["level"]})

    return score


@task
def swap_confidence(config: str = "qwen3-8b", limit_per_cell: int | None = None, max_tokens: int = 64) -> Task:
    return Task(dataset=load(config, limit_per_cell), solver=ask_confidence(max_tokens),
                scorer=confidence_record(), metadata={"config": config})
