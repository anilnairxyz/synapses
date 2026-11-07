"""Forecastable: real prediction-market questions that have since resolved, from ForecastBench.

Right behaviour: a calibrated probability, scored with Brier against the outcome. This is
confidence about the world, not about the model itself (see the plan's framing), and the
category where "I don't know" is *not* the right answer.

Source: ForecastBench question sets and resolution sets (forecastingresearch/
forecastbench-datasets, CC BY-SA 4.0), pinned to one commit so the build is reproducible
(resolution files are rewritten as more questions resolve).

Choices:
- Market questions only (Polymarket, Metaculus, Manifold, INFER): natural-language world
  events. The dataset questions (ACLED, FRED, yfinance, ...) are templated "will this series
  be higher on date X" items, a different task.
- After every candidate model's training cutoff. Each question is taken from the first
  question set due on or after FREEZE_FROM (2026-07-01; its freeze date is about ten days
  earlier) and asked "as of" that freeze date, so the same prompt goes to every model. All
  outcomes resolve from July 2026 on; the resolution date is in meta, so a model with a
  later cutoff can be filtered at run time.
- Not trivial: questions whose crowd probability at the freeze was below 5% or above 95%
  are dropped (mostly "will long-shot candidate X win" families from Polymarket).
- Not near-duplicates: Polymarket lists one market per candidate or bracket, so questions
  are grouped by their text with names and numbers removed, and one is kept per group.
- Source mix: SOURCE_QUOTA, so Polymarket (two thirds of the pool) doesn't dominate.

meta keeps the outcome (0/1), the crowd probability at the freeze (a human baseline the
models can be compared with; never shown to the model), the background and resolution
criteria, and the dates. No difficulty level.
"""

from __future__ import annotations

import json
import random
import re
import urllib.request
from pathlib import Path

from .schema import Question

REPO = "forecastingresearch/forecastbench-datasets"
COMMIT = "360971a45164931d9564daf40518c8bf72514828"  # main on 2026-11-02
SETS = [  # forecast due dates of the question sets used (each has a resolution set)
    "2026-07-05", "2026-07-19", "2026-08-02", "2026-08-16", "2026-08-30",
    "2026-09-13", "2026-09-27", "2026-10-11",
]
FREEZE_FROM = "2026-07-01"
MARKETS = ("polymarket", "metaculus", "manifold", "infer")
CROWD_RANGE = (0.05, 0.95)
SOURCE_QUOTA = {"polymarket": 35, "metaculus": 20, "manifold": 20}

CACHE = Path(__file__).resolve().parents[4] / "data" / "epic2" / "confidence" / "raw" / "forecastbench"


def _fetch(path: str) -> dict:
    local = CACHE / COMMIT[:12] / path
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/datasets/{path}"
        with urllib.request.urlopen(url, timeout=60) as r:
            local.write_bytes(r.read())
    return json.loads(local.read_text())


def _family(question: str) -> str:
    """Question text with names and numbers removed, to group per-candidate markets."""
    words = re.sub(r"[\d$%.,:;?!'\"()]+", " ", question).split()
    return " ".join(w.lower() for w in words[1:] if not w[:1].isupper())


def pool() -> list[dict]:
    resolved: dict[str, dict] = {}
    for d in SETS:
        for r in _fetch(f"resolution_sets/{d}_resolution_set.json")["resolutions"]:
            if (r["source"] in MARKETS and isinstance(r["id"], str) and r["resolved"]
                    and r["resolved_to"] in (0, 1)):
                resolved[r["id"]] = r
    first: dict[str, tuple[str, dict]] = {}
    for d in SETS:
        if d < FREEZE_FROM:
            continue
        for q in _fetch(f"question_sets/{d}-llm.json")["questions"]:
            if isinstance(q["id"], str) and q["id"] in resolved and q["id"] not in first:
                first[q["id"]] = (d, q)

    lo, hi = CROWD_RANGE
    items = []
    for qid, (d, q) in sorted(first.items()):
        r, freeze = resolved[qid], q["freeze_datetime"][:10]
        try:
            crowd = float(q["freeze_datetime_value"])
        except (TypeError, ValueError):
            continue
        if r["resolution_date"] <= freeze or not lo <= crowd <= hi:
            continue
        items.append({"q": q, "set": d, "freeze": freeze, "crowd": crowd,
                      "outcome": int(r["resolved_to"]), "resolved_on": r["resolution_date"]})
    return items


def build(seed: int, quota: dict[str, int] | None = None) -> list[Question]:
    quota = quota or SOURCE_QUOTA
    rng = random.Random(f"forecastable-{seed}")
    items = pool()
    rng.shuffle(items)
    seen: set[str] = set()
    picked: dict[str, list[dict]] = {s: [] for s in quota}
    for it in items:
        src, fam = it["q"]["source"], _family(it["q"]["question"])
        if src in picked and len(picked[src]) < quota[src] and fam not in seen:
            seen.add(fam)
            picked[src].append(it)
    out: list[Question] = []
    for src, its in picked.items():
        if len(its) < quota[src]:
            raise ValueError(f"only {len(its)} {src} questions for a quota of {quota[src]}")
        for it in sorted(its, key=lambda x: (x["freeze"], x["q"]["id"])):
            q = it["q"]
            out.append(
                Question(
                    id=f"fc-{src[:4]}-{len(out) + 1:03d}",
                    category="forecastable",
                    kind="behaviour",
                    level=None,
                    source=f"forecastbench-{src}",
                    question=q["question"],
                    answer_type="probability",
                    gold_answers=[],
                    right_behaviour="probability",
                    meta={"outcome": it["outcome"], "crowd": it["crowd"],
                          "as_of": it["freeze"], "resolved_on": it["resolved_on"],
                          "question_set": it["set"], "fb_id": q["id"], "url": q["url"],
                          "background": q["background"],
                          "resolution_criteria": q["resolution_criteria"]},
                )
            )
    return out
