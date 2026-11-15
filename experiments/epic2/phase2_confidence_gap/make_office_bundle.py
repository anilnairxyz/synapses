"""Writes office/swaps_foundry.jsonl: the E2b answer sets (build_swaps.py) for the Foundry
models, with the question text included so the office bundle needs nothing else.

    uv run --extra evals python experiments/epic2/phase2_confidence_gap/make_office_bundle.py
"""

from __future__ import annotations

import json

import pandas as pd

import phase1

FOUNDRY = ("gpt-4o", "gpt-5-nano", "grok-4-1-fast")
OUT = phase1.HERE / "office" / "swaps_foundry.jsonl"


def main() -> None:
    qs = {q["id"]: q["question"] for q in map(json.loads, phase1.PHASE1_QUESTIONS.read_text().splitlines())}
    d = pd.read_json(phase1.DATA / "swaps.jsonl", lines=True)
    d = d[d.config.isin(FOUNDRY) & (d.split == "test")].copy()
    d["question"] = d.qid.map(qs)
    assert d.question.notna().all()
    d[["config", "qid", "category", "level", "family", "source", "text", "correct", "question"]].to_json(
        OUT, orient="records", lines=True, force_ascii=False)
    print(f"wrote {OUT} ({len(d)} rows)", d.groupby("config").size().to_dict())


if __name__ == "__main__":
    main()
