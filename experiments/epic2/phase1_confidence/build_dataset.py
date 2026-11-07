"""Step 1: build the three-tier question set for the Confidence Audit.

Tiers (default 100 each, fixed seed):
  easy        TriviaQA rc.nocontext validation (gold = answer value + aliases)
  hard        SimpleQA test set (single gold answer)
  unknowable  KUQ {future unknown, underspecified, unsolved} + SelfAware unanswerable,
              25 each; no gold answer, correct behaviour is to abstain / flag uncertainty

Writes data/epic2/confidence/questions.jsonl with fields
  id, tier, source, question, gold_answers (list[str], empty for unknowable)

Usage:
  uv run python experiments/epic2/phase1_confidence/build_dataset.py [--n 100] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "epic2" / "raw"
OUT = ROOT / "data" / "epic2" / "confidence" / "questions.jsonl"

KUQ_CATEGORIES = {
    "future unknown": "kuq_future",
    "underspecified question": "kuq_underspecified",
    "unsolved problem/mistery": "kuq_unsolved",
}


def _fetch(repo: str, filename: str) -> Path:
    return Path(hf_hub_download(repo, filename, repo_type="dataset", local_dir=RAW))


def easy(n: int, seed: int) -> list[dict]:
    df = pd.read_parquet(_fetch("mandarjoshi/trivia_qa", "rc.nocontext/validation-00000-of-00001.parquet"))
    # Prefer short, unambiguous answers so string matching does most of the grading.
    df = df[df.answer.map(lambda a: len(a["value"].split()) <= 4)]
    df = df.sample(n, random_state=seed)
    rows = []
    for r in df.itertuples():
        gold = list(dict.fromkeys([r.answer["value"], *r.answer["aliases"]]))
        rows.append({"id": f"easy-{r.question_id}", "tier": "easy", "source": "triviaqa",
                     "question": r.question, "gold_answers": gold})
    return rows


def hard(n: int, seed: int) -> list[dict]:
    df = pd.read_csv(_fetch("basicv8vc/SimpleQA", "simple_qa_test_set.csv")).sample(n, random_state=seed)
    return [{"id": f"hard-{i}", "tier": "hard", "source": "simpleqa",
             "question": r.problem, "gold_answers": [str(r.answer)]} for i, r in df.iterrows()]


def unknowable(n: int, seed: int) -> list[dict]:
    per = n // 4
    kuq = pd.read_json(_fetch("amayuelas/KUQ", "unknowns_all.jsonl"), lines=True)
    kuq = kuq[kuq.category_agreement & kuq.category.isin(KUQ_CATEGORIES)]
    parts = [
        kuq[kuq.category == c].sample(per, random_state=seed).assign(source=s)
        for c, s in KUQ_CATEGORIES.items()
    ]
    sa = pd.read_parquet(_fetch("JesusCrist/SelfAware", "data/train-00000-of-00001.parquet"))
    sa = sa[~sa.answerable].sample(n - 3 * per, random_state=seed).assign(source="selfaware")
    parts.append(sa)
    rows = []
    for p in parts:
        for i, r in p.iterrows():
            rows.append({"id": f"unk-{r.source}-{i}", "tier": "unknowable", "source": r.source,
                         "question": r.question, "gold_answers": []})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="questions per tier")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = easy(args.n, args.seed) + hard(args.n, args.seed) + unknowable(args.n, args.seed)
    assert len({r["id"] for r in rows}) == len(rows), "duplicate ids"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} questions -> {OUT.relative_to(ROOT)}")
    print(pd.DataFrame(rows).groupby(["tier", "source"]).size().to_string())


if __name__ == "__main__":
    main()
