"""Phase 1B step 2: build the probe TRAINING questions (answerable kinds only).

The Phase 1 question set (questions.jsonl) stays the test set. This script draws new
questions from the same generators, with no overlap with the test set:

- computation: new operand pairs per level (same digit counts).
- perception: new words per level (same length bands, same three task families); no word
  that appears in any test question.
- recall: PopQA items with the same five relations, excluding every subject used anywhere
  in the test set (recall and the PopQA-based ill-posed items). PopQA is thin at both ends
  of the popularity scale, so levels are filled as evenly as the pool allows ("water
  filling"), and relations within a level likewise. The training level mix is therefore
  uneven for recall; that only affects what the probe learns from, not how it is tested.

Writes data/epic2/confidence/train_questions.jsonl (ids prefixed "tr-") and prints counts.

Usage:
  uv run python experiments/epic2/phase1_confidence/build_train_questions.py [--per-kind 2000]
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from questions import computation, perception, recall
from questions.schema import Question

ROOT = Path(__file__).resolve().parents[3]
TEST = ROOT / "data" / "epic2" / "confidence" / "questions.jsonl"
OUT = ROOT / "data" / "epic2" / "confidence" / "train_questions.jsonl"
LEVELS = (1, 2, 3, 4, 5)


def water_fill(capacity: dict, total: int) -> dict:
    """Split ``total`` over keys as evenly as possible without exceeding each capacity."""
    take = dict.fromkeys(capacity, 0)
    left = total
    open_keys = [k for k in capacity if capacity[k] > 0]
    while left > 0 and open_keys:
        share = max(left // len(open_keys), 1)
        for k in list(open_keys):
            add = min(share, capacity[k] - take[k], left)
            take[k] += add
            left -= add
            if take[k] >= capacity[k]:
                open_keys.remove(k)
            if left == 0:
                break
    return take


def build_computation(n_per_level: int, test: list[dict], seed: int) -> list[Question]:
    rng = random.Random(f"train-computation-{seed}")
    seen = {(q["meta"]["a"], q["meta"]["b"]) for q in test if q["category"] == "computation"}
    out = []
    for level, (da, db) in computation.LEVELS.items():
        n = 0
        while n < n_per_level:
            a, b = computation._operand(rng, da), computation._operand(rng, db)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            n += 1
            out.append(Question(
                id=f"tr-comp-L{level}-{n:04d}", category="computation", kind="answerable",
                level=level, source="generated", question=f"What is {a} × {b}?",
                answer_type="integer", gold_answers=[str(a * b)],
                meta={"a": a, "b": b, "digits": [da, db], "family": "multiplication"}))
    return out


def build_perception(n_per_level: int, test: list[dict], seed: int) -> list[Question]:
    rng = random.Random(f"train-perception-{seed}")
    test_words = {q["meta"]["word"] for q in test if q["category"] == "perception"}
    fams = perception.FAMILIES
    out = []
    for level, pool in perception._words_by_level().items():
        pool = [w for w in pool if w not in test_words]
        # Equal numbers per family; a word appears at most once per family.
        per_fam = min(n_per_level // len(fams), len(pool))
        pairs = [(w, f) for f in fams for w in rng.sample(pool, per_fam)]
        rng.shuffle(pairs)
        for i, (word, family) in enumerate(pairs):
            q, atype, gold = perception._item(family, word, rng)
            out.append(Question(
                id=f"tr-perc-L{level}-{i + 1:04d}", category="perception", kind="answerable",
                level=level, source="generated", question=q, answer_type=atype,
                gold_answers=[gold],
                meta={"family": family, "word": word, "length": len(word),
                      "zipf": round(perception.zipf_frequency(word, "en"), 2)}))
    return out


def build_recall(total: int, test: list[dict], seed: int) -> list[Question]:
    rng = random.Random(f"train-recall-{seed}")
    used_ids = {q["meta"].get("popqa_id") for q in test} - {None}
    used_subj = {q["meta"].get("subject") for q in test} - {None}
    texts = " ".join(q["question"] for q in test if q["source"].startswith("popqa"))
    pools: dict[tuple[int, str], list[dict]] = {}
    for key, rows in recall.candidates().items():
        pools[key] = [r for r in rows
                      if r["id"] not in used_ids and r["display"] not in used_subj
                      and r["subj"] not in used_subj and r["display"] not in texts]
    # One question per subject, as in the test set.
    seen_subj: set[str] = set()
    for key in sorted(pools):
        rows = rng.sample(pools[key], len(pools[key]))
        pools[key] = [r for r in rows if not (r["subj"] in seen_subj or seen_subj.add(r["subj"]))]
    per_level = water_fill({lvl: sum(len(pools.get((lvl, rel), [])) for rel in recall.RELATIONS)
                            for lvl in LEVELS}, total)
    out = []
    for level in LEVELS:
        per_rel = water_fill({rel: len(pools.get((level, rel), [])) for rel in recall.RELATIONS},
                             per_level[level])
        picked = [r for rel in recall.RELATIONS for r in pools.get((level, rel), [])[:per_rel[rel]]]
        for i, r in enumerate(picked):
            out.append(Question(
                id=f"tr-rec-L{level}-{i + 1:04d}", category="recall", kind="answerable",
                level=level, source="popqa", question=r["text"], answer_type="short_text",
                gold_answers=r["answers"],
                meta={"relation": r["prop"], "subject": r["display"], "s_pop": r["s_pop"],
                      "log10_pop": round(math.log10(r["s_pop"]), 2),
                      "popqa_id": r["id"], "s_uri": r["s_uri"]}))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-kind", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    test = [json.loads(line) for line in TEST.read_text().splitlines() if line.strip()]
    per_level = args.per_kind // len(LEVELS)
    items = (build_recall(args.per_kind, test, args.seed)
             + build_computation(per_level, test, args.seed)
             + build_perception(per_level, test, args.seed))

    ids = Counter(q.id for q in items)
    if any(n > 1 for n in ids.values()):
        raise SystemExit("duplicate ids")
    test_q = {q["question"] for q in test}
    if overlap := [q.id for q in items if q.question in test_q]:
        raise SystemExit(f"overlap with the test set: {overlap[:5]}")

    args.out.write_text("".join(json.dumps(asdict(q)) + "\n" for q in items))
    counts = Counter((q.category, q.level) for q in items)
    print(f"wrote {len(items)} items to {args.out}")
    for cat in ("recall", "computation", "perception"):
        print(f"  {cat:12s}", " ".join(f"L{lvl}={counts[(cat, lvl)]}" for lvl in LEVELS),
              f"total={sum(counts[(cat, lvl)] for lvl in LEVELS)}")
    rel = Counter((q.level, q.meta["relation"]) for q in items if q.category == "recall")
    print("  recall by level x relation:")
    for lvl in LEVELS:
        print(f"    L{lvl}", {r: rel[(lvl, r)] for r in recall.RELATIONS})


if __name__ == "__main__":
    main()
