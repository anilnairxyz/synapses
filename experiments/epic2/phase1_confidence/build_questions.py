"""Step 1: build the Phase 1 question set (all categories implemented so far).

Writes data/epic2/confidence/questions.jsonl; see questions/schema.py for the record format.

Usage:
  uv run python experiments/epic2/phase1_confidence/build_questions.py [--per-level 30] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from questions import computation, forecastable, ill_posed, perception, recall, unknowable

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "epic2" / "confidence" / "questions.jsonl"

# Answerable categories take --per-level; behaviour categories have their own sizes.
BUILDERS = {
    "recall": lambda per_level, seed: recall.build(per_level, seed),
    "computation": lambda per_level, seed: computation.build(per_level, seed),
    "perception": lambda per_level, seed: perception.build(per_level, seed),
    "ill_posed": lambda per_level, seed: ill_posed.build(seed),
    "unknowable": lambda per_level, seed: unknowable.build(seed),
    "forecastable": lambda per_level, seed: forecastable.build(seed),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-level", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    items = [q for build in BUILDERS.values() for q in build(args.per_level, args.seed)]
    ids = Counter(q.id for q in items)
    dupes = [i for i, n in ids.items() if n > 1]
    if dupes:
        raise SystemExit(f"duplicate ids: {dupes[:5]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for q in items:
            f.write(json.dumps(q.to_dict()) + "\n")
    print(f"wrote {len(items)} questions -> {args.out.relative_to(ROOT)}")
    for (cat, lvl), n in sorted(Counter((q.category, q.level or 0) for q in items).items()):
        print(f"  {cat:12s} {f'L{lvl}' if lvl else '--'}  {n}")


if __name__ == "__main__":
    main()
