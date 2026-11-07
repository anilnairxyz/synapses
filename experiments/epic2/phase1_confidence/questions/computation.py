"""Computation: multi-digit multiplication, difficulty = operand digit counts.

One task family on purpose: mixing families (division, multi-step expressions) would mix
difficulty with task type. The answer can't be stored anywhere (operands are random), so
it has to be worked out, which is where errors snowball during generation.

Operands have exactly the stated number of digits and never end in 0 (no shortcut via
trailing zeros). The answer is exact, graded as an integer.
"""

from __future__ import annotations

import random

from .schema import Question

# (digits of a, digits of b) per level. Calibrate in the pilot: level 1 should be near
# 100% accurate and level 5 well below 50% for non-reasoning models.
LEVELS: dict[int, tuple[int, int]] = {1: (2, 2), 2: (3, 2), 3: (3, 3), 4: (4, 3), 5: (4, 4)}


def _operand(rng: random.Random, digits: int) -> int:
    while True:
        n = rng.randint(10 ** (digits - 1), 10**digits - 1)
        if n % 10:
            return n


def build(per_level: int, seed: int) -> list[Question]:
    rng = random.Random(f"computation-{seed}")
    out: list[Question] = []
    for level, (da, db) in LEVELS.items():
        seen: set[tuple[int, int]] = set()
        while len(seen) < per_level:
            a, b = _operand(rng, da), _operand(rng, db)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            out.append(
                Question(
                    id=f"comp-L{level}-{len(seen):03d}",
                    category="computation",
                    kind="answerable",
                    level=level,
                    source="generated",
                    question=f"What is {a} × {b}?",
                    answer_type="integer",
                    gold_answers=[str(a * b)],
                    meta={"a": a, "b": b, "digits": [da, db], "family": "multiplication"},
                )
            )
    return out
