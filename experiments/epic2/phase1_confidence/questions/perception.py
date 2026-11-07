"""Perception / representation: character-level questions, difficulty = word length.

Models see tokens (word chunks), not letters, so questions about letters probe a blind
spot built into the architecture. Three task families, balanced within every level so
family and difficulty aren't mixed up:

  count    How many times does the letter 'r' appear in "strawberry"?   -> integer
           (the word's most frequent letter)
  nth      What is the 7th letter of "strawberry"?                       -> letter
           (position drawn from the last two-thirds of the word)
  reverse  Spell "strawberry" backwards.                                 -> exact string

Words come from wordfreq's English list (real, reasonably common words: Zipf >= 3).
Longer words are naturally rarer, so each item records its Zipf frequency and token-free
length for analysis; frequency is a possible confound for the length knob.
"""

from __future__ import annotations

import random
from collections import Counter

from wordfreq import top_n_list, zipf_frequency

from .schema import Question

# Word-length range (inclusive) per level.
LEVELS: dict[int, tuple[int, int]] = {1: (4, 5), 2: (6, 7), 3: (8, 9), 4: (10, 12), 5: (13, 16)}
FAMILIES = ("count", "nth", "reverse")
MIN_ZIPF = 3.0


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _words_by_level() -> dict[int, list[str]]:
    words = [
        w for w in top_n_list("en", 60000)
        if w.isascii() and w.isalpha() and w.islower() and zipf_frequency(w, "en") >= MIN_ZIPF
    ]
    return {lvl: sorted(w for w in words if lo <= len(w) <= hi) for lvl, (lo, hi) in LEVELS.items()}


def _item(family: str, word: str, rng: random.Random) -> tuple[str, str, str]:
    """Returns (question, answer_type, gold) for one family."""
    if family == "count":
        # The most frequent letter (ties broken at random), so counts grow with length.
        counts = Counter(word)
        top = max(counts.values())
        letter = rng.choice(sorted(c for c, k in counts.items() if k == top))
        return (f"How many times does the letter '{letter}' appear in the word \"{word}\"?",
                "integer", str(counts[letter]))
    if family == "nth":
        # Positions in the last two-thirds, so longer words mean deeper positions.
        n = rng.randint(-(-len(word) // 3), len(word))
        return (f"What is the {_ordinal(n)} letter of the word \"{word}\"?", "letter", word[n - 1])
    if family == "reverse":
        return (f"Spell the word \"{word}\" backwards.", "exact_string", word[::-1])
    raise ValueError(family)


def build(per_level: int, seed: int) -> list[Question]:
    if per_level % len(FAMILIES):
        raise ValueError(f"per_level must be a multiple of {len(FAMILIES)} to balance families")
    rng = random.Random(f"perception-{seed}")
    pools = _words_by_level()
    out: list[Question] = []
    for level, pool in pools.items():
        words = rng.sample(pool, per_level)  # no word repeats within a level
        for i, word in enumerate(words):
            family = FAMILIES[i % len(FAMILIES)]
            q, atype, gold = _item(family, word, rng)
            out.append(
                Question(
                    id=f"perc-L{level}-{i + 1:03d}",
                    category="perception",
                    kind="answerable",
                    level=level,
                    source="generated",
                    question=q,
                    answer_type=atype,
                    gold_answers=[gold],
                    meta={"family": family, "word": word, "length": len(word),
                          "zipf": round(zipf_frequency(word, "en"), 2)},
                )
            )
    return out
