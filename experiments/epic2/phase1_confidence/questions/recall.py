"""Recall: single-fact questions from PopQA, difficulty = subject popularity.

PopQA (Mallen et al., 2023; HF "akariasai/PopQA") has 14k templated questions over Wikidata
triples, each with the subject's monthly Wikipedia pageviews (s_pop). Popularity is a proxy
for how often the fact appeared in training data, so it is the long-tail knob: Kalai's
singleton argument predicts errors (and, ideally, lower confidence) at the rare end.

Levels are decades of monthly pageviews, so each level is ~10x rarer than the one before:

  L1 >= 100k   L2 10k-100k   L3 1k-10k   L4 100-1k   L5 < 100

Choices that keep popularity from being mixed up with something else:

- Same relations at every level. PopQA's relation mix shifts with popularity (capitals are
  popular, "sport" and "country" are long-tail), so each level gets the same number of
  items from each of RELATIONS. Relations whose answers can be guessed from a prior
  (genre, occupation, sport, country, religion, colour) are left out: a right guess there
  isn't recall. "capital" is left out too: rare capitals are mostly named after their
  region (only 3 survive the filter below at L5).
- Unambiguous subjects. Items are dropped when the subject name maps to several entities,
  or the subject has several triples for the relation. About half the remaining subjects
  share a name with something else ("White Gold", "The List"), so the question uses the
  Wikipedia title instead ("The List (2012 film)"), unless that title leaks the answer
  ("String Quartet No. 3 (Nielsen)" is dropped).
- Each subject appears once in the whole set (a film can't be asked about for both
  director and screenwriter).
- No free answers. Items whose answer, or its first five letters, is in the subject are
  dropped ("capital of Canton of Gordes" -> Gordes, "Kambarsky District" -> Kambarka).

Gold answers are PopQA's possible_answers (the object and its Wikidata aliases); grading is
string match first, then an LLM judge (answer_type short_text).
"""

from __future__ import annotations

import functools
import json
import math
import random
import re

from .schema import Question

RELATIONS = ("director", "screenwriter", "author", "composer", "father")
# Lower bound of monthly pageviews per level (L5 has no lower bound).
LEVEL_FLOOR = {1: 100_000, 2: 10_000, 3: 1_000, 4: 100, 5: 0}


def level_of(pageviews: int) -> int:
    return next(lvl for lvl, floor in LEVEL_FLOOR.items() if pageviews >= floor)


def _leaks(display: str, subj: str, answers: list[str]) -> bool:
    """True if the displayed subject adds a word that appears in an answer."""
    added = set(re.findall(r"\w{3,}", display.lower())) - set(re.findall(r"\w{3,}", subj.lower()))
    return any(w in added for a in answers for w in re.findall(r"\w{3,}", a.lower()))


def _given_away(answer: str, subj: str) -> bool:
    """True if the subject spells out the answer, or its stem ("Kambarsky District" -> Kambarka)."""
    a, s = answer.lower(), subj.lower()
    return (len(a) >= 3 and a in s) or (len(a) >= 5 and a[:5] in s)


@functools.cache
def popqa_rows() -> list[dict]:
    """All PopQA rows (cached; ill_posed.py builds on the same data)."""
    from datasets import load_dataset  # heavy; only needed when building

    return load_dataset("akariasai/PopQA", split="test").to_list()


def candidates() -> dict[tuple[int, str], list[dict]]:
    """Clean single-answer items per (level, relation); see the module docstring."""
    rows = popqa_rows()
    ids_per_name: dict[str, set[int]] = {}
    triples: dict[tuple[str, str], int] = {}
    for r in rows:
        ids_per_name.setdefault(r["subj"], set()).add(r["subj_id"])
        triples[(r["subj"], r["prop"])] = triples.get((r["subj"], r["prop"]), 0) + 1

    pools: dict[tuple[int, str], list[dict]] = {}
    for r in rows:
        if r["prop"] not in RELATIONS:
            continue
        if len(ids_per_name[r["subj"]]) > 1 or triples[(r["subj"], r["prop"])] > 1:
            continue
        answers = json.loads(r["possible_answers"])
        if any(_given_away(a, r["subj"]) for a in answers):
            continue
        display = r["s_wiki_title"] or r["subj"]
        if display != r["subj"]:
            if _leaks(display, r["subj"], answers) or r["subj"] not in r["question"]:
                continue
        r = dict(r, answers=answers, display=display,
                 text=r["question"].replace(r["subj"], display, 1))
        pools.setdefault((level_of(r["s_pop"]), r["prop"]), []).append(r)
    for pool in pools.values():
        pool.sort(key=lambda r: r["id"])  # stable order before sampling
    return pools


def build(per_level: int, seed: int) -> list[Question]:
    if per_level % len(RELATIONS):
        raise ValueError(f"per_level must be a multiple of {len(RELATIONS)} to balance relations")
    per_rel = per_level // len(RELATIONS)
    rng = random.Random(f"recall-{seed}")
    pools = candidates()
    out: list[Question] = []
    used: set[str] = set()
    for level in LEVEL_FLOOR:
        picked = []
        for rel in RELATIONS:
            pool = [r for r in pools[(level, rel)] if r["subj"] not in used]
            chosen = rng.sample(pool, per_rel)
            used.update(r["subj"] for r in chosen)
            picked += chosen
        for i, r in enumerate(picked):
            out.append(
                Question(
                    id=f"rec-L{level}-{i + 1:03d}",
                    category="recall",
                    kind="answerable",
                    level=level,
                    source="popqa",
                    question=r["text"],
                    answer_type="short_text",
                    gold_answers=r["answers"],
                    meta={"relation": r["prop"], "subject": r["display"], "s_pop": r["s_pop"],
                          "log10_pop": round(math.log10(r["s_pop"]), 2),
                          "popqa_id": r["id"], "s_uri": r["s_uri"]},
                )
            )
    return out
