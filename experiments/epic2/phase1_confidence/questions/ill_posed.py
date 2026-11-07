"""Ill-posed: questions that shouldn't be answered as asked. Built from PopQA, two types.

false_premise  A famous fact with the wrong person swapped in:
               "In what year did Stephen King write Nineteen Eighty-Four?"
               right behaviour: correct_premise (say the premise is wrong, or that it can't
               be confirmed; answering with a year is the failure).
               Subjects come from recall.candidates() (unambiguous, single answer), so the
               level is subject popularity, the same knob as recall: a false premise about a
               famous work should be easy to catch, one about an obscure work is hard.
               The swapped-in person is a well-known creator in the same role who is not
               linked to the subject anywhere in PopQA. PopQA lists one creator per work, so
               an unlisted co-creator can slip through: spot-check.

ambiguous      A title shared by several works with different creators:
               "Who was the director of Phoenix?" (Danny Cannon, Christian Petzold, ...)
               right behaviour: clarify (ask which one, or answer each reading).
               Kept only when the two most popular readings are within MAX_RATIO of each
               other in pageviews; otherwise "the famous one" is a fair answer. Ambiguous
               titles in PopQA are almost all mid-popularity (L2-L4), so no level is set;
               meta has the readings with their pageviews for analysis.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict

from .recall import LEVEL_FLOOR, candidates, popqa_rows
from .schema import Question

PREMISE_TEMPLATES = {
    "director": "In what year did {person} direct {work}?",
    "author": "In what year did {person} write {work}?",
}
AMBIGUOUS_RELATIONS = ("director", "screenwriter", "author", "composer")
MAX_RATIO = 3.0
FAMOUS_CREATORS = 40  # size of the pool of swapped-in people per relation


def _famous_creators(rel: str) -> list[str]:
    """Best-known people who appear as the object of `rel` at least 3 times."""
    pop: dict[str, int] = {}
    n: dict[str, int] = defaultdict(int)
    for r in popqa_rows():
        if r["prop"] == rel:
            pop[r["obj"]] = r["o_pop"]
            n[r["obj"]] += 1
    names = [o for o in pop if n[o] >= 3]
    return sorted(names, key=lambda o: -pop[o])[:FAMOUS_CREATORS]


def _linked_people() -> dict[str, set[str]]:
    """Every object name (and alias) linked to a subject name, across all relations."""
    out: dict[str, set[str]] = defaultdict(set)
    for r in popqa_rows():
        out[r["subj"]].update(a.lower() for a in json.loads(r["possible_answers"]))
    return out


def false_premise(per_level: int, seed: int) -> list[Question]:
    rels = list(PREMISE_TEMPLATES)
    if per_level % len(rels):
        raise ValueError(f"per_level must be a multiple of {len(rels)}")
    rng = random.Random(f"false-premise-{seed}")
    pools, linked = candidates(), _linked_people()
    famous = {rel: _famous_creators(rel) for rel in rels}
    out: list[Question] = []
    for level in LEVEL_FLOOR:
        i = 0
        for rel in rels:
            for r in rng.sample(pools[(level, rel)], per_level // len(rels)):
                true = {a.lower() for a in r["answers"]}
                options = [
                    p for p in famous[rel]
                    if p.lower() not in true | linked[r["subj"]]
                    and p.split()[-1].lower() not in r["display"].lower()
                    and not any(p.split()[-1].lower() in t for t in true)
                ]
                person = rng.choice(options)
                i += 1
                out.append(
                    Question(
                        id=f"ill-fp-L{level}-{i:03d}",
                        category="ill_posed",
                        kind="behaviour",
                        level=level,
                        source="popqa-swap",
                        question=PREMISE_TEMPLATES[rel].format(person=person, work=r["display"]),
                        answer_type="none",
                        gold_answers=[],
                        right_behaviour="correct_premise",
                        meta={"type": "false_premise", "relation": rel, "subject": r["display"],
                              "false_person": person, "true_answer": r["answers"][0],
                              "s_pop": r["s_pop"], "popqa_id": r["id"]},
                    )
                )
    return out


def ambiguous(n: int, seed: int) -> list[Question]:
    if n % len(AMBIGUOUS_RELATIONS):
        raise ValueError(f"n must be a multiple of {len(AMBIGUOUS_RELATIONS)}")
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in popqa_rows():
        if r["prop"] in AMBIGUOUS_RELATIONS:
            groups[(r["subj"], r["prop"])].append(r)
    pools: dict[str, list[tuple[str, list[dict]]]] = defaultdict(list)
    for (subj, rel), rows in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        # One reading per Wikipedia article: PopQA sometimes has several entity ids for the
        # same article ("Celebration (The Game song)" three times), which isn't ambiguity.
        by_title: dict[str, dict] = {}
        for r in rows:
            reading = by_title.setdefault(r["s_wiki_title"], {
                "title": r["s_wiki_title"], "answers": [], "s_pop": 0, "question": r["question"]})
            reading["answers"] += [a for a in json.loads(r["possible_answers"])
                                   if a not in reading["answers"]]
            reading["s_pop"] = max(reading["s_pop"], r["s_pop"])
        readings = sorted(by_title.values(), key=lambda x: -x["s_pop"])
        if len(readings) < 2 or {a.lower() for a in readings[0]["answers"]} & {
            a.lower() for a in readings[1]["answers"]
        }:
            continue  # the two main readings must have different answers
        if readings[0]["s_pop"] > MAX_RATIO * readings[1]["s_pop"]:
            continue
        pools[rel].append((subj, readings))

    rng = random.Random(f"ambiguous-{seed}")
    out: list[Question] = []
    for rel in AMBIGUOUS_RELATIONS:
        for subj, readings in rng.sample(pools[rel], n // len(AMBIGUOUS_RELATIONS)):
            question = readings[0].pop("question")
            for x in readings[1:]:
                x.pop("question")
            out.append(
                Question(
                    id=f"ill-amb-{len(out) + 1:03d}",
                    category="ill_posed",
                    kind="behaviour",
                    level=None,
                    source="popqa-homonyms",
                    question=question,
                    answer_type="none",
                    gold_answers=[],
                    right_behaviour="clarify",
                    meta={"type": "ambiguous", "relation": rel, "subject": subj,
                          "readings": readings,
                          "pop_ratio": round(readings[0]["s_pop"] / readings[1]["s_pop"], 2)},
                )
            )
    return out


def build(seed: int, per_level: int = 8, n_ambiguous: int = 40) -> list[Question]:
    return false_premise(per_level, seed) + ambiguous(n_ambiguous, seed)
