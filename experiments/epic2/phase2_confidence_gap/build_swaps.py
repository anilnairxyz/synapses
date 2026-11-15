"""E2 / E3 inputs: the answers to put in the model's mouth (Mac, no model calls).

For each Phase 1B config (qwen3-8b, qwen3-1.7b, qwen3-8b-base) and, for E2b, claude-haiku
and qwen3-8b-think (test questions only), and each answerable test question, three given
answers:

  own      the model's own Phase 1 plain answer (P), verbatim; skipped if it abstained or
           is empty. Correct or not per Phase 1's judged label.
  correct  the gold answer (recall: the model's own wording when its own answer was right,
           since a judged-right answer may use another alias).
  wrong    a plausible answer that is certainly wrong and differs from the model's own:
             computation  one middle digit of the product changed (length and leading
                          digits kept, like the models' real errors)
             perception   count +-1; nth letter: a neighbouring letter of the word; reverse:
                          two adjacent letters of the reversed word swapped
             recall       another config's Phase 1 answer to the same question that was
                          graded wrong (shortest clean one); else the gold answer of another
                          question with the same relation (another director, author, ...)

Injected answers (correct, wrong) are written in the model's own format: where the model
answered in a sentence ("The 4th letter of the word "tumor" is "r"."), the value in that
sentence is replaced; where it answered bare, or the value cannot be found, the bare value
is used (``fmt`` records which). This guards against the model noticing an insertion by
its style (Wang et al. 2026). A substituted sentence that Phase 1's grader would read
differently from how it was built (the letter "a" in a sentence reads as the article) falls
back to the bare value. Check: when the own answer is right, the correct answer's
text should equal it.

For E3 the training questions' own answers are included as well (source own, split train).

Writes data/epic2/confidence/phase2/swaps.jsonl (committed with git add -f, like Phase 1's
question files), one row per (config, split, qid, source):
  config, split, qid, category, level, family, source, text, value, correct (0/1), fmt

    uv run --extra evals python experiments/epic2/phase2_confidence_gap/build_swaps.py
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter

import pandas as pd

import phase1

CONFIGS = ("qwen3-8b", "qwen3-1.7b", "qwen3-8b-base")
E2B = ("claude-haiku", "qwen3-8b-think")  # beyond Qwen / thinking: test questions, verbal only
KINDS = ("recall", "computation", "perception")
MAX_SENTENCE = 160  # longer own answers (rambling, repeated) are not used as a format
OUT = phase1.DATA / "swaps.jsonl"


def rng_for(*key) -> random.Random:
    return random.Random("|".join(map(str, key)))


# ----------------------------------------------------------------------------- wrong answers


def wrong_computation(gold: str, own: str | None, rng: random.Random) -> str:
    """Change one middle digit (never the first or last unless the number is that short)."""
    pos = list(range(1, len(gold) - 1)) or list(range(len(gold)))
    for _ in range(100):
        i = rng.choice(pos)
        d = rng.choice([c for c in "0123456789" if c != gold[i] and not (i == 0 and c == "0")])
        w = gold[:i] + d + gold[i + 1:]
        if w != own:
            return w
    raise ValueError(f"no wrong answer for {gold}")


def nth_index(question: str) -> int:
    m = re.search(r"the (\d+)(?:st|nd|rd|th) letter", question)
    if not m:
        raise ValueError(question)
    return int(m.group(1)) - 1


def wrong_perception(q: dict, own: str | None, rng: random.Random) -> str:
    gold = q["gold_answers"][0].lower()
    fam, word = q["meta"]["family"], q["meta"]["word"].lower()
    if fam == "count":
        c = int(gold)
        options = [x for x in (c - 1, c + 1) if x >= 0 and str(x) != own]
        return str(rng.choice(options) if options else c + 2)
    if fam == "nth":
        i = nth_index(q["question"])
        near = [word[j] for j in (i - 1, i + 1) if 0 <= j < len(word)]
        rest = [ch for ch in word if ch not in near]
        for cands in (near, rest):
            cands = [ch for ch in dict.fromkeys(cands) if ch != gold and ch != own]
            if cands:
                return rng.choice(cands)
        return next(ch for ch in "etaoinshrdlu" if ch not in (gold, own))
    if fam == "reverse":
        spots = [i for i in range(len(gold) - 1) if gold[i] != gold[i + 1]]
        rng.shuffle(spots)
        for i in spots:
            w = gold[:i] + gold[i + 1] + gold[i] + gold[i + 2:]
            if w != own:
                return w
        raise ValueError(f"no wrong reversal for {gold}")
    raise ValueError(fam)


_LEAD = re.compile(r".*\b(?:is|was|are|were|by)\s+", re.S)


def recall_value(text: str) -> str:
    """The name in a recall answer: the bare answer, or what follows the last "is"/"was"/"by"
    in a one-sentence answer; markdown and trailing punctuation removed."""
    t = text.strip().replace("*", "").strip()
    if len(t.split()) > 6:
        t = _LEAD.sub("", t)
    return t.strip().strip("\"'").rstrip(".").strip()


_CONNECTIVES = {"of", "de", "da", "del", "der", "di", "du", "la", "le", "van", "von", "the", "and", "y", "bin", "al"}


def clean_name(v: str) -> bool:
    """A name-like answer: 1 to 6 words, each capitalised apart from small connectives."""
    ct = phase1.ct
    words = re.sub(r"\([^)]*\)", "", v).split()
    return (0 < len(words) <= 6 and not ct.looks_abstain(v) and not ct.is_refusal(v)
            and all(w[0].isupper() or w.lower() in _CONNECTIVES for w in words)
            and not re.search(r"\b(not|unknown|unclear|none|no one|sorry)\b", v.lower()))


def wrong_recall(q: dict, own: str | None, others: list[str], pool: list[str],
                 rng: random.Random) -> tuple[str, str]:
    """(value, origin): another model's graded-wrong answer, else another question's gold."""
    ct = phase1.ct
    bad = {ct.normalise(g) for g in q["gold_answers"]} | ({ct.normalise(own)} if own else set())
    cands = sorted({v for v in map(recall_value, others) if clean_name(v) and ct.normalise(v) not in bad},
                   key=lambda v: (len(v), v))
    if cands:
        return cands[0], "other_model"
    pool = [v for v in pool if ct.normalise(v) not in bad]
    return rng.choice(pool), "other_question"


# ----------------------------------------------------------------------------- format


_COUNT_WORDS = {1: "once", 2: "twice", 3: "three times"}
_COUNT_PHRASE = re.compile(r"\b(?:once|twice|thrice|\d+ times?)\b", re.I)


def count_phrase(own_phrase: str, n: int) -> str:
    """n in the style of the model's own phrase ("twice" / "2 times" / "Once")."""
    words = own_phrase.lower() in ("once", "twice", "thrice")
    out = _COUNT_WORDS.get(n, f"{n} times") if words else f"{n} time{'s' if n != 1 else ''}"
    return out.capitalize() if own_phrase[0].isupper() else out


def bare(q: dict, own_text: str | None, value: str) -> str:
    """The bare value in the own answer's surface style: thousands separators, final stop."""
    t = (own_text or "").strip()
    if q["answer_type"] == "integer" and re.fullmatch(r"-?\d{1,3}(,\d{3})+", t) and value.isdigit():
        value = f"{int(value):,}"
    if t.endswith(".") and not value.endswith("."):
        value += "."
    return value


def render(q: dict, own_text: str | None, value: str, base: bool) -> tuple[str, str]:
    """``value`` written in the format of the model's own answer: (text, fmt)."""
    if not own_text or len(own_text) > MAX_SENTENCE or "\n" in own_text.strip():
        return value, "bare"
    t = own_text.strip()
    if base:
        return bare(q, t, value), "bare"
    cat, kind = q["category"], q["answer_type"]
    if cat == "recall":
        own_v = recall_value(t)
        if own_v == t.replace("*", "").strip().strip("\"'").rstrip(".").strip():
            return bare(q, t, value), "bare"  # the own answer was bare already
        i = t.rfind(own_v)
        return (t[:i] + value + t[i + len(own_v):], "substituted") if i >= 0 else (value, "bare")
    if q["meta"].get("family") == "count" and _COUNT_PHRASE.search(t) and value.isdigit():
        n = int(value)
        m = list(_COUNT_PHRASE.finditer(t))[-1]
        new = t[:m.start()] + count_phrase(m.group(0), n) + t[m.end():]
        return new, ("substituted" if len(t) > len(m.group(0)) + 1 else "bare")
    own_c = phase1.ct.canonical(q, t)
    if own_c is None:
        return value, "bare"
    if kind == "integer":
        variants = [f"{int(own_c):,}", own_c] if own_c.lstrip("-").isdigit() else [own_c]
        for v in variants:
            m = list(re.finditer(rf"(?<![\d,]){re.escape(v)}(?![\d,])", t))
            if m:
                new = f"{int(value):,}" if "," in v else value
                s, e = m[-1].span()
                return (t[:s] + new + t[e:], "substituted") if t != v else (bare(q, t, value), "bare")
        return value, "bare"
    if kind == "letter":
        m = list(re.finditer(rf"(?<![A-Za-z]){re.escape(own_c)}(?![A-Za-z])", t, re.I))
        if not m:
            return value, "bare"
        s, e = m[-1].span()
        new = value.upper() if t[s:e].isupper() else value
        return (t[:s] + new + t[e:], "substituted") if len(t) > 1 else (bare(q, t, new), "bare")
    if kind == "exact_string":
        i = t.lower().rfind(own_c)
        if i < 0:
            return value, "bare"
        new = value.upper() if t[i:i + len(own_c)].isupper() else value
        return ((t[:i] + new + t[i + len(own_c):], "substituted") if len(t) > len(own_c) + 2
                else (bare(q, t, new), "bare"))
    return value, "bare"


# ----------------------------------------------------------------------------- build


def load_questions(name: str) -> dict[str, dict]:
    path = phase1.PHASE1_QUESTIONS.parent / name
    return {q["id"]: q for q in map(json.loads, path.read_text().splitlines())}


def own_answers(config: str, split: str) -> pd.DataFrame:
    """qid, answer, label for the config's plain answers (P) on the answerable questions."""
    if split == "test":
        r = pd.read_parquet(phase1.PHASE1_RESULTS / "rows.parquet")
        r = r[(r.config == config) & (r.method == "P") & r.kind.isin(KINDS)]
        return r[["qid", "answer", "label"]].reset_index(drop=True)
    ans = pd.DataFrame([{"qid": a["qid"], "answer": a["answer"]}
                        for a in phase1.extract.logged_answers(config, "train")])
    lab = phase1.probe.labels(config, "train")[["qid", "label", "category"]]
    d = ans.merge(lab, on="qid", how="inner", validate="one_to_one")
    return d[d.category.isin(KINDS)][["qid", "answer", "label"]].reset_index(drop=True)


def build() -> pd.DataFrame:
    test_q, train_q = load_questions("questions.jsonl"), load_questions("train_questions.jsonl")
    rows_all = pd.read_parquet(phase1.PHASE1_RESULTS / "rows.parquet")
    p_all = rows_all[(rows_all.method == "P") & (rows_all.label == "INCORRECT")]
    wrong_by_q = p_all.groupby("qid").apply(lambda g: list(zip(g.config, g.answer)), include_groups=False)
    gold_by_rel: dict[str, list[str]] = {}
    for q in test_q.values():
        if q["category"] == "recall":
            gold_by_rel.setdefault(q["meta"]["relation"], []).append(q["gold_answers"][0])

    out = []
    for config in (*CONFIGS, *E2B):
        base = config.endswith("-base")
        splits = (("test", test_q),) if config in E2B else (("test", test_q), ("train", train_q))
        for split, qs in splits:
            own = own_answers(config, split).set_index("qid")
            for qid, q in qs.items():
                if q["category"] not in KINDS:
                    continue
                key = {"config": config, "split": split, "qid": qid, "category": q["category"],
                       "level": q["level"], "family": q["meta"].get("family") or q["meta"].get("relation")}
                own_text = own.answer.get(qid)
                own_label = own.label.get(qid)
                has_own = own_label in ("CORRECT", "INCORRECT") and bool((own_text or "").strip())
                if has_own:
                    out.append(key | {"source": "own", "text": own_text.strip(), "value": None,
                                      "correct": int(own_label == "CORRECT"), "fmt": "own"})
                if split == "train":
                    continue
                fmt_text = own_text if has_own else None
                gold, gold_origin = q["gold_answers"][0], "gold"
                if q["category"] == "recall" and has_own and own_label == "CORRECT":
                    gold, gold_origin = recall_value(own_text), "own_wording"  # judged right
                own_c = None
                if has_own and q["category"] != "recall":
                    own_c = phase1.ct.canonical(q, own_text)
                rng = rng_for(config, qid)
                if q["category"] == "computation":
                    wrong, origin = wrong_computation(gold, own_c, rng), "constructed"
                elif q["category"] == "perception":
                    wrong, origin = wrong_perception(q, own_c, rng), "constructed"
                else:
                    others = [a for c, a in wrong_by_q.get(qid, []) if c != config]
                    pool = [g for g in gold_by_rel[q["meta"]["relation"]] if g != gold]
                    wrong, origin = wrong_recall(q, recall_value(own_text) if has_own else None,
                                                 others, pool, rng)
                for source, value, ok in (("correct", gold, 1), ("wrong", wrong, 0)):
                    text, fmt = render(q, fmt_text, value, base)
                    if (q["category"] != "recall"
                            and (phase1.ct.grade_fixed(q, text) == "CORRECT") != bool(ok)):
                        text, fmt = value, "bare"  # e.g. the letter "a" in a sentence reads as the article
                    out.append(key | {"source": source, "text": text, "value": value, "correct": ok,
                                      "fmt": fmt, "origin": origin if source == "wrong" else gold_origin})
    return pd.DataFrame(out)


def check(d: pd.DataFrame) -> None:
    """Every constructed answer graded the way it was built, by Phase 1's own grader."""
    ct = phase1.ct
    qs = load_questions("questions.jsonl")
    inj = d[(d.split == "test") & (d.source != "own")]
    bad = []
    for r in inj.itertuples():
        q = qs[r.qid]
        if q["category"] == "recall":
            if r.origin == "own_wording":  # right by Phase 1's judge
                continue
            hit = ct.normalise(r.value) in {ct.normalise(g) for g in q["gold_answers"]}
            if hit != bool(r.correct):
                bad.append(r)
        elif (ct.grade_fixed(q, r.text) == "CORRECT") != bool(r.correct):
            bad.append(r)
    if bad:
        for r in bad[:10]:
            print("  MISGRADED", r.config, r.qid, r.source, repr(r.text))
        raise SystemExit(f"{len(bad)} injected answers grade differently from how they were built")
    own = d[(d.split == "test") & (d.source == "own")].set_index(["config", "qid"])
    cor = inj[inj.source == "correct"].set_index(["config", "qid"])
    both = own.join(cor, lsuffix="_own", rsuffix="_cor", how="inner")
    right = both[both.correct_own == 1]
    same = (right.text_own == right.text_cor).mean()
    print(f"  injected answers all grade as built ({len(inj)}); correct text equals the own "
          f"answer on {same:.0%} of {len(right)} questions the model got right")


def main() -> None:
    d = build()
    check(d)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_json(OUT, orient="records", lines=True)
    print(f"wrote {OUT} ({len(d)} rows)")
    t = d[d.split == "test"]
    print(t.groupby(["config", "category", "source"]).size().unstack().to_string())
    print(Counter(t[t.source != "own"].fmt))
    print(t[t.source == "wrong"].groupby("category").origin.value_counts().to_string())


if __name__ == "__main__":
    main()
