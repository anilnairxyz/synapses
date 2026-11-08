"""Gold set for validating the judge: sample distinct (question, response) pairs that need a judge.

Draws from the 4090 full-run logs, per judge kind, round-robin over (current judge label, config)
so every label and model is represented, and writes `checks/gold_set.jsonl` with empty
`label` / `reason` fields to be filled by hand (Claude drafts, Anil reviews). Existing labels are
kept if the file is regenerated with the same seed: items are matched on (id, response).

    uv run --extra evals python experiments/epic2/phase1_confidence/make_goldset.py
"""

import argparse
import collections
import glob
import json
import random
from pathlib import Path

from inspect_ai.log import read_eval_log

import confidence_task as ct

HERE = Path(__file__).resolve().parent
SIZES = {"false_premise": 35, "ambiguous": 35, "unknowable": 25, "recall": 25}
OUT = HERE / "checks" / "gold_set.jsonl"


def needs_judge(q: dict, kind: str, a: str) -> bool:
    """True if the current code would send this response to the judge."""
    if kind == "recall":
        return ct.grade_recall_exact(q, a) is None
    if ct.normalise(a) in ("i dont know", "idk"):
        return False
    return not (kind == "false_premise" and ct._BARE_YEAR.match(a))


def background(q: dict, kind: str) -> str:
    m = q["meta"]
    if kind == "recall":
        return "Gold: " + " | ".join(q["gold_answers"][:6])
    if kind == "false_premise":
        return f"{m['false_person']} did not create it; {m['true_answer']} did."
    if kind == "ambiguous":
        return "Works sharing the title: " + "; ".join(
            f"{r['title']} ({', '.join(r['answers'][:2])})" for r in m["readings"][:4])
    return f"Unknowable ({m.get('type')})"


def write_review() -> None:
    """checks/gold_set_review.md: one table per kind for reading the labels; overrides are
    written into the 'You' column and copied back into the jsonl."""
    rows = [json.loads(line) for line in OUT.read_text().splitlines()]
    cell = lambda s: " ".join(str(s).split()).replace("|", "/")  # noqa: E731
    lines = ["# Gold set review", "",
             "Draft labels by Claude. Write a different label in **You** where you disagree "
             "(leave it blank to accept). BORDERLINE items need a decision either way. "
             "The 14B column is the old judge, for reference.", ""]
    for kind in SIZES:
        lines += [f"## {kind}", "", "| id | Question | Response | Draft | Reason | 14B | You |",
                  "|---|---|---|---|---|---|---|"]
        for r in (r for r in rows if r["kind"] == kind):
            lines.append(f"| {r['gid']} | {cell(r['question'])}<br>*{cell(r['background'])[:150]}* | "
                         f"{cell(r['response'])[:300]} | **{r['label']}** | {cell(r['reason'])} | "
                         f"{r['judge_14b']} | |")
        lines.append("")
    (OUT.parent / "gold_set_review.md").write_text("\n".join(lines))
    print(f"wrote {OUT.parent / 'gold_set_review.md'}")


HELDOUT = HERE / "checks" / "heldout_set.jsonl"


def write_heldout(per_kind: int, seed: int, runs: tuple[str, ...] = ("4090/full",), out: Path = HELDOUT,
                  balance: bool = False) -> None:
    """Held-out check of the judge as it ran: `per_kind` judge decisions per kind drawn from the
    re-scored logs under runs/<each of runs>/<config>/, excluding anything in the gold set or an earlier
    held-out set. Uniform over decisions by default (so frequent responses weigh more, as in the
    results); with `balance`, round-robin over configs so each model gets the same share. The
    logged label is stored as `judge` but must not be looked at while labelling; the hand label
    goes in `label`."""
    seen_before = {(r["id"], ct.normalise(r["response"]))
                   for path in [OUT, *sorted(OUT.parent.glob("heldout*.jsonl"))] if path != out
                   for r in map(json.loads, path.read_text().splitlines())}
    calls_by_kind: dict[str, list[dict]] = collections.defaultdict(list)
    for f in sorted(f for r in runs for f in glob.glob(str(HERE / "runs" / r / "*/*.eval"))):
        config = Path(f).parent.name
        for s in read_eval_log(f).samples:
            q = s.metadata["q"]
            kind = ct.judge_kind(q)
            sc = (s.scores or {}).get("confidence_scorer")
            if kind is None or sc is None:
                continue
            calls = s.metadata["calls"]
            for key in sc.metadata.get("judge_raw") or {}:
                if key.startswith("B"):
                    continue
                a = (calls["C"][key[1:]] if key.startswith("C") else calls[key]).get("answer", "")
                if (s.id, ct.normalise(a)) in seen_before:
                    continue
                calls_by_kind[kind].append({"kind": kind, "id": s.id, "question": q["question"],
                                            "background": background(q, kind), "response": a,
                                            "judge": sc.metadata["labels"][key],
                                            "seen_in": f"{config}:{key}"})
    rng = random.Random(seed)
    rows = []
    for kind in SIZES:
        pool, seen, picked = calls_by_kind[kind], set(), []
        rng.shuffle(pool)
        if balance:  # interleave configs: c1, c2, c3, c4, c1, ...
            by_config = collections.defaultdict(list)
            for d in pool:
                by_config[d["seen_in"].split(":")[0]].append(d)
            queues = [by_config[c] for c in sorted(by_config)]
            pool = [q.pop(0) for _ in range(max(map(len, queues), default=0)) for q in queues if q]
        for d in pool:
            k = (d["id"], ct.normalise(d["response"]))
            if k not in seen and len(picked) < per_kind:
                seen.add(k)
                picked.append(d)
        rows += [{"gid": f"h{kind[:3]}-{i:02d}", **d, "label": "", "reason": ""}
                 for i, d in enumerate(picked, 1)]
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {len(rows)} items to {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--review", action="store_true", help="only write the review sheet")
    ap.add_argument("--heldout", type=int, metavar="N", help="write N held-out items per kind")
    ap.add_argument("--runs", nargs="+", default=["4090/full"], help="held-out: logs under runs/<RUN>/<config>/")
    ap.add_argument("--out", type=Path, default=HELDOUT, help="held-out: output file")
    ap.add_argument("--balance", action="store_true", help="held-out: same share per config")
    args = ap.parse_args()
    if args.review:
        write_review()
        return
    if args.heldout:
        write_heldout(args.heldout, args.seed, tuple(args.runs), args.out, args.balance)
        return

    pool: dict[str, dict[tuple, dict]] = collections.defaultdict(dict)  # kind -> key -> item
    for f in sorted(glob.glob(str(HERE / "runs/4090/full/*/*.eval"))):
        config = Path(f).parent.name
        for s in read_eval_log(f).samples:
            q = s.metadata["q"]
            kind = ct.judge_kind(q)
            sc = (s.scores or {}).get("confidence_scorer")
            if kind is None or sc is None:
                continue
            calls, labels = s.metadata["calls"], sc.metadata["labels"]
            named = {"P": calls["P"]} | {k: calls[k] for k in ("A", "D") if k in calls}
            named |= {f"C{t}": r for t, r in calls.get("C", {}).items()}
            for key, rec in named.items():
                a = rec.get("answer", "")
                if "error" in rec or not rec.get("parse_ok") or not needs_judge(q, kind, a):
                    continue
                k = (s.id, ct.normalise(a))
                item = pool[kind].setdefault(k, {
                    "kind": kind, "id": s.id, "question": q["question"],
                    "background": background(q, kind), "response": a,
                    "judge_14b": labels[key], "seen_in": []})
                item["seen_in"].append(f"{config}:{key}")

    old = {}
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            old[(r["id"], ct.normalise(r["response"]))] = r

    rng = random.Random(args.seed)
    rows = []
    for kind, n in SIZES.items():
        strata: dict[tuple, list[dict]] = collections.defaultdict(list)
        for item in pool[kind].values():
            strata[(item["judge_14b"], item["seen_in"][0].split(":")[0])].append(item)
        for v in strata.values():
            v.sort(key=lambda d: (d["id"], d["response"]))
            rng.shuffle(v)
        picked: list[dict] = []
        while len(picked) < n and any(strata.values()):
            for key in sorted(strata):
                if strata[key] and len(picked) < n:
                    picked.append(strata[key].pop())
        for i, item in enumerate(sorted(picked, key=lambda d: d["id"]), 1):
            prev = old.get((item["id"], ct.normalise(item["response"])), {})
            rows.append({"gid": f"{kind[:3]}-{i:02d}", **item,
                         "label": prev.get("label", ""), "reason": prev.get("reason", ""),
                         "reviewed": prev.get("reviewed", False)})
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {len(rows)} items to {OUT}")


if __name__ == "__main__":
    main()
