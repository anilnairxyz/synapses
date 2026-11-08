"""Measure the labelling pipeline (code rules, then the judge) against the hand-labelled gold set.

    # 4090, with the judge served by vLLM:
    uv run --extra evals python experiments/epic2/phase1_confidence/check_judge.py \
        --model openai-api/local/Qwen/Qwen3-14B-FP8
    # rules only (no judge; anything the rules don't decide counts as unlabelled):
    uv run --extra evals python experiments/epic2/phase1_confidence/check_judge.py

Prints agreement per kind (split into rule-decided and judge-decided items) and every
disagreement. --out saves the per-item results as jsonl.
"""

import argparse
import asyncio
import collections
import json
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai.model import GenerateConfig, get_model

import confidence_task as ct

HERE = Path(__file__).resolve().parent
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


async def run(model_name: str | None, think: bool) -> list[dict]:
    qs = {}
    for line in ct.REPO_QUESTIONS.read_text().splitlines():
        q = json.loads(line)
        qs[q["id"]] = q
    gold = [json.loads(line) for line in (HERE / "checks" / "gold_set.jsonl").read_text().splitlines()]
    grader = None
    if model_name:
        config = GenerateConfig() if think else GenerateConfig(extra_body=NO_THINK)
        grader = get_model(model_name, config=config)

    async def one(g: dict) -> dict:
        q, kind, a = qs[g["id"]], g["kind"], g["response"]
        lab, how, raw = ct.rule_label(q, kind, a), "rule", ""
        if lab is None and grader:
            (lab, raw), how = await ct.judge_label(grader, q, kind, a), "judge"
        return {"gid": g["gid"], "kind": kind, "gold": g["label"], "got": lab, "how": how,
                "raw": raw, "response": a}

    return await asyncio.gather(*[one(g) for g in gold])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="judge model; omit to test the code rules only")
    ap.add_argument("--think", action="store_true", help="leave Qwen thinking on")
    ap.add_argument("--out", help="save per-item results here (jsonl)")
    args = ap.parse_args()
    load_dotenv()
    res = asyncio.run(run(args.model, args.think))

    print(f"judge: {args.model or '(none)'}{' thinking on' if args.think else ''}")
    by = collections.defaultdict(lambda: collections.Counter())
    for r in res:
        c = by[r["kind"]]
        c["n"] += 1
        c[f"{r['how']}_n"] += 1
        c[f"{r['how']}_ok"] += r["got"] == r["gold"]
    tot = sum(c["rule_ok"] + c["judge_ok"] for c in by.values())
    for kind, c in by.items():
        ok = c["rule_ok"] + c["judge_ok"]
        print(f"  {kind:14} {ok:3}/{c['n']:<3}  rules {c['rule_ok']}/{c['rule_n']}"
              f"  judge {c['judge_ok']}/{c['judge_n']}")
    print(f"  {'all':14} {tot:3}/{len(res)}")
    bad = [r for r in res if r["got"] != r["gold"] and (r["got"] or args.model)]
    if bad:
        print("\ndisagreements:")
        for r in bad:
            print(f"  {r['gid']} [{r['how']}] gold={r['gold']} got={r['got']}  "
                  f"{' '.join(r['response'].split())[:110]}")
    if args.out:
        Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in res))


if __name__ == "__main__":
    main()
