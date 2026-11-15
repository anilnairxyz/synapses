"""E2b follow-up (Mac): do the replies contain written working, and does that explain the
ownership effect? Reads the full swap logs of every instruct config.

  working   share of replies with more than 40 characters after removing any thinking
            trace (a bare {"confidence": x} is about 20), per config x kind x attribution
  haiku     Claude Haiku's paired separation (correct vs wrong injected answer) split by
            whether it wrote working for both answers of the pair, for neither, or mixed

Writes results/e2_working.csv and results/e2_working_haiku.csv.

    uv run --extra evals python experiments/epic2/phase2_confidence_gap/analyze_replies.py
"""

from __future__ import annotations

import glob
import re

import pandas as pd

import analyze_swaps as an
import phase1

CONFIGS = ("qwen3-8b", "qwen3-1.7b", "qwen3-8b-think", "claude-haiku", "gpt-4o", "grok-4-1-fast", "gpt-5-nano")
WORKING_CHARS = 40


def replies(config: str) -> pd.DataFrame:
    from inspect_ai.log import read_eval_log

    f = sorted(glob.glob(str(phase1.RUNS / an.SWAP_RUNS.get(config, "4090/full") / config / "*.eval")))[0]
    rows = []
    for s in read_eval_log(f).samples:
        m = s.scores["confidence_record"].metadata
        txt = re.sub(r"<think>.*?</think>", "", s.output.completion or "", flags=re.S).strip()
        rows.append({"config": config, "qid": m["qid"], "kind": m.get("category") or (s.metadata or {}).get("category"), "source": m["source"],
                     "attribution": m["attribution"], "confidence": s.scores["confidence_record"].value["confidence"],
                     "works": len(txt) > WORKING_CHARS})
    return pd.DataFrame(rows)


def haiku_split(d: pd.DataFrame) -> pd.DataFrame:
    out = []
    d = d[d.source.isin(["correct", "wrong"])]
    for (kind, att), g in d.groupby(["kind", "attribution"]):
        p = g.pivot_table(index="qid", columns="source", values=["confidence", "works"], aggfunc="first").dropna()
        wc, ww = p[("works", "correct")].astype(bool), p[("works", "wrong")].astype(bool)
        for name, mask in (("all", wc | ~wc), ("both", wc & ww), ("neither", ~wc & ~ww), ("mixed", wc ^ ww)):
            x = p[mask]
            win = an.win_rate(x[("confidence", "correct")].to_numpy(float), x[("confidence", "wrong")].to_numpy(float)) \
                if len(x) else float("nan")
            out.append({"kind": kind, "attribution": att, "pairs": name, "n": len(x), "win": win})
    return pd.DataFrame(out)


def main() -> None:
    d = pd.concat([replies(c) for c in CONFIGS], ignore_index=True)
    w = d.groupby(["config", "kind", "attribution"]).works.mean().unstack("attribution").reset_index()
    w.to_csv(phase1.RESULTS / "e2_working.csv", index=False)
    h = haiku_split(d[d.config == "claude-haiku"])
    h.to_csv(phase1.RESULTS / "e2_working_haiku.csv", index=False)
    with pd.option_context("display.width", 200):
        print(w.round(2).to_string(index=False))
        print(h.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
