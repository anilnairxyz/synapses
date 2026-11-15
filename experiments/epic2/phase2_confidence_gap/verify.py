"""Check every number quoted in the report against the results files (Mac).

Each claim names a results file, the rows it is about, a column, and the value (or range)
the report states. A point claim passes when the file's value rounds to the stated value
(tolerance 0.005 for two decimals); a range claim passes when every selected value lies
inside the stated range (same tolerance) and the range's ends are attained (so "0.50 to
0.55" is neither wider nor narrower than the data). An "all" claim checks a condition on
every selected row.

With --rerun, the analysis scripts that can run on the Mac are run first, so the check is
against freshly recomputed files (E4 Part A's directions.py needs the hidden states on the
PC and is not rerun; its CSV is checked as committed).

    uv run --extra evals python experiments/epic2/phase2_confidence_gap/verify.py
    uv run --extra evals python experiments/epic2/phase2_confidence_gap/verify.py --rerun

Exit status 1 if any claim fails. Claims are grouped by report section; add one for every
number in the report text.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np
import pandas as pd

import phase1

TOL = 0.005
RERUN = ("analyze_swaps.py", "analyze_replies.py", "analyze_steer.py", "analyze_lora.py")

SEP = "e2_separation.csv"
MEANS = "e2_means.csv"
E3 = "e3_auroc.csv"


def point(label, file, where, col, value, tol=TOL):
    return {"type": "point", "label": label, "file": file, "where": where, "col": col, "value": value, "tol": tol}


def span(label, file, where, col, lo, hi, tol=TOL):
    return {"type": "range", "label": label, "file": file, "where": where, "col": col, "lo": lo, "hi": hi, "tol": tol}


def every(label, file, where, col, test, text):
    return {"type": "all", "label": label, "file": file, "where": where, "col": col, "test": test, "text": text}


def sep(config, kind, readout):
    return {"config": config, "kind": kind, "readout": readout}


def own_vs_user(config, kind, source):
    return {"config": config, "kind": kind, "readout": "own-vs-user verbal", "source": source}


KINDS3 = ["recall", "computation", "perception"]
CP = ["computation", "perception"]

CLAIMS = [
    # ---- 4.1 Without reasoning, no checking in the model's own turn
    point("Qwen3-8B 'your answer' recall", SEP, sep("qwen3-8b", "recall", "verbal_self"), "win", 0.55),
    point("  interval low", SEP, sep("qwen3-8b", "recall", "verbal_self"), "win_lo", 0.51),
    point("  interval high", SEP, sep("qwen3-8b", "recall", "verbal_self"), "win_hi", 0.60),
    point("Qwen3-8B 'your answer' computation", SEP, sep("qwen3-8b", "computation", "verbal_self"), "win", 0.51),
    point("Qwen3-8B 'your answer' perception", SEP, sep("qwen3-8b", "perception", "verbal_self"), "win", 0.50),
    every("Qwen3-8B: every planted wrong answer at >= 0.9 ('your answer')", MEANS,
          {"config": "qwen3-8b", "readout": "verbal_self", "kind": KINDS3}, "wrong_ge_0.9", lambda x: x == 1.0, "== 1.0"),
    span("Qwen3-1.7B 'your answer', all kinds", SEP, {"config": "qwen3-1.7b", "readout": "verbal_self"}, "win", 0.51, 0.52),
    every("Grok: 'your answer' computation and perception exactly 0.50", SEP,
          {"config": "grok-4-1-fast", "readout": "verbal_self", "kind": CP}, "win", lambda x: np.isclose(x, 0.5), "== 0.5"),
    every("Grok: confidence 1.00 on every own-turn computation/perception answer", MEANS,
          {"config": "grok-4-1-fast", "readout": "verbal_self", "kind": CP}, "wrong", lambda x: x == 1.0, "wrong mean == 1.0"),
    point("GPT-4o planted wrong product, mean confidence ('your answer')", MEANS,
          {"config": "gpt-4o", "kind": "computation", "readout": "verbal_self"}, "wrong", 0.97),
    point("GPT-4o planted wrong products at >= 0.9", MEANS,
          {"config": "gpt-4o", "kind": "computation", "readout": "verbal_self"}, "wrong_ge_0.9", 0.97),
    point("GPT-4o 'your answer' computation", SEP, sep("gpt-4o", "computation", "verbal_self"), "win", 0.53),
    point("GPT-4o 'your answer' recall", SEP, sep("gpt-4o", "recall", "verbal_self"), "win", 0.93),
    span("Abstract: Qwen3-8B/1.7B and Grok, computation and perception, 'your answer'", SEP,
         {"config": ["qwen3-8b", "qwen3-1.7b", "grok-4-1-fast"], "kind": CP, "readout": "verbal_self"}, "win", 0.50, 0.51),
    # ---- 4.2 Checking, not ownership
    point("Qwen3-8B thinking 'your answer' computation", SEP, sep("qwen3-8b-think", "computation", "verbal_self"), "win", 1.00),
    point("Qwen3-8B thinking 'your answer' perception", SEP, sep("qwen3-8b-think", "perception", "verbal_self"), "win", 0.88),
    point("GPT-5 nano 'your answer' computation", SEP, sep("gpt-5-nano", "computation", "verbal_self"), "win", 0.98),
    point("GPT-5 nano 'your answer' perception", SEP, sep("gpt-5-nano", "perception", "verbal_self"), "win", 0.95),
    span("Reasoning models, computation and perception, both framings", SEP,
         {"config": ["qwen3-8b-think", "gpt-5-nano"], "kind": CP, "readout": ["verbal_self", "verbal_user"]}, "win", 0.88, 1.00),
    point("Haiku 'your answer' computation", SEP, sep("claude-haiku", "computation", "verbal_self"), "win", 0.58),
    point("Haiku 'your answer' perception", SEP, sep("claude-haiku", "perception", "verbal_self"), "win", 0.60),
    point("Haiku 'your answer' recall", SEP, sep("claude-haiku", "recall", "verbal_self"), "win", 0.79),
    point("Haiku 'my answer' computation", SEP, sep("claude-haiku", "computation", "verbal_user"), "win", 0.99),
    point("Haiku 'my answer' perception", SEP, sep("claude-haiku", "perception", "verbal_user"), "win", 0.86),
    point("Haiku 'my answer' recall", SEP, sep("claude-haiku", "recall", "verbal_user"), "win", 0.85),
    point("Haiku writes working, computation, 'my answer'", "e2_working.csv",
          {"config": "claude-haiku", "kind": "computation"}, "user", 0.97),
    point("Haiku writes working, computation, 'your answer'", "e2_working.csv",
          {"config": "claude-haiku", "kind": "computation"}, "self", 0.18),
    every("No other model writes working", "e2_working.csv",
          {"config": ["qwen3-8b", "qwen3-1.7b", "qwen3-8b-think", "gpt-4o", "grok-4-1-fast", "gpt-5-nano"]}, "user",
          lambda x: x == 0, "== 0"),
    point("Haiku, working for both answers, computation 'my answer': separation", "e2_working_haiku.csv",
          {"kind": "computation", "attribution": "user", "pairs": "both"}, "win", 1.00),
    point("  pairs", "e2_working_haiku.csv", {"kind": "computation", "attribution": "user", "pairs": "both"}, "n", 144, 0),
    point("Haiku, no working, computation 'your answer': separation", "e2_working_haiku.csv",
          {"kind": "computation", "attribution": "self", "pairs": "neither"}, "win", 0.45),
    point("  pairs", "e2_working_haiku.csv", {"kind": "computation", "attribution": "self", "pairs": "neither"}, "n", 107, 0),
    point("GPT-4o: user framing lowers a wrong product by", MEANS, own_vs_user("gpt-4o", "computation", "wrong"),
          "own_minus_user", 0.77),
    point("Grok: user framing lowers a wrong product by", MEANS, own_vs_user("grok-4-1-fast", "computation", "wrong"),
          "own_minus_user", 0.63),
    span("GPT-4o and Grok: correct product lowered by", MEANS,
         {"config": ["gpt-4o", "grok-4-1-fast"], "kind": "computation", "readout": "own-vs-user verbal", "source": "correct"},
         "own_minus_user", 0.52, 0.52),
    point("GPT-4o 'my answer' computation", SEP, sep("gpt-4o", "computation", "verbal_user"), "win", 0.63),
    point("Grok 'my answer' computation", SEP, sep("grok-4-1-fast", "computation", "verbal_user"), "win", 0.57),
    point("Qwen3-8B 'my answer' recall", SEP, sep("qwen3-8b", "recall", "verbal_user"), "win", 0.64),
    point("Qwen3-8B 'my answer' computation", SEP, sep("qwen3-8b", "computation", "verbal_user"), "win", 0.67),
    point("Qwen3-8B 'my answer' perception", SEP, sep("qwen3-8b", "perception", "verbal_user"), "win", 0.52),
    # ---- 4.3 Other readouts on the same answers
    span("Probe, own answers, training set, recall and computation", E3,
         {"split": "train", "readout": "probe", "kind": ["recall", "computation"]}, "auroc", 0.86, 0.91),
    span("Probe, own answers, training set, perception", E3,
         {"split": "train", "readout": "probe", "kind": "perception"}, "auroc", 0.70, 0.72),
    {"type": "best", "label": "Best P(True) variant, Qwen3-8B, training set", "file": E3,
     "where": {"split": "train", "config": "qwen3-8b"}, "lo": 0.70, "hi": 0.81, "below_probe": (0.02, 0.10)},
    {"type": "best", "label": "Best P(True) variant, Qwen3-1.7B, training set", "file": E3,
     "where": {"split": "train", "config": "qwen3-1.7b"}, "lo": 0.59, "hi": 0.76, "below_probe": None},
    span("Verbal (A and post-hoc), test set", E3,
         {"split": "test", "config": ["qwen3-8b", "qwen3-1.7b"], "readout": ["A", "posthoc"]}, "auroc", 0.49, 0.68),
    span("Post-hoc 'your answer' minus probe, test set", E3,
         {"split": "test", "config": ["qwen3-8b", "qwen3-1.7b"], "readout": "posthoc"}, "minus_probe", -0.32, -0.20),
    every("  intervals exclude 0 in five cells", E3,
          {"split": "test", "config": "qwen3-8b", "readout": "posthoc"}, "minus_probe_hi", lambda x: x < 0, "< 0"),
    every("  ...", E3, {"split": "test", "config": "qwen3-1.7b", "readout": "posthoc", "kind": CP},
          "minus_probe_hi", lambda x: x < 0, "< 0"),
    point("  and reaches 0 for Qwen3-1.7B recall", E3,
          {"split": "test", "config": "qwen3-1.7b", "readout": "posthoc", "kind": "recall"}, "minus_probe_hi", 0.0),
    point("Probe on planted answers, Qwen3-8B computation", SEP, sep("qwen3-8b", "computation", "probe"), "win", 0.95),
    point("Answer likelihood, Qwen3-8B computation", SEP, sep("qwen3-8b", "computation", "logprob"), "win", 0.93),
    point("Probe within likelihood bands, Qwen3-8B computation", SEP, sep("qwen3-8b", "computation", "probe"), "auroc_lp", 0.80),
    span("Probe within likelihood bands, recall and perception", SEP,
         {"readout": "probe", "kind": ["recall", "perception"]}, "auroc_lp", 0.53, 0.58),
    span("Tail: Qwen3-8B stated >= 0.9 (A), share wrong", "tail.csv", {"config": "qwen3-8b", "readout": "A"},
         "wrong_among_flagged", 0.60, 0.76),
    every("  every test answer flagged", "tail.csv", {"config": "qwen3-8b", "readout": "A"}, "share_flagged",
          lambda x: x == 1.0, "== 1.0"),
    point("Probe's top quarter error, Qwen3-8B training computation", "tail_quarter.csv",
          {"config": "qwen3-8b", "split": "train", "kind": "computation", "readout": "probe"}, "error_top_quarter", 0.03),
    point("  overall error", "tail_quarter.csv",
          {"config": "qwen3-8b", "split": "train", "kind": "computation", "readout": "probe"}, "error_overall", 0.66),
    # ---- 4.4 Mechanism tests
    span("E4: 'my answer' readable from the state, recall and computation", "e4_directions.csv",
         {"target": "verbal_user", "kind": ["recall", "computation"]}, "readable", 0.46, 0.66),
    span("E4: 'my answer' vs the probe, recall and computation", "e4_directions.csv",
         {"target": "verbal_user", "kind": ["recall", "computation"]}, "probe_vs_conf", 0.38, 0.51),
    every("E4: Qwen3-8B steering shift along the probe at most 0.02", "e4_steer_summary.csv", {"config": "qwen3-8b"},
          "shift_probe", lambda x: abs(x) <= 0.02 + TOL, "|x| <= 0.02"),
]


def select(d: pd.DataFrame, where: dict) -> pd.DataFrame:
    for k, v in where.items():
        d = d[d[k].isin(v if isinstance(v, list) else [v])]
    return d


def check(c: dict, cache: dict) -> tuple[bool, str]:
    if c["file"] not in cache:
        cache[c["file"]] = pd.read_csv(phase1.RESULTS / c["file"])
    d = select(cache[c["file"]], c["where"])
    if c["type"] == "best":
        pt = d[d.readout.str.startswith("ptrue")].groupby("kind").auroc.max()
        lo, hi = pt.min(), pt.max()
        ok = abs(lo - c["lo"]) <= TOL and abs(hi - c["hi"]) <= TOL
        msg = f"best per kind {pt.round(2).to_dict()}"
        if c["below_probe"]:
            gap = d[d.readout == "probe"].set_index("kind").auroc - pt
            ok = ok and abs(gap.min() - c["below_probe"][0]) <= TOL and abs(gap.max() - c["below_probe"][1]) <= TOL
            msg += f"; below probe {gap.round(2).to_dict()}"
        return ok, msg
    x = d[c["col"]].to_numpy(float)
    if len(x) == 0:
        return False, "no rows selected"
    if c["type"] == "point":
        if len(x) != 1:
            return False, f"{len(x)} rows selected, expected 1"
        return abs(x[0] - c["value"]) <= c["tol"] + 1e-9, f"file {x[0]:.4f}, report {c['value']}"
    if c["type"] == "range":
        lo, hi = np.nanmin(x), np.nanmax(x)
        ok = abs(lo - c["lo"]) <= c["tol"] + 1e-9 and abs(hi - c["hi"]) <= c["tol"] + 1e-9
        return ok, f"file {lo:.4f} to {hi:.4f} ({len(x)} rows), report {c['lo']} to {c['hi']}"
    ok = all(c["test"](v) for v in x)
    return ok, f"{len(x)} rows, {c['text']}" + ("" if ok else f"; values {np.round(x, 3).tolist()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true", help="rerun the Mac analysis scripts first")
    args = ap.parse_args()
    if args.rerun:
        for script in RERUN:
            if script == "analyze_lora.py" and not list(phase1.RESULTS.glob("e5_scores_*.parquet")):
                print(f"--- skipping {script} (no E5 scores yet)")
                continue
            print(f"--- running {script}")
            subprocess.run([sys.executable, str(phase1.HERE / script)], check=True, cwd=phase1.HERE,
                           stdout=subprocess.DEVNULL)
    cache: dict = {}
    failed = 0
    for c in CLAIMS:
        ok, msg = check(c, cache)
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {c['label']}: {msg}")
    print(f"\n{len(CLAIMS) - failed} of {len(CLAIMS)} claims hold")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
