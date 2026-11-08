"""Step 3: analysis of the judged Confidence Audit logs. No model calls.

Part 1: `load()` turns every full-run log under runs/<source>/full/<config>/ into one tidy
table, one row per (config, question, method), cached in results/rows.parquet. Everything else
in Step 3 is computed from that table.

Part 2: `discrimination()`, per (config, answerable kind, confidence method), on answered
questions only (abstentions are reported as a rate): accuracy, AUROC with a bootstrap 95%
interval, and for methods that give a probability, Brier, its Murphy decomposition and ECE.
AUROC is marked reportable only with at least MIN_CLASS right and MIN_CLASS wrong answers.
Written to results/discrimination.csv.

Part 3, controlling for difficulty (meta-d′ was dropped: open-ended answers have no type-1 d′):
  - within-level AUROC (`auroc_within` in discrimination.csv): right/wrong pairs compared only
    inside the same difficulty level (kind x level for the "answerable" union). Plain AUROC also
    rewards confidence that merely tracks the level ("six-digit products are hard"); within-level
    AUROC is item-level self-knowledge among equally hard questions.
  - `regression()`: per method, pooled over models, logistic regression of correct on the
    within-model percentile rank of confidence, one slope per kind, with a fixed effect for every
    (model, kind, level) cell, so difficulty and model accuracy are held fixed. The slope is the
    log-odds of being right for the model's most vs least confident answer in the same cell.
    Bootstrap intervals resample questions. Written to results/regression.csv.
  - `cells()`: accuracy and AUROC per (model, kind, level, method), for the accuracy-vs-AUROC
    plot (results/cells.csv).

Part 4, the hard-easy effect (Lichtenstein & Fischhoff): per (model, kind, level, method), for
methods that give a probability, accuracy and mean confidence on answered questions, and
overconfidence = confidence - accuracy (results/hard_easy.csv). Summarised per (model, kind,
method) by the *sensitivity* of confidence to difficulty: the slope of mean confidence on
accuracy across the five levels (1 = confidence falls exactly as accuracy falls, 0 = confidence
ignores difficulty, the hard-easy effect at its strongest), and overconfidence on the easiest
and hardest level (results/hard_easy_summary.csv).

Part 5, the methods compared (`methods_compared()`): Spearman correlation between the methods'
confidences per model on answerable questions (results/method_corr.csv); D vs A, paired per
question: mean shift and within-level AUROC difference with a paired bootstrap, Haiku's replies
with a reasoning preamble left out (results/d_vs_a.csv); and the questions where stated
confidence and sample agreement disagree most (A >= 0.9 with B <= 0.5, and A <= 0.5 with
B >= 0.9), with P's accuracy in each group (results/disagree_counts.csv, disagree_examples.csv).

Part 6, behaviour categories (false_premise, ambiguous, unknowable; `behaviour()`): rate of the
right behaviour per model, kind and call (P, A, D, C at each threshold), and stated confidence
(A, D) by outcome (results/behaviour_rates.csv, behaviour_conf.csv). Plus the Kalai threshold
calls on answerable questions (`kalai()`): answer rate, accuracy of what is answered, and the
penalised score at each t (+1 right, -t/(1-t) wrong, 0 abstain), against the t = 0 answers
scored under the same penalty, i.e. what abstaining gained (results/kalai.csv).

Part 7, forecasts (`forecasts()`): Brier and Murphy decomposition of F and FB per model, AUROC,
mean sample spread, against the crowd forecast and the base rate on the same 75 questions
(results/forecasts.csv). Confidence about the world, so reported apart from self-knowledge.

Methods (rows), and which answer each row's `correct` refers to:
  P        the plain answer (no confidence of its own; carries accuracy and abstention)
  A, D     stated confidence after / before answering, scored on their own answer
  B        agreement of the k samples with P's answer, scored on P's answer
  E, E_mean, E_min
           answer-token logprobs of P (sum = log P(whole answer), the main summary; mean and
           min kept as robustness checks), scored on P's answer
  C        implied confidence from the Kalai threshold calls: the number of thresholds
           (t = 0, .5, .75, .9) at which the model still answers, 0 to 4, scored on the
           t = 0 answer (the rule there says always answer)
  C0.0 ... C0.9
           each threshold call on its own, for answer / abstention rates
  F, FB    forecastable items: the stated probability and the mean of the samples; `correct`
           holds the outcome (0/1), so Brier etc. apply directly

Columns: source, config, qid, category, kind, level, method, answer, label, correct (0/1/NaN),
abstained (0/1/NaN), conf, preamble, crowd, spread.
  kind     computation, perception, recall, false_premise, ambiguous, unknowable, forecastable
  label    the grader's label (CORRECT / INCORRECT / ABSTAINS / CORRECTS_PREMISE / ...);
           ERROR / UNPARSED when the call failed or its reply couldn't be read
  correct  1 if the label is the right answer or right behaviour for the kind, NaN if no label
  preamble A and D only: the reply has text before its JSON (code fences aside), against the
           prompt's "Reply with only a JSON object". Only Claude Haiku does this: it works
           computation out first (86% of A, 73% of D replies), so its D is not confidence
           *before* answering there, and on ill-posed items a clarification can sit in the
           preamble while the judged JSON answer picks one work.
  crowd    F and FB only: the crowd's probability on the as-of date (ForecastBench)
  spread   FB only: standard deviation of the k sampled probabilities

Usage (repo root):
  uv run --extra evals python experiments/epic2/phase1_confidence/analyze.py check
  uv run --extra evals python experiments/epic2/phase1_confidence/analyze.py discrimination
  uv run --extra evals python experiments/epic2/phase1_confidence/analyze.py difficulty
  uv run --extra evals python experiments/epic2/phase1_confidence/analyze.py hard-easy
  uv run --extra evals python experiments/epic2/phase1_confidence/analyze.py methods
  uv run --extra evals python experiments/epic2/phase1_confidence/analyze.py behaviour
  uv run --extra evals python experiments/epic2/phase1_confidence/analyze.py forecasts
  uv run --extra evals python experiments/epic2/phase1_confidence/analyze.py all   # every table
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import confidence_task as ct  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402

from synapses.calibration import (  # noqa: E402
    auroc,
    bootstrap_ci,
    brier,
    ece,
    murphy,
    stratified_auroc,
)

RUNS = HERE / "runs"
RESULTS = HERE / "results"
ROWS = RESULTS / "rows.parquet"
THRESHOLDS = ("0.0", "0.5", "0.75", "0.9")
ANSWERABLE = ("computation", "perception", "recall")
CONF_METHODS = ("A", "D", "B", "E", "E_mean", "E_min", "C")
# Methods whose confidence is (or maps to) a probability of being right, so calibration
# applies. E's sum of logprobs is log P(whole answer).
AS_PROB = {"A": lambda c: c, "D": lambda c: c, "B": lambda c: c, "E": np.exp}
MIN_CLASS = 10


def _ok(kind: str, label: str | None) -> float:
    if label is None or label in ct.NO_LABEL:
        return np.nan
    right = "CORRECT" if kind in ("computation", "perception", "recall") else ct.RIGHT[kind]
    return float(label == right)


def _abstained(label: str | None) -> float:
    return np.nan if label is None or label in ct.NO_LABEL else float(label == "ABSTAINS")


def has_preamble(text: str | None) -> bool | None:
    """True if a JSON reply has text before the object (markdown code fences don't count)."""
    t = re.sub(r"```(?:json)?", "", text or "")
    i = t.find("{")
    return None if i < 0 else bool(t[:i].strip())


def sample_rows(source: str, config: str, s) -> list[dict]:
    """The rows for one logged sample (question)."""
    q = s.metadata["q"]
    calls = s.metadata["calls"]
    sc = s.scores["confidence_scorer"]
    kind = ct.judge_kind(q) or q["category"]
    base = {"source": source, "config": config, "qid": s.id, "category": q["category"],
            "kind": kind, "level": q["level"]}

    def row(method, answer=None, label=None, correct=np.nan, abstained=np.nan, conf=np.nan,
            preamble=None):
        return base | {"method": method, "answer": answer, "label": label,
                       "correct": correct, "abstained": abstained,
                       "conf": np.nan if conf is None else float(conf), "preamble": preamble}

    if q["category"] == "forecastable":
        y = float(q["meta"]["outcome"])
        fb = [p for p in sc.metadata.get("fb", []) if p is not None]
        crowd = q["meta"].get("crowd")
        return [row("F", conf=calls["F"].get("confidence"), correct=y) | {"crowd": crowd},
                row("FB", conf=np.mean(fb) if fb else np.nan, correct=y)
                | {"crowd": crowd, "spread": np.std(fb) if len(fb) > 1 else np.nan}]

    labels = sc.metadata["labels"]
    rows = []
    lp, p_lab = calls["P"].get("lp") or {}, labels["P"]
    p = dict(answer=calls["P"].get("answer"), label=p_lab, correct=_ok(kind, p_lab),
             abstained=_abstained(p_lab))
    rows.append(row("P", **p))
    if "b_agree" in sc.value:
        rows.append(row("B", **p, conf=sc.value["b_agree"]))
    for m, key in (("E", "sum"), ("E_mean", "mean"), ("E_min", "min")):
        if lp.get(key) is not None:
            rows.append(row(m, **p, conf=lp[key]))
    for m in ("A", "D"):
        if m in calls:
            lab = labels[m]
            rows.append(row(m, answer=calls[m].get("answer"), label=lab, correct=_ok(kind, lab),
                            abstained=_abstained(lab), conf=calls[m].get("confidence"),
                            preamble=has_preamble(calls[m].get("text"))))
    if calls.get("C"):
        answered = []
        for t in THRESHOLDS:
            lab = labels.get(f"C{t}")
            ab = _abstained(lab)
            answered.append(np.nan if np.isnan(ab) else 1 - ab)
            rows.append(row(f"C{t}", answer=calls["C"][t].get("answer"), label=lab,
                            correct=_ok(kind, lab), abstained=ab))
        c0 = rows[-len(THRESHOLDS)]
        implied = np.nan if any(np.isnan(answered)) else sum(answered)
        rows.append(row("C", answer=c0["answer"], label=c0["label"], correct=c0["correct"],
                        abstained=c0["abstained"], conf=implied))
    return rows


def load(refresh: bool = False) -> pd.DataFrame:
    """The tidy table for every full run (cached; `refresh` re-reads the logs)."""
    if ROWS.exists() and not refresh:
        return pd.read_parquet(ROWS)
    rows = []
    for f in sorted(glob.glob(str(RUNS / "*/full/*/*.eval"))):
        path = Path(f)
        source, config = path.parents[2].name, path.parent.name
        log = read_eval_log(f)
        if log.status != "success":
            raise SystemExit(f"{f}: status {log.status}")
        for s in log.samples:
            sc = (s.scores or {}).get("confidence_scorer")
            if sc is None or not sc.metadata.get("judged", True) and sc.value.get("ungraded"):
                raise SystemExit(f"{f}: {s.id} not judged; run judge_4090.sh first")
            rows += sample_rows(source, config, s)
    df = pd.DataFrame(rows)
    for col in ("crowd", "spread"):
        df[col] = df[col].astype(float)
    RESULTS.mkdir(exist_ok=True)
    df.to_parquet(ROWS, index=False)
    return df


def check(df: pd.DataFrame) -> None:
    """Figures to compare with the per-log summaries made earlier from the scores."""
    print(f"{len(df)} rows, {df.config.nunique()} configs, {df.qid.nunique()} questions\n")
    print("Rows per method and config:")
    print(df.pivot_table(index="config", columns="method", values="qid", aggfunc="count")
          .fillna(0).astype(int).to_string(), "\n")
    p = df[df.method == "P"]
    print("P accuracy by kind (correct = right answer or right behaviour):")
    print(p.pivot_table(index="config", columns="kind", values="correct").round(2).to_string(), "\n")
    a = df[(df.method == "A") & df.kind.isin(["computation", "perception", "recall",
                                               "false_premise", "ambiguous", "unknowable"])]
    rw = a.groupby(["config", "correct"]).conf.mean().unstack().round(2)
    rw.columns = ["A conf when wrong", "A conf when right"]
    print(rw.to_string(), "\n")
    f = df[df.method == "F"]
    print("Forecast Brier (F):")
    print(((f.conf - f.correct) ** 2).groupby(f.config).mean().round(3).to_string(), "\n")
    print("Missing labels (ERROR / UNPARSED) per method:")
    bad = df[df.label.isin(ct.NO_LABEL)]
    print(bad.groupby(["config", "method"]).size().to_string() if len(bad) else "none")


def discrimination(df: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    """One row per (config, kind, method) over the answerable kinds, plus their union
    ("answerable"; its AUROC also rewards confidence that differs between categories)."""
    d = df[df.method.isin(CONF_METHODS) & df.kind.isin(ANSWERABLE) & df.correct.notna()]
    out = []
    for (config, method), g in d.groupby(["config", "method"]):
        for kind, h in [*((k, g[g.kind == k]) for k in ANSWERABLE), ("answerable", g)]:
            a = h[(h.abstained == 0) & h.conf.notna()]
            p, y = a.conf.to_numpy(), a.correct.to_numpy()
            n_right, n_wrong = int(y.sum()), int(len(y) - y.sum())
            row = {"config": config, "kind": kind, "method": method, "n": len(h),
                   "abstain": h.abstained.mean(), "n_answered": len(a), "n_right": n_right,
                   "n_wrong": n_wrong, "acc": y.mean() if len(y) else np.nan,
                   "reportable": n_right >= MIN_CLASS and n_wrong >= MIN_CLASS,
                   "auroc": auroc(p, y) if len(y) else np.nan, "auroc_lo": np.nan,
                   "auroc_hi": np.nan}
            if row["reportable"]:
                row["auroc_lo"], row["auroc_hi"] = bootstrap_ci(auroc, p, y, n_boot=n_boot)
            strata = (a.kind + ":" + a.level.astype(str)).to_numpy()
            both = pd.Series(y).groupby(strata).transform(lambda v: 0 < v.sum() < len(v)).to_numpy()
            w_right, w_wrong = int(y[both].sum()), int(both.sum() - y[both].sum())
            row |= {"n_right_within": w_right, "n_wrong_within": w_wrong,
                    "reportable_within": w_right >= MIN_CLASS and w_wrong >= MIN_CLASS,
                    "auroc_within": stratified_auroc(p, y, strata) if len(y) else np.nan,
                    "auroc_within_lo": np.nan, "auroc_within_hi": np.nan}
            if row["reportable_within"]:
                row["auroc_within_lo"], row["auroc_within_hi"] = bootstrap_ci(
                    stratified_auroc, p, y, strata, n_boot=n_boot)
            if method in AS_PROB and len(y):
                pp = AS_PROB[method](p)
                m = murphy(pp, y)
                row |= {"mean_conf": pp.mean(), "brier": brier(pp, y), "ece": ece(pp, y),
                        "reliability": m.reliability, "resolution": m.resolution,
                        "uncertainty": m.uncertainty}
            out.append(row)
    res = pd.DataFrame(out)
    return res


def show_discrimination(res: pd.DataFrame, methods=("A", "D", "B", "E", "C")) -> None:
    """AUROC [95% interval] per config and method, one table per kind; '-' = too few right
    or wrong answers (n_right/n_wrong shown)."""
    def cell(r):
        if not r.reportable:
            return f"- ({r.n_right}/{r.n_wrong})"
        return f"{r.auroc:.2f} [{r.auroc_lo:.2f},{r.auroc_hi:.2f}]"
    for kind in (*ANSWERABLE, "answerable"):
        k = res[(res.kind == kind) & res.method.isin(methods)].copy()
        k["cell"] = k.apply(cell, axis=1)
        t = k.pivot(index="config", columns="method", values="cell").reindex(columns=methods)
        acc = res[(res.kind == kind) & (res.method == "B")].set_index("config").acc
        t.insert(0, "acc (P, answered)", acc.round(2))
        print(f"== {kind}: AUROC [95% CI]\n{t.fillna('').to_string()}\n")


def show_within(res: pd.DataFrame, methods=("A", "D", "B", "E")) -> None:
    """Plain -> within-level AUROC per config and method, one table per kind."""
    def cell(r):
        if not r.reportable_within:
            return "-"
        return f"{r.auroc:.2f}->{r.auroc_within:.2f} [{r.auroc_within_lo:.2f},{r.auroc_within_hi:.2f}]"
    for kind in (*ANSWERABLE, "answerable"):
        k = res[(res.kind == kind) & res.method.isin(methods)].copy()
        k["cell"] = k.apply(cell, axis=1)
        t = k.pivot(index="config", columns="method", values="cell").reindex(columns=methods)
        print(f"== {kind}: plain -> within-level AUROC [95% CI]\n{t.fillna('').to_string()}\n")


def _answered(df: pd.DataFrame, method: str) -> pd.DataFrame:
    d = df[(df.method == method) & df.kind.isin(ANSWERABLE) & (df.abstained == 0)
           & df.correct.notna() & df.conf.notna()].copy()
    d["z"] = d.groupby("config").conf.rank(pct=True)  # within-model percentile rank
    d["cell"] = d.config + "|" + d.kind + "|" + d.level.astype(str)
    var = d.groupby("cell").correct.transform(lambda v: 0 < v.sum() < len(v))
    return d[var].reset_index(drop=True)  # cells with no right/wrong mix carry no information


def _slopes(d: pd.DataFrame) -> dict[str, float]:
    cells = pd.get_dummies(d.cell, dtype=float)
    slopes = pd.DataFrame({k: d.z * (d.kind == k) for k in ANSWERABLE}, dtype=float)
    X = pd.concat([cells, slopes], axis=1).to_numpy()
    m = LogisticRegression(C=1e4, fit_intercept=False, max_iter=5000).fit(X, d.correct)
    return dict(zip(ANSWERABLE, m.coef_[0][-len(ANSWERABLE):]))


def regression(df: pd.DataFrame, methods=("A", "D", "B", "E", "C"), n_boot: int = 300,
               seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for method in methods:
        d = _answered(df, method)
        est = _slopes(d)
        qids = d.qid.unique()
        by_q = d.groupby("qid").indices
        boots = []
        for _ in range(n_boot):  # resample questions (all models' answers to a question together)
            take = np.concatenate([by_q[q] for q in rng.choice(qids, len(qids))])
            b = d.iloc[take].reset_index(drop=True)
            b["cell"] = b.cell  # duplicate questions stay in their cell
            boots.append(_slopes(b))
        boots = pd.DataFrame(boots)
        for k in ANSWERABLE:
            lo, hi = np.quantile(boots[k], [0.025, 0.975])
            out.append({"method": method, "term": k, "slope": est[k], "lo": lo, "hi": hi,
                        "n": int((d.kind == k).sum()), "models": d[d.kind == k].config.nunique()})
        for k in ("perception", "computation"):
            diff = boots[k] - boots["recall"]
            lo, hi = np.quantile(diff, [0.025, 0.975])
            out.append({"method": method, "term": f"{k} - recall", "slope": est[k] - est["recall"],
                        "lo": lo, "hi": hi, "n": np.nan, "models": np.nan})
    return pd.DataFrame(out)


def cells(df: pd.DataFrame, min_class: int = 5) -> pd.DataFrame:
    out = []
    d = df[df.method.isin(CONF_METHODS) & df.kind.isin(ANSWERABLE) & (df.abstained == 0)
           & df.correct.notna() & df.conf.notna()]
    for (config, kind, level, method), g in d.groupby(["config", "kind", "level", "method"]):
        y = g.correct.to_numpy()
        n_right, n_wrong = int(y.sum()), int(len(y) - y.sum())
        ok = n_right >= min_class and n_wrong >= min_class
        out.append({"config": config, "kind": kind, "level": level, "method": method,
                    "n": len(g), "acc": y.mean(), "n_right": n_right, "n_wrong": n_wrong,
                    "auroc": auroc(g.conf, y) if ok else np.nan})
    return pd.DataFrame(out)


def hard_easy(df: pd.DataFrame, min_n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df[df.method.isin(list(AS_PROB)) & df.kind.isin(ANSWERABLE) & (df.abstained == 0)
           & df.correct.notna() & df.conf.notna()].copy()
    d["p"] = [AS_PROB[m](c) for m, c in zip(d.method, d.conf)]
    lv = (d.groupby(["config", "kind", "method", "level"])
          .agg(n=("p", "size"), acc=("correct", "mean"), conf=("p", "mean")).reset_index())
    lv["over"] = lv.conf - lv.acc
    out = []
    for (config, kind, method), g in lv[lv.n >= min_n].groupby(["config", "kind", "method"]):
        g = g.sort_values("level")
        spread = g.acc.max() - g.acc.min()
        out.append({"config": config, "kind": kind, "method": method, "levels": len(g),
                    "acc_spread": spread,
                    # slope undefined when accuracy barely moves across levels
                    "sensitivity": np.polyfit(g.acc, g.conf, 1)[0] if spread >= 0.3 else np.nan,
                    "over_easiest": g.over.iloc[0], "over_hardest": g.over.iloc[-1],
                    "acc_easiest": g.acc.iloc[0], "acc_hardest": g.acc.iloc[-1]})
    return lv, pd.DataFrame(out)


PROB_METHODS = ("A", "D", "B", "E")
BEHAVIOUR = ("false_premise", "ambiguous", "unknowable")


def _wide(df: pd.DataFrame, value: str, methods, kinds=ANSWERABLE) -> pd.DataFrame:
    d = df[df.method.isin(methods) & df.kind.isin(kinds)]
    return d.pivot_table(index=["config", "qid", "kind", "level"], columns="method",
                         values=value, aggfunc="first").reset_index()


def method_corr(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between methods' confidences, per model, answered questions."""
    d = df[(df.abstained == 0) | df.method.isin(["B", "E"])]  # B, E inherit P's abstention
    d = d[d.abstained == 0]
    w = _wide(d, "conf", ("A", "D", "B", "E", "C"))
    out = []
    for config, g in w.groupby("config"):
        ms = [m for m in ("A", "D", "B", "E", "C") if m in g and g[m].notna().sum() > 20]
        for i, a in enumerate(ms):
            for b in ms[i + 1:]:
                pair = g[[a, b]].dropna()
                out.append({"config": config, "a": a, "b": b, "n": len(pair),
                            "spearman": pair[a].corr(pair[b], method="spearman")})
    return pd.DataFrame(out)


def d_vs_a(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Paired per question: both A and D answered, neither with a reasoning preamble."""
    d = df[df.method.isin(["A", "D"]) & df.kind.isin(ANSWERABLE)]
    w = d.pivot_table(index=["config", "qid", "kind", "level"], columns="method",
                      values=["conf", "correct", "abstained", "preamble"], aggfunc="first")
    w.columns = [f"{v}_{m}" for v, m in w.columns]
    w = w.reset_index()
    ok = ((w.abstained_A == 0) & (w.abstained_D == 0) & w.correct_A.notna() & w.correct_D.notna()
          & w.conf_A.notna() & w.conf_D.notna())
    rng = np.random.default_rng(seed)
    out = []
    for (config, kind), g in w[ok].groupby(["config", "kind"]):
        dropped = int(((g.preamble_A == True) | (g.preamble_D == True)).sum())  # noqa: E712
        g = g[(g.preamble_A != True) & (g.preamble_D != True)].reset_index(drop=True)  # noqa: E712
        lv = g.level.to_numpy()

        def diff(idx):
            h = g.iloc[idx]
            return (stratified_auroc(h.conf_D, h.correct_D, h.level)
                    - stratified_auroc(h.conf_A, h.correct_A, h.level))
        row = {"config": config, "kind": kind, "n": len(g), "dropped_preamble": dropped,
               "conf_A": g.conf_A.mean(), "conf_D": g.conf_D.mean(),
               "acc_A": g.correct_A.mean(), "acc_D": g.correct_D.mean(),
               "auroc_within_A": stratified_auroc(g.conf_A, g.correct_A, lv) if len(g) else np.nan,
               "auroc_within_D": stratified_auroc(g.conf_D, g.correct_D, lv) if len(g) else np.nan}
        both = all(min(g[f"correct_{m}"].sum(), (1 - g[f"correct_{m}"]).sum()) >= MIN_CLASS
                   for m in "AD")
        row["reportable"] = both
        row["diff"] = row["auroc_within_D"] - row["auroc_within_A"]
        if both:
            boots = np.array([diff(rng.integers(0, len(g), len(g))) for _ in range(n_boot)])
            boots = boots[~np.isnan(boots)]
            row["diff_lo"], row["diff_hi"] = np.quantile(boots, [0.025, 0.975])
        out.append(row)
    return pd.DataFrame(out)


def disagreements(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Questions where stated confidence (A) and sample agreement (B) point opposite ways."""
    w = _wide(df[df.abstained == 0], "conf", ("A", "B"))
    p = df[(df.method == "P") & df.kind.isin(ANSWERABLE)][["config", "qid", "answer", "correct"]]
    a_ok = df[df.method == "A"][["config", "qid", "correct"]].rename(columns={"correct": "a_correct"})
    w = w.dropna(subset=["A", "B"]).merge(p, on=["config", "qid"]).merge(a_ok, on=["config", "qid"])
    w["group"] = np.select([(w.A >= 0.9) & (w.B <= 0.5), (w.A <= 0.5) & (w.B >= 0.9)],
                           ["says sure, samples scatter", "says unsure, samples agree"], "other")
    counts = (w.groupby(["config", "group"])
              .agg(n=("qid", "size"), p_acc=("correct", "mean")).reset_index())
    counts["share"] = counts.n / counts.groupby("config").n.transform("sum")
    examples = w[w.group != "other"].sort_values(["group", "config", "kind", "qid"])
    return counts, examples


def behaviour(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    calls = ("P", "A", "D", "C0.0", "C0.5", "C0.75", "C0.9")
    d = df[df.method.isin(calls) & df.kind.isin(BEHAVIOUR) & df.correct.notna()]
    rates = (d.groupby(["config", "kind", "method"])
             .agg(n=("qid", "size"), right=("correct", "mean"), abstain=("abstained", "mean"))
             .reset_index())
    lab = d[d.method.isin(["P"])].groupby(["config", "kind"]).label.value_counts(normalize=True)
    rates = rates.merge(lab.unstack().reset_index().assign(method="P"),
                        on=["config", "kind", "method"], how="left")
    c = d[d.method.isin(["A", "D"])]
    conf = (c.groupby(["config", "kind", "method", "label"])
            .agg(n=("qid", "size"), conf=("conf", "mean")).reset_index())
    return rates, conf


def kalai(df: pd.DataFrame) -> pd.DataFrame:
    """The threshold calls on answerable questions: what abstaining gained under the penalty."""
    d = df[df.method.str.startswith("C0") & df.kind.isin(ANSWERABLE) & df.correct.notna()]
    base = d[d.method == "C0.0"].set_index(["config", "qid"])
    out = []
    for (config, kind, method), g in d.groupby(["config", "kind", "method"]):
        t = float(method[1:])
        pen = t / (1 - t)
        answered = g.abstained == 0
        score = np.where(answered, np.where(g.correct == 1, 1.0, -pen), 0.0).mean()
        b = base.loc[[(config, q) for q in g.qid if (config, q) in base.index]]
        b_ans = b.abstained == 0
        always = np.where(b_ans, np.where(b.correct == 1, 1.0, -pen), 0.0).mean()
        out.append({"config": config, "kind": kind, "t": t, "n": len(g),
                    "answer_rate": answered.mean(),
                    "acc_answered": g.correct[answered].mean() if answered.any() else np.nan,
                    "score": score, "score_t0_answers": always, "gain": score - always})
    return pd.DataFrame(out)


def forecasts(df: pd.DataFrame) -> pd.DataFrame:
    f = df[df.method.isin(["F", "FB"]) & df.conf.notna()]
    out = []
    crowd = f[f.method == "F"].drop_duplicates("qid")
    y_all = crowd.correct.to_numpy()
    ref = {"crowd": crowd.crowd.to_numpy(), "base rate": np.full(len(y_all), y_all.mean())}
    for name, pr in ref.items():
        m = murphy(pr, y_all)
        out.append({"config": name, "method": "", "n": len(y_all), "brier": brier(pr, y_all),
                    "reliability": m.reliability, "resolution": m.resolution,
                    "auroc": auroc(pr, y_all), "mean_p": pr.mean()})
    for (config, method), g in f.groupby(["config", "method"]):
        p, y = g.conf.to_numpy(), g.correct.to_numpy()
        m = murphy(p, y)
        out.append({"config": config, "method": method, "n": len(g), "brier": brier(p, y),
                    "reliability": m.reliability, "resolution": m.resolution,
                    "auroc": auroc(p, y), "mean_p": p.mean(),
                    "crowd_brier_same_q": brier(g.crowd, y),
                    "mean_spread": g.spread.mean() if method == "FB" else np.nan})
    return pd.DataFrame(out)


def save(res: pd.DataFrame, name: str) -> pd.DataFrame:
    """Write a results table (only main() does, so tests never touch results/)."""
    RESULTS.mkdir(exist_ok=True)
    res.to_csv(RESULTS / f"{name}.csv", index=False, float_format="%.4f")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["check", "discrimination", "difficulty", "hard-easy",
                                        "methods", "behaviour", "forecasts", "all"])
    ap.add_argument("--refresh", action="store_true", help="re-read the logs")
    args = ap.parse_args()
    df = load(args.refresh)
    cmd = args.command

    def run(c: str) -> bool:
        return cmd in (c, "all")

    if cmd == "check":
        check(df)
    if run("discrimination") or run("difficulty"):
        res = save(discrimination(df), "discrimination")
        show_discrimination(res) if cmd == "discrimination" else show_within(res)
    if run("difficulty"):
        save(cells(df), "cells")
        reg = save(regression(df), "regression")
        print("== Regression: confidence slope per kind (log-odds, most vs least confident answer"
              " in the same model x kind x level cell) [95% CI]")
        reg["cell"] = reg.apply(lambda r: f"{r.slope:+.2f} [{r.lo:+.2f},{r.hi:+.2f}]", axis=1)
        print(reg.pivot(index="term", columns="method", values="cell")
              .reindex(columns=["A", "D", "B", "E", "C"]).to_string(), "\n")
    if run("hard-easy"):
        lv, summ = hard_easy(df)
        save(lv, "hard_easy")
        save(summ, "hard_easy_summary")
        for col, title in (("sensitivity", "Sensitivity of confidence to difficulty (slope of "
                            "confidence on accuracy across levels; 1 = tracks, 0 = flat; '' = "
                            "accuracy spread < 0.3)"),
                           ("over_easiest", "Overconfidence (confidence - accuracy), easiest level"),
                           ("over_hardest", "Overconfidence, hardest level")):
            t = summ.pivot_table(index="config", columns=["kind", "method"], values=col)
            print(f"== {title}\n{t.round(2).fillna('').to_string()}\n")
    if run("methods"):
        corr = save(method_corr(df), "method_corr")
        corr["pair"] = corr.a + "~" + corr.b
        print("== Spearman correlation between methods (answerable, answered)")
        print(corr.pivot(index="config", columns="pair", values="spearman").round(2)
              .fillna("").to_string(), "\n")
        dva = save(d_vs_a(df), "d_vs_a")
        dva["cell"] = dva.apply(lambda r: (f"{r.auroc_within_A:.2f} vs {r.auroc_within_D:.2f}  "
                                           f"{r['diff']:+.2f} [{r.diff_lo:+.2f},{r.diff_hi:+.2f}]")
                                if r.reportable else "-", axis=1)
        print("== Within-level AUROC, A (after) vs D (before), D - A [95% CI]; Haiku preamble"
              " replies left out")
        print(dva.pivot(index="config", columns="kind", values="cell").to_string(), "\n")
        print("== Mean stated confidence, A vs D (same questions)")
        print(dva.pivot(index="config", columns="kind", values=["conf_A", "conf_D"]).round(2)
              .to_string(), "\n")
        counts, ex = disagreements(df)
        save(counts, "disagree_counts")
        save(ex, "disagree_examples")
        print("== A vs B disagreements: share of answered answerable questions, P accuracy")
        c = counts[counts.group != "other"]
        print(c.pivot(index="config", columns="group", values=["share", "p_acc"]).round(2)
              .to_string(), "\n")
    if run("behaviour"):
        rates, conf = behaviour(df)
        save(rates, "behaviour_rates")
        save(conf, "behaviour_conf")
        print("== Right behaviour rate (correct premise / clarify / abstain) by call")
        print(rates.pivot_table(index="config", columns=["kind", "method"], values="right")
              .reindex(columns=["P", "A", "D", "C0.0", "C0.9"], level=1).round(2).to_string(), "\n")
        print("== Mean stated confidence (A) by P-label kind of response")
        a = conf[conf.method == "A"]
        print(a.pivot_table(index="config", columns=["kind", "label"], values="conf").round(2)
              .fillna("").to_string(), "\n")
        print("== Mean stated confidence (D, before answering) by label")
        dd = conf[conf.method == "D"]
        print(dd.pivot_table(index="config", columns=["kind", "label"], values="conf").round(2)
              .fillna("").to_string(), "\n")
        k = save(kalai(df), "kalai")
        print("== Kalai threshold calls, answerable: answer rate / gain from abstaining vs"
              " answering everything (penalised score)")
        k["cell"] = k.apply(lambda r: f"{r.answer_rate:.2f} / {r.gain:+.2f}", axis=1)
        print(k[k.t > 0].pivot_table(index="config", columns=["kind", "t"], values="cell",
                                     aggfunc="first").to_string(), "\n")
    if run("forecasts"):
        fc = save(forecasts(df), "forecasts")
        print("== Forecasts (75 questions): Brier, Murphy, AUROC")
        print(fc.round(3).fillna("").to_string(index=False))

if __name__ == "__main__":
    main()
