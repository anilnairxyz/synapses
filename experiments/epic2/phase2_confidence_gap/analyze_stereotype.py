"""E1, stereotype analysis (hypothesis 6). Mac, existing Phase 1 / 1B outputs, no model calls.

Question: within a difficulty level, what does stated confidence track? If it comes from what
the question looks like (number sizes, how famous the subject is, word length) rather than
from the model's state about this answer, it is a stereotype.

For every config with stated confidence (A: stated after answering; D: stated before) and
each answerable kind, on the answers it gave (not abstentions), stated confidence is compared
with:

  details    the question's look, as one detail per kind fixed in advance with its expected
             sign (recall: subject popularity; computation: minus operand size, log a + log b;
             perception: minus word length). No fitting. Every config.
  correct    whether that answer is right (A's or D's own answer).
  probe_before, probe_after
             the Phase 1B hidden-state probe scores (positions "gen" and "ans_last", trained
             on the same kind, scored on the test questions). What the model's state says.
             Open models only (qwen3-8b, qwen3-1.7b). They score the plain answer P, which is
             usually but not always the same as A's or D's answer.
  meta_probe the Phase 1B metadata probe: correctness predicted from the question's details.
             Whether stated confidence follows the part of the details that predicts success.

All comparisons are within strict strata (level x task family or relation, as in Phase 1B), as
within-stratum Spearman correlations: ranks are taken inside each stratum, centred, pooled,
and correlated. Strata where the stated confidence does not vary are dropped (common to every
comparison, so the comparisons are paired). 95% intervals from 1,000 bootstrap resamples of
questions; paired differences (details - probe_after, details - correct) on the same resamples.

Regression (open models): within-stratum centred ranks of stated confidence on those of
details, probe_before and probe_after, all standardised (so coefficients are standardised
betas), OLS, bootstrap intervals.

Prediction (plan, written before the run): within a level, stated confidence is predicted at
least as well by the question's details as by the hidden-state probe; on perception, by neither.

Writes results/e1_stereotype.csv, results/e1_regression.csv, results/figures/e1_stereotype.png.
--null: the details score against stated confidence shuffled within strata, 200 times per cell
(results/e1_null_check.csv); it should centre on 0.

    uv run --extra evals python experiments/epic2/phase2_confidence_gap/analyze_stereotype.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import phase1

KINDS = ("recall", "computation", "perception")
OPEN = ("qwen3-8b", "qwen3-1.7b")
N_BOOT = 1000
SEED = 0


# ----------------------------------------------------------------------------- statistics


def centred_ranks(x, strata) -> np.ndarray:
    """Ranks inside each stratum, scaled to [0, 1] and centred (ties share the average rank)."""
    s = pd.Series(np.asarray(x, float))
    g = pd.Series(np.asarray(strata))
    r = s.groupby(g).rank(method="average")
    n = s.groupby(g).transform("count")
    return ((r - 1) / (n - 1).where(n > 1) - 0.5).fillna(0.0).to_numpy()


def within_spearman(x, y, strata) -> float:
    """Spearman correlation of x and y counting only variation inside strata."""
    rx, ry = centred_ranks(x, strata), centred_ranks(y, strata)
    den = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / den) if den > 0 else np.nan


def varying_strata(conf, strata) -> np.ndarray:
    """Mask of rows whose stratum has more than one distinct stated confidence."""
    s = pd.Series(np.asarray(conf, float))
    return s.groupby(np.asarray(strata)).transform("nunique").to_numpy() > 1


def ci(vals) -> tuple[float, float]:
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    return (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) if len(v) else (np.nan, np.nan)


# ----------------------------------------------------------------------------- data


def questions() -> pd.DataFrame:
    rows = [json.loads(line) for line in phase1.PHASE1_QUESTIONS.read_text().splitlines()]
    return pd.DataFrame({"qid": [q["id"] for q in rows], "meta": [q["meta"] for q in rows]})


def stated(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per (config, method A or D, answered test question) with conf and correctness."""
    d = rows[rows.method.isin(["A", "D"]) & rows.kind.isin(KINDS)]
    d = d[(d.abstained == 0) & d.conf.notna() & d.correct.notna()]
    return d[["config", "method", "qid", "kind", "level", "conf", "correct"]].rename(columns={"kind": "category"})


def probe_scores() -> pd.DataFrame:
    """qid x config -> probe_before, probe_after, meta_probe (test set, same-kind probes)."""
    pa = phase1.probes_analysis
    s = pd.read_parquet(phase1.PHASE1_RESULTS / "probe_scores.parquet")
    s = s[(s.train_kind == s.category) & s.category.isin(KINDS)]
    keep = ((s.probe == "logreg") & s.position.isin([pa.BEFORE, pa.AFTER])) | (s.probe == "meta")
    s = s[keep].assign(col=lambda x: np.where(x.probe == "meta", "meta_probe",
                                              np.where(x.position == pa.BEFORE, "probe_before", "probe_after")))
    return s.pivot_table(index=["config", "qid"], columns="col", values="score").reset_index()


DETAIL = {  # kind -> (metadata feature, sign): one fixed detail per kind, chosen in advance
    "recall": ("log10_pop", +1),       # more famous subject -> expect higher confidence
    "computation": ("size", -1),       # bigger numbers (log a + log b) -> lower
    "perception": ("length", -1),      # longer word -> lower
}


def details_score(d: pd.DataFrame, kind: str) -> np.ndarray:
    """The question's look, as one pre-specified detail per kind (no fitting).

    A fitted, cross-validated prediction was tried first and dropped: out-of-fold predictions
    anti-correlate with the target under no signal (shuffle check: mean -0.09), which would
    bias the comparison. The Phase 1B metadata probe (fitted on the separate training
    questions) is the combined version, for the open models.
    """
    f = phase1.probe.metadata_features(d, kind).reset_index(drop=True)
    if kind == "computation":
        f["size"] = f.log_a + f.log_b
    col, sign = DETAIL[kind]
    return sign * f[col].to_numpy(float)


def null_check(d: pd.DataFrame, n: int = 200) -> pd.DataFrame:
    """The details score against stated confidence shuffled within strata: should centre on 0."""
    rng = np.random.default_rng(SEED + 1)
    out = []
    for (config, method, kind), h in d.groupby(["config", "method", "category"]):
        h = h.reset_index(drop=True)
        h = h[varying_strata(h.conf, h.s_strict)].reset_index(drop=True)
        if len(h) < 20:
            continue
        x, st = details_score(h, kind), h.s_strict.to_numpy()
        vals = []
        for _ in range(n):
            shuf = h.conf.groupby(st).transform(lambda v: rng.permutation(v.to_numpy()))
            vals.append(within_spearman(shuf, x, st))
        out.append({"config": config, "method": method, "kind": kind, "null_mean": np.mean(vals),
                    "null_sd": np.std(vals)})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- analysis


COMPARE = ("details", "correct", "probe_before", "probe_after", "meta_probe")
DIFFS = (("details", "probe_after"), ("details", "probe_before"), ("details", "correct"))


def analyse_cell(h: pd.DataFrame, idx: np.ndarray) -> dict:
    st = h.s_strict.to_numpy()
    conf = h.conf.to_numpy()
    row = {"n": len(h), "n_distinct": int(h.conf.nunique()),
           "modal_share": float(h.conf.value_counts(normalize=True).iloc[0]),
           "acc": float(h.correct.mean())}
    have = [c for c in COMPARE if c in h and h[c].notna().all()]
    for c in have:
        x = h[c].to_numpy(float)
        row[c] = within_spearman(conf, x, st)
        row[f"{c}_lo"], row[f"{c}_hi"] = ci([within_spearman(conf[i], x[i], st[i]) for i in idx])
    for a, b in DIFFS:
        if a in have and b in have:
            xa, xb = h[a].to_numpy(float), h[b].to_numpy(float)
            row[f"{a}-{b}"] = row[a] - row[b]
            row[f"{a}-{b}_lo"], row[f"{a}-{b}_hi"] = ci(
                [within_spearman(conf[i], xa[i], st[i]) - within_spearman(conf[i], xb[i], st[i]) for i in idx])
    row["auroc_within"] = phase1.cal.stratified_auroc(conf, h.correct.to_numpy(), st)
    return row


def regression_cell(h: pd.DataFrame, idx: np.ndarray) -> dict:
    st = h.s_strict.to_numpy()
    cols = ["details", "probe_before", "probe_after"]
    Y = centred_ranks(h.conf, st)
    X = np.column_stack([centred_ranks(h[c], st) for c in cols])

    def fit(i):
        x, yv = X[i], Y[i]
        x = (x - x.mean(0)) / x.std(0)
        yv = (yv - yv.mean()) / yv.std()  # standardised betas
        return np.linalg.lstsq(np.column_stack([np.ones(len(i)), x]), yv, rcond=None)[0][1:]

    est = fit(np.arange(len(h)))
    boots = np.array([fit(i) for i in idx])
    out = {"n": len(h)}
    for j, c in enumerate(cols):
        out[c] = float(est[j])
        out[f"{c}_lo"], out[f"{c}_hi"] = ci(boots[:, j])
    return out


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    pa = phase1.probes_analysis
    rows = pd.read_parquet(phase1.PHASE1_RESULTS / "rows.parquet")
    d = pa.add_strata(stated(rows), pa.family_map()).merge(questions(), on="qid", how="left")
    d = d.merge(probe_scores(), on=["config", "qid"], how="left")
    rng = np.random.default_rng(SEED)
    res, reg = [], []
    for (config, method, kind), h in d.groupby(["config", "method", "category"], sort=True):
        h = h.reset_index(drop=True)
        h["details"] = details_score(h, kind)
        h = h[varying_strata(h.conf, h.s_strict)].reset_index(drop=True)
        key = {"config": config, "method": method, "kind": kind}
        if len(h) < 20:
            res.append(key | {"n": len(h), "note": "stated confidence barely varies within strata"})
            continue
        idx = rng.integers(0, len(h), size=(N_BOOT, len(h)))
        res.append(key | analyse_cell(h, idx))
        if config in OPEN:
            reg.append(key | regression_cell(h, idx))
    return pd.DataFrame(res), pd.DataFrame(reg)


# ----------------------------------------------------------------------------- figure


def figure(res: pd.DataFrame) -> None:
    pl = phase1.plots
    import matplotlib.pyplot as plt

    series = [("details", "Question's look (one detail)", pl.ORANGE, "o"),
              ("probe", "Hidden-state probe (same timing)", pl.BLUE, "s"),
              ("correct", "Being right", pl.AQUA, "D")]
    configs = [c for c in pl.MODELS if c in set(res.config)]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), sharey=True, sharex=True)
    for r, method in enumerate(("A", "D")):
        for c, kind in enumerate(KINDS):
            ax = axes[r, c]
            g = res[(res.method == method) & (res.kind == kind)].set_index("config")
            for j, (key, name, colour, marker) in enumerate(series):
                col = ("probe_after" if method == "A" else "probe_before") if key == "probe" else key
                for i, cfg in enumerate(configs):
                    if cfg not in g.index or col not in g or pd.isna(g.loc[cfg, col]):
                        continue
                    y = i + (j - 1) * 0.22
                    v, lo, hi = g.loc[cfg, [col, f"{col}_lo", f"{col}_hi"]]
                    ax.plot([lo, hi], [y, y], color=colour, lw=1.4, alpha=0.6)
                    ax.plot(v, y, marker, color=colour, ms=6, mec=pl.SURFACE, mew=1, label=name)
            ax.axvline(0, color=pl.MUTED, lw=0.8)
            ax.set_yticks(range(len(configs)), [pl.MODEL_NAME[x] for x in configs])
            ax.set_xlim(-0.45, 0.85)
            if r == c == 0:
                ax.invert_yaxis()
            if r == 0:
                ax.set_title(kind.capitalize())
            if c == 0:
                ax.set_ylabel("Stated after (A)" if method == "A" else "Stated before (D)")
    seen = {}
    for ax in axes.flat:
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l_, h_)
    fig.legend(seen.values(), seen.keys(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("What stated confidence tracks within a difficulty level", y=1.07, fontweight="bold")
    fig.supxlabel("Within-level Spearman correlation with stated confidence", fontsize=10, color=pl.INK2)
    fig.tight_layout()
    out = phase1.RESULTS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "e1_stereotype.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / "e1_stereotype.png")


def main() -> None:
    import sys

    if "--null" in sys.argv:
        pa = phase1.probes_analysis
        rows = pd.read_parquet(phase1.PHASE1_RESULTS / "rows.parquet")
        d = pa.add_strata(stated(rows), pa.family_map()).merge(questions(), on="qid", how="left")
        nc = null_check(d)
        nc.to_csv(phase1.RESULTS / "e1_null_check.csv", index=False)
        print(nc.round(3).to_string(index=False))
        print("mean of null means", round(nc.null_mean.mean(), 3))
        return
    res, reg = run()
    phase1.RESULTS.mkdir(parents=True, exist_ok=True)
    res.to_csv(phase1.RESULTS / "e1_stereotype.csv", index=False)
    reg.to_csv(phase1.RESULTS / "e1_regression.csv", index=False)
    cols = ["config", "method", "kind", "n", "n_distinct", "details", "correct", "probe_before",
            "probe_after", "meta_probe", "details-probe_after", "auroc_within"]
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(res[[c for c in cols if c in res]].round(2).to_string(index=False))
        print()
        print(reg.round(2).to_string(index=False))
    figure(res)


if __name__ == "__main__":
    main()
