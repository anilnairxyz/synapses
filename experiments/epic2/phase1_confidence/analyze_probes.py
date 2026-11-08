"""Phase 1B step 6: analysis of the probe results. No model calls, no hidden states.

Inputs (from probe.py on the PC): results/probe_layers.csv, probe_scores.parquet (test set,
the Phase 1 questions), probe_oof.parquet (out-of-fold scores on the ~6,000 training
questions); and results/rows.parquet (Phase 1, for A, D, B, E on the same test questions).

Label: 1 if the plain answer (P) is right, 0 if wrong or an abstention. Scores are probe
logits; AUROC is rank-only, so they compare directly with any confidence.

Difficulty control: within-level AUROC (right/wrong pairs only inside a difficulty level),
as in Phase 1. "strict" also splits the level by task family (perception: count / nth /
reverse) or relation (recall: director, author, ...), because the metadata probe showed
those differ in difficulty within a level. Computation has one family, so strict = level.

Positions: q_last (end of question), gen (end of prompt, just before the first answer
token), ans_last (last answer token), ans_mean (mean over the answer). "Before" = gen,
"after" = ans_last (fixed in advance; q_last and ans_mean are reported alongside).

Tables (results/):
  probe_auroc.csv      training set (out-of-fold): within-level AUROC per config, kind,
                       position (and the metadata probe), level and strict, bootstrap CI
  probe_gaps.csv       training set: paired differences with bootstrap CIs: after - before,
                       before - metadata, after - metadata; and base - instruct per position
  probe_vs_methods.csv test set: probe (before, after) vs A, D, B, E within-level AUROC on
                       the questions P answered, paired differences probe - A and probe - B
  probe_transfer.csv   test set: probe trained on one kind, scored on another
  probe_abstain.csv    test set: does the pooled probe's score predict abstention on
                       unknowable and ill-posed questions (AUROC of -score for abstaining)?

Usage (repo root):
  uv run --extra evals python experiments/epic2/phase1_confidence/analyze_probes.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from synapses.calibration import auroc, stratified_auroc  # noqa: E402

RESULTS = HERE / "results"
KINDS = ("recall", "computation", "perception")
POSITIONS = ("q_last", "gen", "ans_last", "ans_mean")
BEFORE, AFTER = "gen", "ans_last"
INSTRUCT, BASE = "qwen3-8b", "qwen3-8b-base"
N_BOOT = 1000


def family_map() -> dict[str, str]:
    """qid -> task family (perception) or relation (recall); '' otherwise."""
    out = {}
    for f in ("questions.jsonl", "train_questions.jsonl"):
        for line in (ROOT / "data" / "epic2" / "confidence" / f).read_text().splitlines():
            q = json.loads(line)
            m = q["meta"]
            out[q["id"]] = (m.get("family", "") if q["category"] == "perception"
                            else m.get("relation", "") if q["category"] == "recall" else "")
    return out


def add_strata(d: pd.DataFrame, fam: dict[str, str]) -> pd.DataFrame:
    d = d.copy()
    lv = d.category + ":" + d.level.astype("Int64").astype(str)
    d["s_level"] = lv
    d["s_strict"] = lv + ":" + d.qid.map(fam).fillna("")
    return d


def wide(d: pd.DataFrame, cols=("qid", "category", "level", "y", "s_level", "s_strict")) -> pd.DataFrame:
    """One row per question, one column per position (scores)."""
    return d.pivot_table(index=list(cols), columns="position", values="score").reset_index()


def boot_idx(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(N_BOOT, n))


def ci(vals) -> tuple[float, float]:
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    return (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) if len(v) else (np.nan, np.nan)


def within_with_ci(score, y, strata, idx) -> tuple[float, float, float]:
    score, y, strata = map(np.asarray, (score, y, strata))
    est = stratified_auroc(score, y, strata)
    lo, hi = ci([stratified_auroc(score[i], y[i], strata[i]) for i in idx])
    return est, lo, hi


def diff_with_ci(a, b, y, strata, idx) -> tuple[float, float, float]:
    """Paired within-level AUROC difference a - b on the same questions."""
    a, b, y, strata = map(np.asarray, (a, b, y, strata))
    est = stratified_auroc(a, y, strata) - stratified_auroc(b, y, strata)
    lo, hi = ci([stratified_auroc(a[i], y[i], strata[i]) - stratified_auroc(b[i], y[i], strata[i])
                 for i in idx])
    return est, lo, hi


# ----------------------------------------------------------------------------- training set


def oof_tables(oof: pd.DataFrame, fam: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """probe_auroc and probe_gaps from the out-of-fold scores (own-kind probes)."""
    d = oof[oof.category.isin(KINDS) & (oof.train_kind == oof.category)]
    d = add_strata(d, fam)
    auc_rows, gap_rows, by_cfg = [], [], {}
    for (config, kind), g in d.groupby(["config", "category"]):
        w = wide(g)
        by_cfg[(config, kind)] = w
        idx = boot_idx(len(w))
        y = w.y.to_numpy()
        for strata in ("level", "strict"):
            st = w[f"s_{strata}"].to_numpy()
            for pos in ("meta", *POSITIONS):
                est, lo, hi = within_with_ci(w[pos], y, st, idx)
                auc_rows.append({"config": config, "kind": kind, "position": pos, "strata": strata,
                                 "auroc_within": est, "lo": lo, "hi": hi,
                                 "n_right": int(y.sum()), "n_wrong": int(len(y) - y.sum())})
            for name, a, b in (("after - before", AFTER, BEFORE),
                               ("ans_mean - q_last", "ans_mean", "q_last"),
                               ("before - meta", BEFORE, "meta"),
                               ("after - meta", AFTER, "meta")):
                est, lo, hi = diff_with_ci(w[a], w[b], y, st, idx)
                gap_rows.append({"config": config, "kind": kind, "comparison": name,
                                 "strata": strata, "diff": est, "lo": lo, "hi": hi})
    # Base vs instruct: same questions, each model scored against its own right/wrong labels.
    for kind in KINDS:
        if (INSTRUCT, kind) not in by_cfg or (BASE, kind) not in by_cfg:
            continue
        m = by_cfg[(INSTRUCT, kind)].merge(by_cfg[(BASE, kind)], on=["qid", "s_level", "s_strict"],
                                          suffixes=("_i", "_b"))
        idx = boot_idx(len(m), seed=1)
        for strata in ("level", "strict"):
            st = m[f"s_{strata}"].to_numpy()
            for pos in POSITIONS:
                yb, yi = m.y_b.to_numpy(), m.y_i.to_numpy()
                sb, si = m[f"{pos}_b"].to_numpy(), m[f"{pos}_i"].to_numpy()
                est = stratified_auroc(sb, yb, st) - stratified_auroc(si, yi, st)
                lo, hi = ci([stratified_auroc(sb[i], yb[i], st[i]) - stratified_auroc(si[i], yi[i], st[i])
                             for i in idx])
                gap_rows.append({"config": "base - instruct", "kind": kind,
                                 "comparison": f"{pos}: base - instruct", "strata": strata,
                                 "diff": est, "lo": lo, "hi": hi})
    return pd.DataFrame(auc_rows), pd.DataFrame(gap_rows)


# ----------------------------------------------------------------------------- test set


def vs_methods(scores: pd.DataFrame, rows: pd.DataFrame, fam: dict[str, str]) -> pd.DataFrame:
    """Probe vs Phase 1 methods on the test questions the plain answer answered.

    The probe, B and E score P's answer; A and D score their own answers (as in Phase 1)."""
    s = scores[(scores.probe == "logreg") & (scores.train_kind == scores.category)
               & scores.category.isin(KINDS) & scores.position.isin([BEFORE, AFTER])]
    out = []
    for config, g in s.groupby("config"):
        w = add_strata(g, fam).pipe(wide)
        r = rows[(rows.config == config) & rows.kind.isin(KINDS)]
        p = r[r.method == "P"].set_index("qid")
        w = w[w.qid.map(p.abstained) == 0]  # answered by P, as Phase 1's AUROCs
        for m in ("A", "D", "B", "E"):
            mm = r[r.method == m].set_index("qid")
            w[m] = w.qid.map(mm.conf)
            w[f"y_{m}"] = w.qid.map(mm.correct) if m in ("A", "D") else w.y
            w[f"ok_{m}"] = w.qid.map(mm.abstained) == 0 if m in ("A", "D") else True
        for kind, h in w.groupby("category"):
            idx = boot_idx(len(h), seed=2)
            row = {"config": config, "kind": kind, "n": len(h), "n_right": int(h.y.sum())}
            for strata in ("level", "strict"):
                st = h[f"s_{strata}"].to_numpy()
                y = h.y.to_numpy()
                for col in (BEFORE, AFTER):
                    row[f"{col}_{strata}"] = stratified_auroc(h[col], y, st)
                for m in ("A", "D", "B", "E"):
                    ok = (h[f"ok_{m}"] & h[m].notna() & h[f"y_{m}"].notna()).to_numpy()
                    row[f"{m}_{strata}"] = (stratified_auroc(h[m][ok], h[f"y_{m}"][ok], st[ok])
                                            if ok.sum() else np.nan)
                # Paired differences on P's answer (B, E share it); vs A on A's own answer is
                # not paired, so only B and E get a paired CI.
                for m in ("B", "E"):
                    ok = h[m].notna().to_numpy()
                    for col in (BEFORE, AFTER):
                        est, lo, hi = diff_with_ci(h[col][ok], h[m][ok], y[ok], st[ok],
                                                   boot_idx(int(ok.sum()), seed=3))
                        row |= {f"{col}-{m}_{strata}": est, f"{col}-{m}_{strata}_lo": lo,
                                f"{col}-{m}_{strata}_hi": hi}
                # Probe vs A: bootstrap questions jointly, each on its own labels.
                ok = (h.ok_A & h.A.notna() & h.y_A.notna()).to_numpy()
                for col in (BEFORE, AFTER):
                    hh = h[ok]
                    a_s, a_y, p_s, p_y = (hh.A.to_numpy(), hh.y_A.to_numpy(float),
                                          hh[col].to_numpy(), hh.y.to_numpy())
                    sti = st[ok]
                    est = stratified_auroc(p_s, p_y, sti) - stratified_auroc(a_s, a_y, sti)
                    lo, hi = ci([stratified_auroc(p_s[i], p_y[i], sti[i])
                                 - stratified_auroc(a_s[i], a_y[i], sti[i])
                                 for i in boot_idx(len(hh), seed=4)])
                    row |= {f"{col}-A_{strata}": est, f"{col}-A_{strata}_lo": lo,
                            f"{col}-A_{strata}_hi": hi}
            del idx
            out.append(row)
    return pd.DataFrame(out)


def transfer(scores: pd.DataFrame, fam: dict[str, str]) -> pd.DataFrame:
    """Within-level AUROC of a probe trained on one kind, on each test kind."""
    s = scores[(scores.probe == "logreg") & scores.category.isin(KINDS)
               & scores.position.isin([BEFORE, AFTER])]
    s = add_strata(s, fam)
    out = []
    for (config, tk, pos, kind), g in s.groupby(["config", "train_kind", "position", "category"]):
        out.append({"config": config, "train_kind": tk, "position": pos, "test_kind": kind,
                    "auroc_within": stratified_auroc(g.score, g.y, g.s_level)})
    return pd.DataFrame(out)


def abstain(scores: pd.DataFrame) -> pd.DataFrame:
    """On unknowable and ill-posed questions: AUROC of the probe's *low* score for abstaining."""
    s = scores[(scores.probe == "logreg") & scores.category.isin(["unknowable", "ill_posed"])]
    out = []
    for (config, tk, pos, cat), g in s.groupby(["config", "train_kind", "position", "category"]):
        y = g.abstained.to_numpy()
        if 0 < y.sum() < len(y):
            lo, hi = ci([auroc(-g.score.to_numpy()[i], y[i]) for i in boot_idx(len(g), seed=5)])
            out.append({"config": config, "train_kind": tk, "position": pos, "category": cat,
                        "n": len(g), "abstain_rate": y.mean(),
                        "auroc": auroc(-g.score.to_numpy(), y), "lo": lo, "hi": hi})
    return pd.DataFrame(out)


def save(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df.to_csv(RESULTS / f"{name}.csv", index=False, float_format="%.4f")
    return df


def main() -> None:
    fam = family_map()
    oof = pd.read_parquet(RESULTS / "probe_oof.parquet")
    scores = pd.read_parquet(RESULTS / "probe_scores.parquet")
    rows = pd.read_parquet(RESULTS / "rows.parquet")
    pd.set_option("display.width", 220)

    auc, gaps = oof_tables(oof, fam)
    save(auc, "probe_auroc")
    save(gaps, "probe_gaps")
    auc["cell"] = auc.apply(lambda r: f"{r.auroc_within:.2f} [{r.lo:.2f},{r.hi:.2f}]", axis=1)
    for strata in ("level", "strict"):
        print(f"== Training set (out-of-fold), within-{strata} AUROC [95% CI]")
        print(auc[auc.strata == strata].pivot(index=["config", "kind"], columns="position",
                                               values="cell")
              .reindex(columns=["meta", *POSITIONS]).to_string(), "\n")
    gaps["cell"] = gaps.apply(lambda r: f"{r['diff']:+.2f} [{r.lo:+.2f},{r.hi:+.2f}]", axis=1)
    print("== Paired differences (training set)")
    print(gaps.pivot_table(index=["config", "kind"], columns=["strata", "comparison"],
                           values="cell", aggfunc="first").to_string(), "\n")

    vm = save(vs_methods(scores, rows, fam), "probe_vs_methods")
    cols = ["config", "kind", "n", "n_right"] + [f"{c}_level" for c in (BEFORE, AFTER, "A", "D", "B", "E")]
    print("== Test set: within-level AUROC, probe vs Phase 1 methods (questions P answered)")
    print(vm[cols].round(2).to_string(index=False), "\n")

    tr = save(transfer(scores, fam), "probe_transfer")
    print("== Test set: transfer (train kind -> test kind), within-level AUROC")
    print(tr.pivot_table(index=["config", "position", "train_kind"], columns="test_kind",
                         values="auroc_within").round(2).to_string(), "\n")

    ab = save(abstain(scores), "probe_abstain")
    print("== Test set: low probe score predicts abstention (AUROC)")
    sel = ab[ab.train_kind == "pooled"]
    print(sel.pivot_table(index=["config", "category"], columns="position", values="auroc")
          .round(2).to_string())


if __name__ == "__main__":
    main()
