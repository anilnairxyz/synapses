"""E2 and E3 analysis (Mac; no model calls). Inputs: results/extract_<config>.parquet and the
swap logs in runs/4090/full/ (Phase 2), plus Phase 1's rows.parquet (A, B, D, E on the test
questions) and Phase 1B's probe_oof.parquet (probe scores on the training questions).

Readouts of confidence in a given answer (higher = more confident it is right):

  verbal_self   "your answer": the answer in the assistant turn, verbal 0-1 (swap_task)
  verbal_user   "my answer": the user gives the answer, verbal 0-1 (Sanz-Guerrero's framing)
  ptrue_<placement>_<shots>
                P(True), Kadavath's prompt; placement user (neutral) or assistant (own);
                0 or 8 worked examples (extract_answers). Answers whose variant put < 0.5 of
                the probability on the two letters are left out (rule fixed in the plan
                before analysis).
  probe         Phase 1B's hidden-state probe after the answer (ans_last), refitted
  probe_l0      the same probe at layer 0 (sees only the answer's token embeddings)
  probe_before  the probe before answering (gen): the same for every answer to a question,
                so it cannot separate answers given to one question (E3 only)
  logprob       the answer's mean token log-probability under the model (familiarity)
  A, D, B, E    Phase 1: stated after / before, sample agreement, token probability (E3)

Difficulty control: strict strata (kind x level x task family or relation), as in E1.

E2 (test questions; the injected correct and wrong answers to the same question):
  - mean confidence per source (own, correct, wrong) and readout;
  - separation: the paired win rate, the share of questions whose correct answer gets more
    confidence than the wrong one (ties count half; 0.5 = no separation), with 1,000-sample
    bootstrap intervals over questions, and the mean difference correct - wrong;
  - familiarity control: within-stratum AUROC of correct vs wrong with the strata also
    split by log-probability band (quintiles within kind), so only answers the model finds
    about equally likely are compared. (Regressing the log-probability out was planned, but
    with readouts that are mostly tied, e.g. verbal 1.0 for both answers, the residuals would
    break the ties by log-probability alone and bias the result; the bands avoid that.)

E3 (own answers; test questions, and for P(True) and the probe the ~5,700 training
questions with out-of-fold probe scores): within-stratum AUROC per readout with bootstrap
intervals, the paired difference to the probe (ceiling), and ECE / Brier for the readouts
that are probabilities. Labels as in Phase 1B: A and D on their own answers, the rest on
the plain answer P.

Tail (own answers, test): reference share q = share of answers with stated confidence A >= 0.9
in the cell; for each readout, the answers it flags as most confident (>= 0.9 for verbal
readouts; its top q for P(True), the probe and the other scores) and how many of them are
wrong, and the share of all wrong answers that are flagged. Plus, for E2, the share of
injected wrong answers given verbal confidence >= 0.9 under each attribution.

Secondary tail measure (added after the pre-specified one was degenerate for Qwen3-8B, which
states >= 0.9 on every answer): the error rate among each readout's most confident quarter.

Writes results/e2_means.csv, e2_separation.csv, e3_auroc.csv, e3_calibration.csv,
tail.csv, tail_quarter.csv, ptrue_excluded.csv, and figures e2_separation.png, e2_means.png, e3_readouts.png.

    uv run --extra evals python experiments/epic2/phase2_confidence_gap/analyze_swaps.py
"""

from __future__ import annotations

import glob

import numpy as np
import pandas as pd

import phase1

CONFIGS = ("qwen3-8b", "qwen3-1.7b", "qwen3-8b-base")
E2B = ("claude-haiku", "qwen3-8b-think", "gpt-4o", "gpt-5-nano", "grok-4-1-fast")  # verbal only (E2b)
SWAP_RUNS = {"claude-haiku": "api/full", "qwen3-8b-think": "4090/full", "gpt-4o": "foundry/full",
             "gpt-5-nano": "foundry/full", "grok-4-1-fast": "foundry/full"}
INSTRUCT = ("qwen3-8b", "qwen3-1.7b")
KINDS = ("recall", "computation", "perception")
PTRUE = ("ptrue_user_0", "ptrue_user_8", "ptrue_assistant_0", "ptrue_assistant_8")
MASS_MIN = 0.5
N_BOOT = 1000
SEED = 0
E2_READOUTS = ("verbal_self", "verbal_user", *PTRUE, "probe", "probe_l0", "logprob")
PROB_READOUTS = ("verbal_self", "verbal_user", *PTRUE, "A", "D")


# ----------------------------------------------------------------------------- statistics


def boot_idx(n: int, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(N_BOOT, n))


def ci(vals) -> tuple[float, float]:
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    return (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) if len(v) else (np.nan, np.nan)


def win_rate(c: np.ndarray, w: np.ndarray) -> float:
    """Share of pairs where the correct answer's score is higher; ties count half."""
    ok = ~(np.isnan(c) | np.isnan(w))
    if not ok.any():
        return np.nan
    c, w = c[ok], w[ok]
    return float(((c > w) + 0.5 * (c == w)).mean())


def within_auroc(s, y, strata) -> float:
    s, y, strata = np.asarray(s, float), np.asarray(y), np.asarray(strata)
    ok = ~np.isnan(s)
    return phase1.cal.stratified_auroc(s[ok], y[ok], strata[ok]) if ok.sum() else np.nan


def with_ci(stat, n: int, *arrays, seed: int = SEED) -> tuple[float, float, float]:
    arrays = [np.asarray(a) for a in arrays]
    est = stat(*arrays)
    lo, hi = ci([stat(*(a[i] for a in arrays)) for i in boot_idx(n, seed)])
    return est, lo, hi


# ----------------------------------------------------------------------------- data


def extract(config: str) -> pd.DataFrame:
    d = pd.read_parquet(phase1.RESULTS / f"extract_{config}.parquet")
    d["stratum"] = d.category + ":" + d.level.astype(str) + ":" + d.family.fillna("")
    for v in PTRUE:
        if v in d:
            d.loc[d[f"ab_mass_{v[6:]}"] < MASS_MIN, v] = np.nan
    return d.rename(columns={"probe_ans_last": "probe", "probe_l0_ans_last": "probe_l0",
                             "probe_gen": "probe_before", "logprob_mean": "logprob"})


def swaps_verbal(config: str) -> pd.DataFrame:
    """qid, source, verbal_self, verbal_user from the full swap log."""
    from inspect_ai.log import read_eval_log

    files = sorted(glob.glob(str(phase1.RUNS / SWAP_RUNS.get(config, "4090/full") / config / "*.eval")))
    if len(files) != 1:
        raise FileNotFoundError(f"expected one swap log for {config}, found {len(files)}")
    f = files[0]
    rows = []
    for s in read_eval_log(f).samples:
        m = s.scores["confidence_record"].metadata
        rows.append({"qid": m["qid"], "source": m["source"], "att": m["attribution"],
                     "conf": s.scores["confidence_record"].value["confidence"]})
    d = pd.DataFrame(rows).pivot_table(index=["qid", "source"], columns="att", values="conf",
                                       dropna=False).reset_index()
    return d.rename(columns={"self": "verbal_self", "user": "verbal_user"})


def assemble(config: str) -> pd.DataFrame:
    d = extract(config)
    if config in INSTRUCT:
        d = d.merge(swaps_verbal(config).assign(split="test"), on=["split", "qid", "source"], how="left")
    return d


def excluded_counts(d: pd.DataFrame, config: str) -> list[dict]:
    raw = pd.read_parquet(phase1.RESULTS / f"extract_{config}.parquet")
    return [{"config": config, "variant": v, "n": int(raw[v].notna().sum()),
             "excluded": int((raw[f"ab_mass_{v[6:]}"] < MASS_MIN).sum())} for v in PTRUE if v in raw]


# ----------------------------------------------------------------------------- E2


def e2(d: pd.DataFrame, config: str) -> tuple[list[dict], list[dict]]:
    t = d[d.split == "test"]
    means, sep = [], []
    readouts = [r for r in E2_READOUTS if r in t and t[r].notna().any()]
    has_lp = "logprob" in t and t.logprob.notna().any()
    for kind in KINDS:
        k = t[t.category == kind]
        for r in readouts:
            row = {"config": config, "kind": kind, "readout": r}
            for src in ("own", "correct", "wrong"):
                row[src] = float(k[k.source == src][r].mean())
            if r.startswith("verbal"):
                wr = k[k.source == "wrong"][r]
                row["wrong_ge_0.9"] = float((wr >= 0.9).mean())
            means.append(row)
        inj = k[k.source.isin(["correct", "wrong"])]
        piv = inj.pivot_table(index="qid", columns="source", values=list(readouts), dropna=False)
        if has_lp:
            bands = pd.qcut(inj.logprob.rank(method="first"), 5, labels=False).astype(str)
            strata_lp = (inj.stratum + ":lp" + bands).to_numpy()
        for r in readouts:
            c, w = piv[(r, "correct")].to_numpy(float), piv[(r, "wrong")].to_numpy(float)
            est, lo, hi = with_ci(win_rate, len(c), c, w)
            diff = c - w
            row = {"config": config, "kind": kind, "readout": r, "n_pairs": int((~np.isnan(diff)).sum()),
                   "win": est, "win_lo": lo, "win_hi": hi, "diff": float(np.nanmean(diff))}
            s, y = inj[r].to_numpy(float), inj.correct.to_numpy()
            row["auroc"], row["auroc_lo"], row["auroc_hi"] = with_ci(
                within_auroc, len(s), s, y, inj.stratum.to_numpy(), seed=1)
            if r != "logprob" and has_lp:
                row["auroc_lp"], row["auroc_lp_lo"], row["auroc_lp_hi"] = with_ci(
                    within_auroc, len(s), s, y, strata_lp, seed=2)
            sep.append(row)
        # ownership: the same answers, self minus user (verbal), assistant minus user (P(True))
        for a, b, name in (("verbal_self", "verbal_user", "own-vs-user verbal"),
                           ("ptrue_assistant_8", "ptrue_user_8", "own-vs-user P(True) 8-shot")):
            if a in k and k[a].notna().any():
                for src in ("own", "correct", "wrong"):
                    x = k[k.source == src]
                    dd = (x[a] - x[b]).to_numpy(float)
                    dd = dd[~np.isnan(dd)]
                    lo, hi = ci([dd[i].mean() for i in boot_idx(len(dd), 3)])
                    means.append({"config": config, "kind": kind, "readout": name, "source": src,
                                  "own_minus_user": float(dd.mean()), "lo": lo, "hi": hi})
    return means, sep


def verbal_only(config: str) -> pd.DataFrame:
    """E2b: the swap rows (test) with the verbal readouts; no P(True), probe or likelihood."""
    d = pd.read_json(phase1.DATA / "swaps.jsonl", lines=True)
    d = d[(d.config == config) & (d.split == "test")].drop(columns=["text"])
    d["stratum"] = d.category + ":" + d.level.astype(str) + ":" + d.family.fillna("")
    return d.merge(swaps_verbal(config), on=["qid", "source"], how="left")


# ----------------------------------------------------------------------------- E3


def phase1_methods(config: str) -> pd.DataFrame:
    r = pd.read_parquet(phase1.PHASE1_RESULTS / "rows.parquet")
    r = r[(r.config == config) & r.kind.isin(KINDS) & r.method.isin(["A", "D", "B", "E"])]
    out = r.pivot_table(index="qid", columns="method", values="conf")
    lab = r[r.method.isin(["A", "D"])].assign(ok=lambda x: np.where(x.abstained == 0, x.correct, np.nan))
    lab = lab.pivot_table(index="qid", columns="method", values="ok")
    return out.join(lab.add_prefix("y_")).reset_index()


def e3(d: pd.DataFrame, config: str) -> tuple[list[dict], list[dict]]:
    auc, cal = [], []
    test = d[(d.split == "test") & (d.source == "own")].merge(phase1_methods(config), on="qid", how="left")
    test = test.rename(columns={"verbal_self": "posthoc", "verbal_user": "user_framing"})
    oof = pd.read_parquet(phase1.PHASE1_RESULTS / "probe_oof.parquet")
    oof = oof[(oof.config == config) & (oof.probe == "logreg") & (oof.train_kind == oof.category)
              & oof.position.isin(["gen", "ans_last"])]
    oof = oof.pivot_table(index="qid", columns="position", values="score").rename(
        columns={"gen": "probe_before", "ans_last": "probe"}).reset_index()
    train = d[(d.split == "train") & (d.source == "own") & ~d.is_shot].drop(
        columns=["probe", "probe_before", "probe_l0"], errors="ignore").merge(oof, on="qid", how="left")
    readouts_test = ["A", "D", "posthoc", "user_framing", *PTRUE, "B", "E", "probe_before", "probe"]
    readouts_train = [*PTRUE, "probe_before", "probe"]
    for split, data, readouts in (("test", test, readouts_test), ("train", train, readouts_train)):
        for kind in KINDS:
            k = data[data.category == kind].reset_index(drop=True)
            st = k.stratum.to_numpy()
            idx = boot_idx(len(k), 4)
            for r in readouts:
                if r not in k or k[r].isna().all():
                    continue
                y = k[f"y_{r}"].to_numpy(float) if r in ("A", "D") else k.correct.to_numpy(float)
                s = k[r].to_numpy(float)
                ok = ~(np.isnan(s) | np.isnan(y))

                def stat(i, s=s, y=y, ok=ok):
                    j = i[ok[i]]
                    return within_auroc(s[j], y[j], st[j])

                row = {"config": config, "split": split, "kind": kind, "readout": r, "n": int(ok.sum()),
                       "n_right": int(y[ok].sum())}
                row["auroc"] = stat(np.arange(len(k)))
                row["auroc_lo"], row["auroc_hi"] = ci([stat(i) for i in idx])
                if r != "probe" and "probe" in k and r not in ("A", "D"):
                    p = k.probe.to_numpy(float)

                    def dstat(i, s=s, y=y, ok=ok, p=p):
                        j = i[ok[i] & ~np.isnan(p[i])]
                        return within_auroc(s[j], y[j], st[j]) - within_auroc(p[j], y[j], st[j])

                    row["minus_probe"] = dstat(np.arange(len(k)))
                    row["minus_probe_lo"], row["minus_probe_hi"] = ci([dstat(i) for i in idx])
                auc.append(row)
                if r in PROB_READOUTS:
                    cal.append({"config": config, "split": split, "kind": kind, "readout": r, "n": int(ok.sum()),
                                "mean_conf": float(s[ok].mean()), "acc": float(y[ok].mean()),
                                "ece": phase1.cal.ece(s[ok], y[ok]), "brier": phase1.cal.brier(s[ok], y[ok])})
    return auc, cal


# ----------------------------------------------------------------------------- tail


def tail(d: pd.DataFrame, config: str) -> list[dict]:
    if config not in INSTRUCT:
        return []
    t = d[(d.split == "test") & (d.source == "own")].merge(phase1_methods(config), on="qid", how="left")
    out = []
    for kind in KINDS:
        k = t[t.category == kind]
        q = float((k.A.dropna() >= 0.9).mean())
        for r in ("A", "verbal_self", "verbal_user", "ptrue_user_8", "ptrue_assistant_8", "B", "E", "probe"):
            y = (k.y_A if r == "A" else k.correct).to_numpy(float)
            s = k[r].to_numpy(float)
            ok = ~(np.isnan(s) | np.isnan(y))
            s, y = s[ok], y[ok]
            if r in ("A", "verbal_self", "verbal_user"):
                flag = s >= 0.9
            else:  # the same share as A's >= 0.9, by rank
                flag = pd.Series(s).rank(pct=True, method="average").to_numpy() > 1 - q
            out.append({"config": config, "kind": kind, "readout": r, "q_ref": q, "n": len(s),
                        "share_flagged": float(flag.mean()),
                        "wrong_among_flagged": float((y[flag] == 0).mean()) if flag.any() else np.nan,
                        "wrong_flagged": float(flag[y == 0].mean()) if (y == 0).any() else np.nan})
    return out


def tail_quarter(d: pd.DataFrame, config: str) -> list[dict]:
    """Secondary tail measure, added after the pre-specified one turned out degenerate for
    Qwen3-8B (stated confidence >= 0.9 on every answer, so matching its share flags all):
    the error rate among each readout's most confident quarter of own answers, against the
    overall error rate. Ties at the cut are shared out by average rank. Test questions (all
    readouts) and training questions (P(True), probe)."""
    test = d[(d.split == "test") & (d.source == "own")].merge(phase1_methods(config), on="qid", how="left")
    oof = pd.read_parquet(phase1.PHASE1_RESULTS / "probe_oof.parquet")
    oof = oof[(oof.config == config) & (oof.probe == "logreg") & (oof.train_kind == oof.category)
              & (oof.position == "ans_last")][["qid", "score"]].rename(columns={"score": "probe"})
    train = d[(d.split == "train") & (d.source == "own") & ~d.is_shot].drop(columns=["probe"]).merge(oof, on="qid")
    out = []
    for split, data, readouts in (("test", test, ("A", "verbal_self", "verbal_user", *PTRUE, "B", "E", "probe")),
                                  ("train", train, (*PTRUE, "probe"))):
        for kind in KINDS:
            k = data[data.category == kind]
            for r in readouts:
                if r not in k or k[r].isna().all():
                    continue
                y = (k.y_A if r == "A" else k.correct).to_numpy(float)
                sc = k[r].to_numpy(float)
                ok = ~(np.isnan(sc) | np.isnan(y))
                sc, y = sc[ok], y[ok]
                pct = pd.Series(sc).rank(pct=True, method="average").to_numpy()
                top = pct > 0.75
                w = np.where(top, 1.0, 0.0)
                # spread a tied block straddling the cut so exactly a quarter is "flagged"
                cut = np.sort(sc)[int(np.floor(0.75 * len(sc)))] if len(sc) else np.nan
                tied = sc == cut
                if tied.any():
                    need = 0.25 * len(sc) - (sc > cut).sum()
                    w = np.where(sc > cut, 1.0, np.where(tied, max(need, 0) / tied.sum(), 0.0))
                out.append({"config": config, "split": split, "kind": kind, "readout": r, "n": len(sc),
                            "error_overall": float(1 - y.mean()),
                            "error_top_quarter": float((w * (1 - y)).sum() / w.sum()) if w.sum() else np.nan})
    return out


# ----------------------------------------------------------------------------- figures


NAMES = {"verbal_self": "Verbal, \"your answer\"", "verbal_user": "Verbal, \"my answer\" (user)",
         "ptrue_assistant_8": "P(True), own (8-shot)", "ptrue_user_8": "P(True), neutral (8-shot)",
         "ptrue_assistant_0": "P(True), own (0-shot)", "ptrue_user_0": "P(True), neutral (0-shot)",
         "probe": "Hidden-state probe", "probe_l0": "Probe, layer 0 (control)",
         "logprob": "Answer likelihood (familiarity)", "A": "Stated with answer (A)",
         "D": "Stated before (D)", "posthoc": "Post-hoc verbal", "user_framing": "User framing",
         "B": "Sample agreement (B)", "E": "Token probability (E)", "probe_before": "Probe, before answering"}


def fig_separation(sep: pd.DataFrame) -> None:
    pl = phase1.plots
    import matplotlib.pyplot as plt

    order = ["verbal_self", "verbal_user", "ptrue_assistant_8", "ptrue_user_8", "probe", "probe_l0", "logprob"]
    colours = {"verbal_self": pl.ORANGE, "verbal_user": pl.YELLOW, "ptrue_assistant_8": pl.RED,
               "ptrue_user_8": pl.AQUA, "probe": pl.BLUE, "probe_l0": pl.MUTED, "logprob": pl.INK2}
    configs = [c for c in CONFIGS if c in set(sep.config)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.6), sharey=True, sharex=True)
    for c, kind in enumerate(KINDS):
        ax = axes[c]
        for j, cfg in enumerate(configs):
            g = sep[(sep.config == cfg) & (sep.kind == kind)].set_index("readout")
            for i, r in enumerate(order):
                if r not in g.index:
                    continue
                y = i + (j - (len(configs) - 1) / 2) * 0.25
                ax.plot([g.loc[r, "win_lo"], g.loc[r, "win_hi"]], [y, y], color=colours[r], lw=1.4, alpha=0.6)
                ax.plot(g.loc[r, "win"], y, ["o", "s", "D"][j], color=colours[r], ms=5.5, mec=pl.SURFACE, mew=0.8)
        ax.axvline(0.5, color=pl.MUTED, lw=0.8)
        ax.set_yticks(range(len(order)), [NAMES[r] for r in order])
        ax.set_xlim(0.2, 1.02)
        ax.set_title(kind.capitalize())
        if c == 0:
            ax.invert_yaxis()
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=m, ls="", color=pl.INK2, label=pl.MODEL_NAME[cfg])
               for m, cfg in zip(["o", "s", "D"], configs, strict=False)]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Does the readout prefer the right answer to the wrong one, for the same question?",
                 y=1.09, fontweight="bold")
    fig.supxlabel("Share of questions where the correct answer gets more confidence (0.5 = none)",
                  fontsize=10, color=pl.INK2)
    pl.caption(fig, "Each question gets a correct and a plausible wrong answer, built by construction and "
                    "written in the model's own format. 95% bootstrap intervals over the 150 questions per "
                    "kind. Verbal readouts: instruct models only. Base model: P(True) neutral, probe, "
                    "likelihood.", y=-0.04)
    fig.tight_layout()
    save(fig, "e2_separation")


def fig_ownership(sep: pd.DataFrame) -> None:
    """Key figure: separation under "your answer" vs "my answer", every model with verbal readouts,
    grouped into models that answer directly and models that reason first."""
    pl = phase1.plots
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory

    have = set(sep.config)
    groups = [("Answer directly", [c for c in ("qwen3-1.7b", "qwen3-8b", "claude-haiku", "gpt-4o", "grok-4-1-fast")
                                   if c in have]),
              ("Reason first", [c for c in ("qwen3-8b-think", "gpt-5-nano") if c in have])]
    rows, ypos, heads, y = [], [], [], 0.0
    for name, ms in groups:
        heads.append((name, y))
        y += 0.75
        for m in ms:
            rows.append(m)
            ypos.append(y)
            y += 1.0
        y += 0.35
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 1.2 + 0.52 * y), sharey=True, sharex=True)
    for c, kind in enumerate(KINDS):
        ax = axes[c]
        r0 = ypos[len(groups[0][1])] - 0.5 - 0.75
        ax.axhspan(r0, y - 0.35, color=pl.MUTED, alpha=0.12, lw=0)
        for cfg, yy in zip(rows, ypos, strict=True):
            g = sep[(sep.config == cfg) & (sep.kind == kind)].set_index("readout")
            for r, colour, dy in (("verbal_self", pl.ORANGE, -0.15), ("verbal_user", pl.BLUE, 0.15)):
                if r not in g.index:
                    continue
                ax.plot([g.loc[r, "win_lo"], g.loc[r, "win_hi"]], [yy + dy, yy + dy], color=colour, lw=1.5,
                        alpha=0.6)
                ax.plot(g.loc[r, "win"], yy + dy, "o", color=colour, ms=6.5, mec=pl.SURFACE, mew=1,
                        label=NAMES[r] if (c == 0 and cfg == rows[0]) else None)
        ax.axvline(0.5, color=pl.MUTED, lw=0.8)
        ax.set_yticks(ypos, [pl.MODEL_NAME[m] for m in rows])
        ax.set_xlim(0.35, 1.03)
        ax.set_ylim(y - 0.35, -0.3)
        ax.set_title(kind.capitalize())
    tr = blended_transform_factory(axes[0].transAxes, axes[0].transData)
    for name, yy in heads:
        axes[0].text(-0.02, yy + 0.15, name, transform=tr, ha="right", va="center", fontweight="bold",
                     fontsize=10.5, color=pl.INK if hasattr(pl, "INK") else "black")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Unless it reasons first, a model barely checks an answer placed in its own turn",
                 y=1.13, fontweight="bold")
    fig.supxlabel("Share of questions where the correct answer gets more confidence (0.5 = none)",
                  fontsize=10, color=pl.INK2)
    fig.tight_layout()
    save(fig, "e2_ownership")


def fig_means(means: pd.DataFrame) -> None:
    pl = phase1.plots
    import matplotlib.pyplot as plt

    m = means[means.readout.isin(["verbal_self", "verbal_user"]) & means.config.isin(INSTRUCT)]
    fig, axes = plt.subplots(len(INSTRUCT), 3, figsize=(10, 5.2), sharey=True)
    for r_, cfg in enumerate(INSTRUCT):
        for c, kind in enumerate(KINDS):
            ax = axes[r_, c]
            g = m[(m.config == cfg) & (m.kind == kind)].set_index("readout")
            x = np.arange(3)
            for j, (rd, colour) in enumerate((("verbal_self", pl.ORANGE), ("verbal_user", pl.YELLOW))):
                if rd in g.index:
                    vals = [g.loc[rd, s] for s in ("own", "correct", "wrong")]
                    ax.bar(x + (j - 0.5) * 0.36, vals, 0.34, color=colour, label=NAMES[rd], edgecolor=pl.SURFACE)
            ax.set_xticks(x, ["Own", "Correct", "Wrong"])
            ax.set_ylim(0, 1.05)
            if r_ == 0:
                ax.set_title(kind.capitalize())
            if c == 0:
                ax.set_ylabel(f"{pl.MODEL_NAME[cfg]}\nmean verbal confidence")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Confidence in a given answer: as the model's own vs as the user's", y=1.09, fontweight="bold")
    pl.caption(fig, "Own: the model's Phase 1 answer. Correct and wrong: injected, in the model's format. "
                    "\"Your answer\": the answer in the assistant turn. \"My answer\": the user gives it.", y=-0.03)
    fig.tight_layout()
    save(fig, "e2_means")


def fig_e3(auc: pd.DataFrame) -> None:
    pl = phase1.plots
    import matplotlib.pyplot as plt

    order = ["A", "posthoc", "user_framing", "ptrue_user_0", "ptrue_user_8", "ptrue_assistant_0",
             "ptrue_assistant_8", "B", "E", "probe_before", "probe"]
    t = auc[auc.split == "test"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 5), sharey=True, sharex=True)
    for c, kind in enumerate(KINDS):
        ax = axes[c]
        for j, cfg in enumerate(CONFIGS):
            g = t[(t.config == cfg) & (t.kind == kind)].set_index("readout")
            for i, r in enumerate(order):
                if r not in g.index or pd.isna(g.loc[r, "auroc"]):
                    continue
                y = i + (j - 1) * 0.25
                colour = pl.BLUE if r.startswith("probe") else pl.ORANGE if r.startswith("ptrue") else pl.INK2
                ax.plot([g.loc[r, "auroc_lo"], g.loc[r, "auroc_hi"]], [y, y], color=colour, lw=1.3, alpha=0.6)
                ax.plot(g.loc[r, "auroc"], y, ["o", "s", "D"][j], color=colour, ms=5, mec=pl.SURFACE, mew=0.8)
        ax.axvline(0.5, color=pl.MUTED, lw=0.8)
        ax.set_yticks(range(len(order)), [NAMES[r] for r in order])
        ax.set_xlim(0.25, 1.0)
        ax.set_title(kind.capitalize())
        if c == 0:
            ax.invert_yaxis()
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=m, ls="", color=pl.INK2, label=pl.MODEL_NAME[cfg])
               for m, cfg in zip(["o", "s", "D"], CONFIGS, strict=True)]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Which way of asking recovers what the model knows? (own answers, within levels)",
                 y=1.09, fontweight="bold")
    fig.supxlabel("Within-level AUROC for the model's own answer being right", fontsize=10, color=pl.INK2)
    pl.caption(fig, "Test questions (150 per kind); 95% bootstrap intervals. Blue: hidden-state probe "
                    "(the ceiling); orange: P(True); grey: Phase 1 methods and the verbal readouts.", y=-0.03)
    fig.tight_layout()
    save(fig, "e3_readouts")


def save(fig, name: str) -> None:
    import matplotlib.pyplot as plt

    out = phase1.RESULTS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / f"{name}.png")


# ----------------------------------------------------------------------------- main


def main() -> None:
    means, sep, auc, cal, tl, tq, exc = [], [], [], [], [], [], []
    for config in CONFIGS:
        d = assemble(config)
        exc += excluded_counts(d, config)
        m, s = e2(d, config)
        means += m
        sep += s
        a, c = e3(d, config)
        auc += a
        cal += c
        tl += tail(d, config)
        tq += tail_quarter(d, config)
        print("done", config)
    for config in E2B:
        try:
            d = verbal_only(config)
        except FileNotFoundError as e:
            print("skipped", config, "-", e)
            continue
        m, s = e2(d, config)
        means += m
        sep += s
        print("done", config, "(verbal only)")
    out = {"e2_means": means, "e2_separation": sep, "e3_auroc": auc, "e3_calibration": cal,
           "tail": tl, "tail_quarter": tq, "ptrue_excluded": exc}
    frames = {k: pd.DataFrame(v) for k, v in out.items()}
    for k, f in frames.items():
        f.to_csv(phase1.RESULTS / f"{k}.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 300, "display.max_columns", 30):
        s = frames["e2_separation"]
        print(s[["config", "kind", "readout", "n_pairs", "win", "win_lo", "win_hi", "auroc_lp"]].round(2).to_string(index=False))
    fig_separation(frames["e2_separation"][frames["e2_separation"].config.isin(CONFIGS)])
    fig_ownership(frames["e2_separation"])
    fig_means(frames["e2_means"])
    fig_e3(frames["e3_auroc"])


if __name__ == "__main__":
    main()
