"""E4 Part A: which confidence report can be read from the answer's state, and along which
direction? (PC, CPU; needs the E2 states from extract_answers.py and Phase 1's training states.)

For each config (qwen3-8b, qwen3-1.7b) and kind, on the E2 states of every given answer
(own, correct, wrong; Phase 1's prompt plus the answer; position ans_last at the layer the
Phase 1B probe chose for that kind):

  correctness probe  Phase 1B's logistic probe, refitted on the Phase 1 training states (as in
                     extract_answers.py); its weight vector in the raw state space is the
                     correctness direction. A probe trained on shuffled labels gives a baseline
                     direction (fitted directions share the states' dominant axes, so a random
                     vector is too weak a baseline for the cosine).
  confidence readout ridge regression from the states to a verbal confidence: "my answer"
                     (verbal_user), "your answer" (verbal_self) if it varies, and A (stated with
                     the answer, Phase 1) on the model's own answers. Out-of-fold predictions,
                     5 folds grouped by question (a question's own, correct and wrong answers
                     stay in one fold); alpha by inner cross-validation.

Measures (strata = kind x level x family or relation, as in E1 to E3; 95% intervals from
1,000 bootstrap resamples of questions):
  readable     within-stratum Spearman of the out-of-fold readout with the confidence
  cos          cosine of the readout's direction (fitted on all rows) with the correctness
               direction, raw state space; cos_shuffled, the same with the shuffled-label probe
  probe_vs_conf  within-stratum Spearman of the probe's score with the confidence
  readout_auroc  within-stratum AUROC of the out-of-fold readout for the answer being right

Also writes, for steer.py (Part B), each probe's unit direction in the raw state space, its
layer, and the median norm of the states at that layer and position:
results/e4_probe_dirs.npz. Tables: results/e4_directions.csv.

    uv run --extra evals python experiments/epic2/phase2_confidence_gap/directions.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import analyze_swaps as an
import analyze_stereotype as e1
import phase1

CONFIGS = ("qwen3-8b", "qwen3-1.7b")
KINDS = ("recall", "computation", "perception")
STATES = phase1.HERE / "states"
ALPHAS = np.logspace(0, 6, 13)
N_BOOT = 1000
SEED = 0


def raw_direction(pipe) -> np.ndarray:
    """A standardise-then-linear pipeline's weights expressed in the raw state space."""
    scaler, lin = pipe[0], pipe[-1]
    w = np.ravel(lin.coef_) / scaler.scale_
    return w / np.linalg.norm(w)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def boot_questions(qids: np.ndarray, seed: int) -> list[np.ndarray]:
    """Row indices for bootstrap resamples of whole questions."""
    rng = np.random.default_rng(seed)
    uq = np.unique(qids)
    by_q = {q: np.flatnonzero(qids == q) for q in uq}
    return [np.concatenate([by_q[q] for q in rng.choice(uq, size=len(uq))]) for _ in range(N_BOOT)]


def rows_and_states(config: str) -> tuple[pd.DataFrame, np.ndarray]:
    meta = pd.read_parquet(STATES / config / "swaps.parquet")  # row order of swaps.npy
    states = np.load(STATES / config / "swaps.npy", mmap_mode="r")
    ext = pd.read_parquet(phase1.RESULTS / f"extract_{config}.parquet")
    ext = ext[ext.split == "test"][["qid", "source", "correct", "probe_ans_last"]]
    d = meta[["qid", "category", "level", "family", "source"]].merge(ext, on=["qid", "source"], how="left",
                                                                     validate="one_to_one")
    d = d.merge(an.swaps_verbal(config), on=["qid", "source"], how="left")
    a = pd.read_parquet(phase1.PHASE1_RESULTS / "rows.parquet")
    a = a[(a.config == config) & (a.method == "A") & (a.abstained == 0)][["qid", "conf"]].rename(columns={"conf": "A"})
    d = d.merge(a, on="qid", how="left")
    d.loc[d.source != "own", "A"] = np.nan  # A is about the model's own answer only
    d["stratum"] = d.category + ":" + d.level.astype(str) + ":" + d.family.fillna("")
    return d, states


def probe_fit(config: str, kind: str, layer: int, C: float, shuffled: bool = False):
    pr = phase1.probe
    tr_states, tr_rows = pr.load(config, "train")
    tr_rows = tr_rows[tr_rows.n_answer > 0]
    a = (tr_rows.category == kind).to_numpy()
    X = np.asarray(tr_states[tr_rows.index.to_numpy()[a], layer, list(pr.POSITIONS).index("ans_last"), :],
                   dtype=np.float32)
    y = tr_rows.y.to_numpy()[a]
    if shuffled:
        y = np.random.default_rng(SEED).permutation(y)
    return pr.logreg(C).fit(X, y)


def readout(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(out-of-fold predictions, raw-space unit direction fitted on all rows)."""
    def model():
        return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, groups):
        oof[te] = model().fit(X[tr], y[tr]).predict(X[te])
    return oof, raw_direction(model().fit(X, y))


def main() -> None:
    chosen = pd.read_parquet(phase1.PHASE1_RESULTS / "probe_scores.parquet")
    pi = list(phase1.probe.POSITIONS).index("ans_last")
    out, dirs = [], {}
    for config in CONFIGS:
        d, states = rows_and_states(config)
        ch = (chosen[(chosen.config == config) & (chosen.probe == "logreg") & (chosen.train_kind == chosen.category)
                     & (chosen.position == "ans_last")].groupby("train_kind")[["layer", "C"]].first())
        for kind in KINDS:
            layer, C = int(ch.loc[kind, "layer"]), float(ch.loc[kind, "C"])
            k = d[d.category == kind].reset_index()
            X = np.asarray(states[k["index"].to_numpy(), layer, pi, :], dtype=np.float32)
            probe = probe_fit(config, kind, layer, C)
            w_probe = raw_direction(probe)
            w_shuf = raw_direction(probe_fit(config, kind, layer, C, shuffled=True))
            dirs[f"{config}|{kind}|dir"] = w_probe
            dirs[f"{config}|{kind}|layer"] = np.array(layer)
            dirs[f"{config}|{kind}|norm"] = np.array(float(np.median(np.linalg.norm(X, axis=1))))
            p_score = probe.decision_function(X)
            check = np.corrcoef(p_score, k.probe_ans_last)[0, 1]  # same probe as extract_answers
            print(f"{config} {kind}: layer {layer}, probe reproduces extract_answers r = {check:.4f}")
            for target in ("verbal_user", "verbal_self", "A"):
                m = k[target].notna().to_numpy()
                y = k[target].to_numpy(float)[m]
                row = {"config": config, "kind": kind, "target": target, "layer": layer, "n": int(m.sum()),
                       "n_distinct": int(len(np.unique(y))), "sd": float(y.std()) if len(y) else np.nan}
                if row["n"] < 30 or row["n_distinct"] < 3 or row["sd"] < 0.02:
                    out.append(row | {"note": "confidence barely varies: nothing to read"})
                    continue
                Xm, st, qid = X[m], k.stratum.to_numpy()[m], k.qid.to_numpy()[m]
                corr = k.correct.to_numpy(float)[m]
                oof, w_conf = readout(Xm, y, qid)
                ps = p_score[m]
                idx = boot_questions(qid, SEED)

                def stat_ci(f, idx=idx):
                    vals = np.array([f(i) for i in idx], float)
                    vals = vals[~np.isnan(vals)]
                    return tuple(np.quantile(vals, [0.025, 0.975])) if len(vals) else (np.nan, np.nan)

                full = np.arange(len(y))
                readable = lambda i: e1.within_spearman(y[i], oof[i], st[i])  # noqa: E731
                pvc = lambda i: e1.within_spearman(y[i], ps[i], st[i])  # noqa: E731
                rauc = lambda i: an.within_auroc(oof[i], corr[i], st[i])  # noqa: E731
                row |= {"readable": readable(full), "cos": cosine(w_conf, w_probe),
                        "cos_shuffled": cosine(w_conf, w_shuf), "cos_random_sd": 1 / np.sqrt(len(w_probe)),
                        "probe_vs_conf": pvc(full), "readout_auroc": rauc(full)}
                row["readable_lo"], row["readable_hi"] = stat_ci(readable)
                row["probe_vs_conf_lo"], row["probe_vs_conf_hi"] = stat_ci(pvc)
                row["readout_auroc_lo"], row["readout_auroc_hi"] = stat_ci(rauc)
                out.append(row)
    res = pd.DataFrame(out)
    phase1.RESULTS.mkdir(parents=True, exist_ok=True)
    res.to_csv(phase1.RESULTS / "e4_directions.csv", index=False)
    np.savez(phase1.RESULTS / "e4_probe_dirs.npz", **dirs)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        cols = ["config", "kind", "target", "n", "n_distinct", "readable", "readable_lo", "readable_hi", "cos",
                "cos_shuffled", "probe_vs_conf", "readout_auroc", "note"]
        print(res[[c for c in cols if c in res]].round(3).to_string(index=False))
    print("wrote results/e4_directions.csv and results/e4_probe_dirs.npz")


if __name__ == "__main__":
    main()
