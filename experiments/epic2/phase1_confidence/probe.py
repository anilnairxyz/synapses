"""Phase 1B step 5: linear probes on the hidden states (runs on the PC, CPU).

Reads states/<config>/{train,test}.npy|.parquet (extract_states.py) and the judged P labels
from the logs (runs/4090/train and runs/4090/full). Label: 1 if the plain answer is
correct, 0 if it is wrong or an abstention (as P is scored in Phase 1).

Per config, position (q_last, gen, ans_last, ans_mean) and training kind (recall,
computation, perception, or all three pooled):

1. Layer sweep: standardise, logistic regression (L2, C = SWEEP_C, balanced class weights)
   on 80% of the training set; within-level AUROC on the other 20% picks the layer. Test
   AUROCs per layer are recorded too, for the curves (not used for any choice).
2. At the chosen layer: C from {1e-4, 1e-3, 1e-2, 1e-1} by 5-fold CV on the whole training
   set, refit, and score every test question (all kinds, so the same probe gives the
   cross-kind and the abstention results).
3. Controls at the chosen layer: the same probe trained on shuffled labels; and, per kind,
   a logistic regression on what the question builder knows (metadata) instead of states.
   --mlp adds Ni et al.'s MLP (512-64-32, Adam) at the chosen layer as a check.

The test set is never used for a choice. Writes (committed; small):
  results/probe_layers.csv    config, train_kind, position, layer, val and test AUROCs
  results/probe_scores.parquet one row per (config, probe, train_kind, position, test question)

--oof (after the main run): out-of-fold scores on the TRAINING questions at the layer and C
already chosen. 5-fold CV, so each training question is scored by a probe that never saw
it; about 2,000 questions per kind instead of 150, for the probe-only comparisons (before
vs after, base vs instruct, across kinds). The layer was chosen on a 20% split of the same
data, a small optimistic bias shared by every position. Writes results/probe_oof.parquet.

Usage (repo root, PC):
  uv run --extra evals python experiments/epic2/phase1_confidence/probe.py qwen3-8b [--mlp]
  uv run --extra evals python experiments/epic2/phase1_confidence/probe.py qwen3-8b --oof
"""

from __future__ import annotations

import argparse
import glob
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))

from synapses.calibration import auroc, stratified_auroc  # noqa: E402

NO_LABEL = ("ERROR", "UNPARSED")  # as confidence_task.NO_LABEL (not imported: needs inspect_ai)

STATES = HERE / "states"
RUNS = HERE / "runs" / "4090"
RESULTS = HERE / "results"
POSITIONS = ("q_last", "gen", "ans_last", "ans_mean")
KINDS = ("recall", "computation", "perception")
TRAIN_KINDS = (*KINDS, "pooled")
SWEEP_C = 1e-3
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1)
SEED = 0


# ----------------------------------------------------------------------------- data


def labels(config: str, split: str) -> pd.DataFrame:
    """qid, category, level, y (P correct), abstained, and the question metadata."""
    from inspect_ai.log import read_eval_log

    run = {"train": "train", "test": "full"}[split]
    (f,) = glob.glob(str(RUNS / run / config / "*.eval"))
    rows = []
    for s in read_eval_log(f).samples:
        q = s.metadata["q"]
        if q["category"] == "forecastable":  # not probed (no P answer)
            continue
        lab = s.scores["confidence_scorer"].metadata["labels"]["P"]
        if lab is None or lab in NO_LABEL:
            raise SystemExit(f"{f}: {s.id} has P label {lab}; judge the log first")
        rows.append({"qid": s.id, "category": q["category"], "level": q["level"],
                     "y": int(lab == "CORRECT"), "abstained": int(lab == "ABSTAINS"),
                     "label": lab, "meta": q["meta"]})
    return pd.DataFrame(rows)


def load(config: str, split: str) -> tuple[np.ndarray, pd.DataFrame]:
    """The states (memory-mapped) and their rows, joined with labels."""
    states = np.load(STATES / config / f"{split}.npy", mmap_mode="r")
    rows = pd.read_parquet(STATES / config / f"{split}.parquet")
    lab = labels(config, split).drop(columns=["category", "level"])
    rows = rows.merge(lab, on="qid", how="left", validate="one_to_one")
    if rows.y.isna().any():
        raise SystemExit(f"{config}/{split}: states without labels")
    return states, rows


def metadata_features(rows: pd.DataFrame, kind: str) -> pd.DataFrame:
    """What the question builder knows about each question, per kind (the difficulty control)."""
    m = rows.meta
    if kind == "recall":
        return pd.DataFrame({"log10_pop": [x["log10_pop"] for x in m],
                             "relation": [x["relation"] for x in m]})
    if kind == "computation":
        return pd.DataFrame({"da": [x["digits"][0] for x in m], "db": [x["digits"][1] for x in m],
                             "log_a": [math.log10(x["a"]) for x in m],
                             "log_b": [math.log10(x["b"]) for x in m]})
    if kind == "perception":
        return pd.DataFrame({"length": [x["length"] for x in m], "zipf": [x["zipf"] for x in m],
                             "family": [x["family"] for x in m]})
    raise ValueError(kind)


# ----------------------------------------------------------------------------- probes


def logreg(C: float):
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=C, class_weight="balanced", max_iter=2000))


def within(score, y, level) -> float:
    return stratified_auroc(score, y, level)


def sweep_layer(X: np.ndarray, y: np.ndarray, level: np.ndarray, tr: np.ndarray, va: np.ndarray,
                Xte: np.ndarray, yte: np.ndarray, lte: np.ndarray) -> dict:
    """Fit on the train part at one layer; AUROCs on the validation part and the test set."""
    X = X.astype(np.float32)
    clf = logreg(SWEEP_C).fit(X[tr], y[tr])
    sv = clf.decision_function(X[va])
    st = clf.decision_function(Xte.astype(np.float32))
    return {"val_auroc": auroc(sv, y[va]), "val_within": within(sv, y[va], level[va]),
            "test_auroc": auroc(st, yte) if len(set(yte)) > 1 else np.nan,
            "test_within": within(st, yte, lte) if len(set(yte)) > 1 else np.nan}


def choose_c(X: np.ndarray, y: np.ndarray) -> float:
    """C with the best mean 5-fold CV AUROC on the training set."""
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
    best, best_c = -1.0, C_GRID[0]
    for C in C_GRID:
        s = np.mean([auroc(logreg(C).fit(X[a], y[a]).decision_function(X[b]), y[b])
                     for a, b in folds.split(X, y)])
        if s > best:
            best, best_c = s, C
    return best_c


def mlp_scores(X: np.ndarray, y: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    """Ni et al.'s estimator: MLP 512-64-32, ReLU, Adam lr 5e-5, 30 epochs, 3 seeds averaged."""
    from sklearn.neural_network import MLPClassifier

    sc = StandardScaler().fit(X)
    out = []
    for seed in (0, 42, 100):
        m = MLPClassifier((512, 64, 32), learning_rate_init=5e-5, max_iter=30, random_state=seed)
        m.fit(sc.transform(X), y)
        out.append(m.predict_proba(sc.transform(Xte))[:, 1])
    return np.mean(out, axis=0)


# ----------------------------------------------------------------------------- driver


def run_config(config: str, mlp: bool, n_jobs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr_states, tr_rows = load(config, "train")
    te_states, te_rows = load(config, "test")
    tr_rows = tr_rows[tr_rows.n_answer > 0]
    te_keep = (te_rows.n_answer > 0).to_numpy()
    te_rows = te_rows[te_keep].reset_index(drop=True)
    n_layers = tr_states.shape[1]
    layer_rows, score_rows = [], []
    base_cols = te_rows[["qid", "category", "level", "y", "abstained"]]

    def emit(probe, kind, pos, layer, C, scores):
        score_rows.append(base_cols.assign(config=config, probe=probe, train_kind=kind,
                                           position=pos, layer=layer, C=C, score=scores))

    for kind in KINDS:  # metadata control: no states involved
        a = tr_rows.category.to_numpy() == kind
        b = (te_rows.category == kind).to_numpy()
        ftr, fte = metadata_features(tr_rows[a], kind), metadata_features(te_rows[b], kind)
        cat = [c for c in ftr.columns if not pd.api.types.is_numeric_dtype(ftr[c])]
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(ftr[cat]) if cat else None

        def mat(f, enc=enc, cat=cat):
            num = f.drop(columns=cat).to_numpy(float)
            return np.hstack([num, enc.transform(f[cat])]) if cat else num

        clf = logreg(1.0).fit(mat(ftr), tr_rows.y.to_numpy()[a])
        s = np.full(len(te_rows), np.nan)
        s[b] = clf.decision_function(mat(fte))
        emit("meta", kind, "meta", -1, 1.0, s)

    for pos_i, pos in enumerate(POSITIONS):
        t0 = time.time()
        Xall = np.asarray(tr_states[tr_rows.index.to_numpy(), :, pos_i, :])  # [n, layers, hidden]
        Tall = np.asarray(te_states[np.flatnonzero(te_keep), :, pos_i, :])
        for kind in TRAIN_KINDS:
            a = (tr_rows.category.isin(KINDS) if kind == "pooled"
                 else tr_rows.category == kind).to_numpy()
            X, y = Xall[a], tr_rows.y.to_numpy()[a]
            level = (tr_rows.category + tr_rows.level.astype(str)).to_numpy()[a]
            tr, va = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
            b = (te_rows.category.isin(KINDS) if kind == "pooled"
                 else te_rows.category == kind).to_numpy()
            lte = (te_rows.category + te_rows.level.astype(str)).to_numpy()[b]
            res = Parallel(n_jobs=n_jobs)(
                delayed(sweep_layer)(X[:, L], y, level, tr, va, Tall[b][:, L],
                                     te_rows.y.to_numpy()[b], lte)
                for L in range(n_layers))
            for L, r in enumerate(res):
                layer_rows.append({"config": config, "train_kind": kind, "position": pos,
                                   "layer": L, **r})
            best = int(np.nanargmax([r["val_within"] for r in res]))
            XL = X[:, best].astype(np.float32)
            C = choose_c(XL, y)
            emit("logreg", kind, pos, best, C,
                 logreg(C).fit(XL, y).decision_function(Tall[:, best].astype(np.float32)))
            y_shuf = np.random.default_rng(SEED).permutation(y)
            emit("shuffled", kind, pos, best, C,
                 logreg(C).fit(XL, y_shuf).decision_function(Tall[:, best].astype(np.float32)))
            if mlp:
                emit("mlp", kind, pos, best, np.nan,
                     mlp_scores(XL, y, Tall[:, best].astype(np.float32)))
            print(f"  {pos:8s} {kind:11s} layer {best:2d} C {C:g} "
                  f"val within-level AUROC {res[best]['val_within']:.3f}", flush=True)
        print(f"  {pos} done in {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(layer_rows), pd.concat(score_rows, ignore_index=True)


def oof_scores(X: np.ndarray, y: np.ndarray, C: float) -> np.ndarray:
    """Out-of-fold logistic-probe scores (5-fold, stratified)."""
    out = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        out[b] = logreg(C).fit(X[a], y[a]).decision_function(X[b])
    return out


def run_oof(config: str, chosen: pd.DataFrame) -> pd.DataFrame:
    """Out-of-fold scores on the training set at each probe's chosen layer and C, plus the
    metadata control."""
    states, rows = load(config, "train")
    rows = rows[rows.n_answer > 0]
    base_cols = rows[["qid", "category", "level", "y", "abstained"]].reset_index(drop=True)
    out = []
    for kind in KINDS:
        a = (rows.category == kind).to_numpy()
        f = metadata_features(rows[a], kind)
        cat = [c for c in f.columns if not pd.api.types.is_numeric_dtype(f[c])]
        num = f.drop(columns=cat).to_numpy(float)
        X = np.hstack([num, OneHotEncoder(sparse_output=False).fit_transform(f[cat])]) if cat else num
        s = np.full(len(rows), np.nan)
        s[a] = oof_scores(X, rows.y.to_numpy()[a], 1.0)
        out.append(base_cols.assign(config=config, probe="meta", train_kind=kind, position="meta",
                                    layer=-1, score=s))
    for r in chosen.itertuples():
        t0 = time.time()
        a = (rows.category.isin(KINDS) if r.train_kind == "pooled"
             else rows.category == r.train_kind).to_numpy()
        pos_i = POSITIONS.index(r.position)
        X = np.asarray(states[rows.index.to_numpy()[a], int(r.layer), pos_i, :], dtype=np.float32)
        s = np.full(len(rows), np.nan)
        s[a] = oof_scores(X, rows.y.to_numpy()[a], r.C)
        out.append(base_cols.assign(config=config, probe="logreg", train_kind=r.train_kind,
                                    position=r.position, layer=int(r.layer), score=s))
        print(f"  {r.position:8s} {r.train_kind:11s} layer {int(r.layer):2d} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return pd.concat(out, ignore_index=True).dropna(subset=["score"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--mlp", action="store_true", help="add Ni et al.'s MLP at the chosen layer")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--oof", action="store_true",
                    help="out-of-fold scores on the training set at the chosen layers")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    if args.oof:
        oof_f = RESULTS / "probe_oof.parquet"
        sc = pd.read_parquet(RESULTS / "probe_scores.parquet")
        for config in args.configs:
            print(f"=== {config} out-of-fold {time.strftime('%H:%M')}", flush=True)
            chosen = (sc[(sc.config == config) & (sc.probe == "logreg")]
                      .drop_duplicates(["train_kind", "position"])[["train_kind", "position", "layer", "C"]])
            oof = run_oof(config, chosen)
            if oof_f.exists():
                oof = pd.concat([pd.read_parquet(oof_f).query("config != @config"), oof])
            oof.to_parquet(oof_f, index=False)
            print(f"  wrote {oof_f.name}", flush=True)
        return
    layers_f, scores_f = RESULTS / "probe_layers.csv", RESULTS / "probe_scores.parquet"
    for config in args.configs:
        print(f"=== {config} {time.strftime('%H:%M')}", flush=True)
        layers, scores = run_config(config, args.mlp, args.n_jobs)
        # Replace this config's rows, keep the others (configs can be run one at a time).
        if layers_f.exists():
            layers = pd.concat([pd.read_csv(layers_f).query("config != @config"), layers])
        if scores_f.exists():
            scores = pd.concat([pd.read_parquet(scores_f).query("config != @config"), scores])
        layers.to_csv(layers_f, index=False)
        scores.to_parquet(scores_f, index=False)
        print(f"  wrote {layers_f.name}, {scores_f.name}", flush=True)


if __name__ == "__main__":
    main()
