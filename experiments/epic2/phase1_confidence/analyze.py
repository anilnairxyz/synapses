"""Step 3 (+4): metrics, tables and plots from Confidence Audit Inspect logs.

Reads every .eval log under --log-dir, one row per (run, question), then reports per run
and tier: accuracy, abstention, Brier (+ Murphy decomposition), ECE and AUROC for verbal
confidence, and for sample-consistency confidence when the run used k_samples > 0.

Pool definitions:
  attempted  parsed, not abstained, not judged a decline. These carry a confidence that is
             scored against correctness. Unknowable-tier commitments count as incorrect.
  all        every parsed sample; used for the Kalai penalised score, where abstaining is 0.

Usage:
  uv run python experiments/epic2/phase1_confidence/analyze.py --log-dir outputs/epic2/confidence
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inspect_ai.log import list_eval_logs, read_eval_log

from synapses.calibration import auroc, brier, ece, murphy, penalised_score, reliability, selective_curve

ROOT = Path(__file__).resolve().parents[3]
SCORER = "confidence_scorer"


def load(log_dir: Path) -> pd.DataFrame:
    rows = []
    latest: dict[str, object] = {}  # run name -> newest successful log (re-runs replace)
    for info in list_eval_logs(str(log_dir)):
        log = read_eval_log(info)
        if log.status != "success" or not log.samples:
            continue
        args = log.eval.task_args or {}
        run = f"{log.eval.model.split('/')[-1]}·{args.get('elicit', 'post')}·k{args.get('k_samples', 0)}"
        if run not in latest or log.eval.created > latest[run].eval.created:
            latest[run] = log
    for run, log in latest.items():
        args = log.eval.task_args or {}
        for s in log.samples:
            sc = s.scores.get(SCORER) if s.scores else None
            if sc is None:
                continue
            rows.append({
                "run": run,
                "model": log.eval.model,
                "elicit": args.get("elicit", "post"),
                "k": int(args.get("k_samples", 0) or 0),
                "id": s.id,
                "tier": sc.metadata["tier"],
                "source": sc.metadata["source"],
                "answer": sc.answer,
                **sc.value,
            })
    if not rows:
        raise SystemExit(f"no successful confidence_audit logs in {log_dir}")
    df = pd.DataFrame(rows)
    df.loc[df.consistency < 0, "consistency"] = np.nan
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    groups = list(df.groupby(["run", "tier"])) + [((r, "ALL"), g) for r, g in df.groupby("run")]
    for (run, tier), g in groups:
        parsed = g[g.parse_ok == 1]
        att = parsed[parsed.attempted == 1]
        row = {
            "run": run,
            "tier": tier,
            "n": len(g),
            "parse_ok": parsed.shape[0] / len(g),
            "abstain": parsed.abstained.mean(),
            "acc_attempted": att.correct.mean() if len(att) else np.nan,
            "mean_conf": att.confidence.mean() if len(att) else np.nan,
        }
        if len(att):
            d = murphy(att.confidence, att.correct)
            row |= {
                "brier": brier(att.confidence, att.correct),
                "ece": ece(att.confidence, att.correct),
                "auroc": auroc(att.confidence, att.correct),
                "murphy_rel": d.reliability,
                "murphy_res": d.resolution,
            }
            cons = att.dropna(subset=["consistency"])
            if len(cons):
                row |= {
                    "brier_cons": brier(cons.consistency, cons.correct),
                    "ece_cons": ece(cons.consistency, cons.correct),
                    "auroc_cons": auroc(cons.consistency, cons.correct),
                }
        out.append(row)
    order = {"easy": 0, "hard": 1, "unknowable": 2, "ALL": 3}
    return pd.DataFrame(out).sort_values(["run", "tier"], key=lambda c: c.map(order) if c.name == "tier" else c)


def plot_reliability(df: pd.DataFrame, path: Path) -> None:
    runs = sorted(df.run.unique())
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 4.2), squeeze=False)
    for ax, run in zip(axes[0], runs):
        att = df[(df.run == run) & (df.parse_ok == 1) & (df.attempted == 1)]
        ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
        for col, label, c in [("confidence", "verbal", "C0"), ("consistency", "consistency", "C1")]:
            d = att.dropna(subset=[col])
            if not len(d):
                continue
            r = reliability(d[col], d.correct)
            m = r.count > 0
            ax.plot(r.confidence[m], r.accuracy[m], "o-", c=c, label=label)
            for x, yv, n in zip(r.confidence[m], r.accuracy[m], r.count[m]):
                ax.annotate(str(n), (x, yv), fontsize=7, xytext=(3, -9), textcoords="offset points", c=c)
        ax.set(title=run, xlabel="confidence", ylabel="accuracy", xlim=(0, 1), ylim=(0, 1))
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_selective_and_penalty(df: pd.DataFrame, path: Path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    ts = np.linspace(0, 0.95, 39)
    for run in sorted(df.run.unique()):
        g = df[(df.run == run) & (df.parse_ok == 1)]
        att = g[g.attempted == 1]
        cov, acc = selective_curve(att.confidence, att.correct)
        a1.plot(cov, acc, label=run)
        pen = [penalised_score(g.confidence, g.correct, t, abstained=g.abstained == 1) for t in ts]
        a2.plot(ts, pen, label=run)
    a1.set(title="Selective accuracy (attempted, by verbal confidence)", xlabel="coverage", ylabel="accuracy")
    a2.axhline(0, c="grey", lw=1)
    a2.set(title="Kalai penalised score: wrong costs t/(1-t)", xlabel="threshold t", ylabel="score / question")
    a1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def disagreements(df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """Largest gaps between verbal confidence and sample consistency."""
    d = df[(df.attempted == 1)].dropna(subset=["consistency"]).copy()
    d["gap"] = d.confidence - d.consistency
    cols = ["run", "tier", "id", "answer", "confidence", "consistency", "correct"]
    return pd.concat([d.nlargest(n, "gap")[cols], d.nsmallest(n, "gap")[cols]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, default=ROOT / "outputs" / "epic2" / "confidence")
    ap.add_argument("--out", type=Path, default=None, help="defaults to <log-dir>/analysis")
    args = ap.parse_args()
    out = args.out or args.log_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    df = load(args.log_dir)
    df.to_csv(out / "samples.csv", index=False)
    summ = summarise(df)
    summ.to_csv(out / "summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(summ.to_string(index=False))
        if df.consistency.notna().any():
            print("\nVerbal vs consistency: largest disagreements (verbal >> consistency, then <<)")
            print(disagreements(df).to_string(index=False))
    plot_reliability(df, out / "reliability.png")
    plot_selective_and_penalty(df, out / "selective_penalty.png")
    print(f"\nwrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}/ (summary.csv, samples.csv, *.png)")


if __name__ == "__main__":
    main()
