"""E4 Part B analysis: does steering along the correctness probe change the reported confidence?
(Mac; inputs results/e4_steer_<config>.parquet from steer.py.)

Per config x kind x attribution ("self" = "your answer", "user" = "my answer"):

  shift        mean confidence at +8 SDs minus at -8 SDs along the probe axis (the graded
               range, in which the model's output stays well formed), for the probe direction
               and for the random direction of the same size; and probe minus random, with
               95% intervals from 1,000 bootstrap resamples of the answers (each answer's
               own +8 / -8 pair kept together)
  separation   share of questions whose correct answer gets more confidence than its wrong
               answer (ties half), unsteered and at -8 / +8 SDs along the probe: whether
               steering changes how well the report tells right from wrong
  collapse     share of replies equal to the most common reply, at each strength: a push that
               turns every reply into one constant ("confidence: 0" for every answer) changes
               the output but carries no information about the answer

Writes results/e4_steer_summary.csv and results/figures/e4_steer.png.

    uv run --extra evals python experiments/epic2/phase2_confidence_gap/analyze_steer.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import analyze_swaps as an
import phase1

CONFIGS = ("qwen3-8b", "qwen3-1.7b")
KINDS = ("recall", "computation", "perception")
N_BOOT = 1000
SEED = 0


def load(config: str) -> pd.DataFrame:
    return pd.read_parquet(phase1.RESULTS / f"e4_steer_{config}.parquet").assign(config=config)


def shift_table(r: pd.DataFrame) -> pd.DataFrame:
    """One row per answer: confidence at +8 and -8 SDs, probe and random, plus unsteered."""
    sd = r[r.unit == "sd"]
    p = sd.pivot_table(index=["qid", "source", "correct"], columns=["direction", "alpha"], values="confidence")
    out = pd.DataFrame({
        "probe": p[("probe", 8.0)] - p[("probe", -8.0)],
        "random": p[("random", 8.0)] - p[("random", -8.0)],
        "base": p[("probe", 0.0)],
        "probe_lo_end": p[("probe", -8.0)], "probe_hi_end": p[("probe", 8.0)],
    }).reset_index()
    return out


def ci(vals) -> tuple[float, float]:
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    return (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) if len(v) else (np.nan, np.nan)


def separation(t: pd.DataFrame, col: str) -> float:
    piv = t[t.source.isin(["correct", "wrong"])].pivot_table(index="qid", columns="source", values=col)
    return an.win_rate(piv["correct"].to_numpy(float), piv["wrong"].to_numpy(float))


def main() -> None:
    rows = []
    frames = []
    for config in CONFIGS:
        r = load(config)
        frames.append(r)
        for (kind, att), g in r.groupby(["category", "attribution"]):
            t = shift_table(g)
            idx = np.random.default_rng(SEED).integers(0, len(t), size=(N_BOOT, len(t)))
            row = {"config": config, "kind": kind, "attribution": att, "n": len(t)}
            for col in ("probe", "random"):
                x = t[col].to_numpy(float)
                row[f"shift_{col}"] = float(np.nanmean(x))
                row[f"shift_{col}_lo"], row[f"shift_{col}_hi"] = ci([np.nanmean(x[i]) for i in idx])
            dd = (t.probe - t.random).to_numpy(float)
            row["probe_minus_random"] = float(np.nanmean(dd))
            row["pmr_lo"], row["pmr_hi"] = ci([np.nanmean(dd[i]) for i in idx])
            for col, name in (("base", "sep_unsteered"), ("probe_lo_end", "sep_minus8"), ("probe_hi_end", "sep_plus8")):
                row[name] = separation(t, col)
            for unit, alpha in (("sd", -8.0), ("sd", 8.0), ("norm", -0.5), ("norm", 0.5)):
                x = g[(g.direction == "probe") & (g.unit == unit) & (g.alpha == alpha)].reply
                row[f"collapse_{unit}{alpha:+g}"] = float(x.value_counts(normalize=True).iloc[0]) if len(x) else np.nan
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(phase1.RESULTS / "e4_steer_summary.csv", index=False)
    with pd.option_context("display.width", 230, "display.max_columns", 40):
        cols = ["config", "kind", "attribution", "shift_probe", "shift_probe_lo", "shift_probe_hi", "shift_random",
                "probe_minus_random", "pmr_lo", "pmr_hi", "sep_unsteered", "sep_minus8", "sep_plus8",
                "collapse_sd+8", "collapse_norm-0.5", "collapse_norm+0.5"]
        print(res[cols].round(2).to_string(index=False))
    figure(pd.concat(frames))


def figure(r: pd.DataFrame) -> None:
    pl = phase1.plots
    import matplotlib.pyplot as plt

    sd = r[r.unit == "sd"]
    fig, axes = plt.subplots(len(CONFIGS), 3, figsize=(10.5, 5.6), sharex=True, sharey=True)
    styles = {("self", "probe"): (pl.ORANGE, "-"), ("self", "random"): (pl.ORANGE, ":"),
              ("user", "probe"): (pl.BLUE, "-"), ("user", "random"): (pl.BLUE, ":")}
    for i, cfg in enumerate(CONFIGS):
        for j, kind in enumerate(KINDS):
            ax = axes[i, j]
            g = sd[(sd.config == cfg) & (sd.category == kind)]
            for (att, direction), (colour, ls) in styles.items():
                h = g[(g.attribution == att) & ((g.direction == direction) | (g.alpha == 0))]
                m = h.groupby("alpha").confidence.mean()
                label = f"{'Your answer' if att == 'self' else 'My answer'}, {direction}"
                ax.plot(m.index, m.values, ls, color=colour, lw=2 if direction == "probe" else 1.4,
                        label=label if i == j == 0 else None)
            ax.set_ylim(-0.02, 1.02)
            if i == 0:
                ax.set_title(kind.capitalize())
            if j == 0:
                ax.set_ylabel(f"{pl.MODEL_NAME[cfg]}\nmean stated confidence")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04), fontsize=9)
    fig.suptitle("Pushing the correctness direction does not move the report the way the probe reads it",
                 y=1.09, fontweight="bold")
    fig.supxlabel("Push along the correctness probe (SDs of the probe axis; − = towards wrong, + = towards right)",
                  fontsize=10, color=pl.INK2)
    pl.caption(fig, "A vector added after the layer the Phase 1B probe reads, at every position; solid: the "
                    "probe's direction, dotted: a random direction of the same size. 50 answers per kind and "
                    "source (own, correct, wrong), greedy decoding.", y=-0.03)
    fig.tight_layout()
    an.save(fig, "e4_steer")


if __name__ == "__main__":
    main()
