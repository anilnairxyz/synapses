"""E5 analysis (Mac): does training the "your answer" report on correctness make it read what
the model knows? Inputs: results/e5_scores_<config>.parquet (lora_confidence.py) and the probe
scores on the same test answers (results/extract_<config>.parquet, Phase 1B's probe refitted).

Per config, variant (base = untrained; per_answer; lin; shuffled; loko_<kind>) and kind:

  own answers, "your answer" (the E3 question, now after training):
    within   within-stratum AUROC of the reported confidence's expected value (EV) for the
             answer being right, 95% bootstrap interval; minus_probe: paired difference to the
             probe after the answer on the same answers, with interval
    pooled   AUROC across levels within the kind (the coarse signal Lin et al. trained on)
    greedy   within-stratum AUROC of the greedy reply's number (what the model says)
    ece      expected calibration error of EV
  injected answers (E2's correct vs wrong, same question):
    sep_self, sep_user  paired win rate under "your answer" / "my answer" (0.5 = none)
  transfer (loko_<kind> on the held-out kind): share of the in-kind gain (per_answer - base)
           that the model trained without that kind reaches (loko - base)

Writes results/e5_summary.csv and results/figures/e5_lora.png.

    uv run --extra evals python experiments/epic2/phase2_confidence_gap/analyze_lora.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import analyze_swaps as an
import phase1

CONFIGS = ("qwen3-1.7b", "qwen3-8b")
KINDS = ("recall", "computation", "perception")
ORDER = ("base", "shuffled", "lin", "per_answer", "loko")
N_BOOT = 1000
SEED = 0


def load(config: str) -> pd.DataFrame | None:
    f = phase1.RESULTS / f"e5_scores_{config}.parquet"
    if not f.exists():
        return None
    d = pd.read_parquet(f)
    d["stratum"] = d.category + ":" + d.level.astype(str) + ":" + d.family.fillna("")
    ext = pd.read_parquet(phase1.RESULTS / f"extract_{config}.parquet")
    ext = ext[ext.split == "test"][["qid", "source", "probe_ans_last"]].rename(columns={"probe_ans_last": "probe"})
    return d.merge(ext, on=["qid", "source"], how="left")


def summarise(d: pd.DataFrame, config: str) -> list[dict]:
    rows = []
    for variant, v in d.groupby("variant"):
        for kind in KINDS:
            k = v[v.category == kind]
            own = k[(k.source == "own") & (k.attribution == "self")].reset_index(drop=True)
            y, s, g, p, st = (own.correct.to_numpy(float), own.ev.to_numpy(float), own.greedy.to_numpy(float),
                              own.probe.to_numpy(float), own.stratum.to_numpy())
            idx = np.random.default_rng(SEED).integers(0, len(own), size=(N_BOOT, len(own)))
            row = {"config": config, "variant": variant, "kind": kind, "n_own": len(own), "n_right": int(y.sum())}
            row["within"] = an.within_auroc(s, y, st)
            row["within_lo"], row["within_hi"] = an.ci([an.within_auroc(s[i], y[i], st[i]) for i in idx])
            row["probe_within"] = an.within_auroc(p, y, st)
            dd = [an.within_auroc(s[i], y[i], st[i]) - an.within_auroc(p[i], y[i], st[i]) for i in idx]
            row["minus_probe"] = row["within"] - row["probe_within"]
            row["minus_probe_lo"], row["minus_probe_hi"] = an.ci(dd)
            row["pooled"] = phase1.cal.auroc(s, y)
            row["greedy_within"] = an.within_auroc(g, y, st)
            row["ece"] = phase1.cal.ece(s, y)
            row["mean_ev"] = float(s.mean())
            for att in ("self", "user"):
                inj = k[(k.attribution == att) & k.source.isin(["correct", "wrong"])]
                piv = inj.pivot_table(index="qid", columns="source", values="ev")
                c, w = piv["correct"].to_numpy(float), piv["wrong"].to_numpy(float)
                row[f"sep_{att}"], row[f"sep_{att}_lo"], row[f"sep_{att}_hi"] = an.with_ci(an.win_rate, len(c), c, w)
            rows.append(row)
    return rows


def transfer(res: pd.DataFrame) -> pd.DataFrame:
    out = []
    for config, g in res.groupby("config"):
        t = g.set_index(["variant", "kind"])
        for kind in KINDS:
            if ("base", kind) not in t.index or ("per_answer", kind) not in t.index or (f"loko_{kind}", kind) not in t.index:
                continue
            base, full, loko = (t.loc[("base", kind), "within"], t.loc[("per_answer", kind), "within"],
                                t.loc[(f"loko_{kind}", kind), "within"])
            out.append({"config": config, "kind": kind, "base": base, "per_answer": full, "held_out": loko,
                        "share_of_gain": (loko - base) / (full - base) if full != base else np.nan})
    return pd.DataFrame(out)


def figure(res: pd.DataFrame) -> None:
    pl = phase1.plots
    import matplotlib.pyplot as plt

    configs = [c for c in CONFIGS if c in set(res.config)]
    names = {"base": "Untrained", "shuffled": "Shuffled labels", "lin": "Lin-style (level accuracy)",
             "per_answer": "Per-answer correctness", "loko": "Per-answer, kind held out"}
    colours = {"base": pl.MUTED, "shuffled": pl.INK2, "lin": pl.YELLOW, "per_answer": pl.ORANGE, "loko": pl.AQUA}
    fig, axes = plt.subplots(len(configs), 3, figsize=(10.5, 1.2 + 2.6 * len(configs)), sharex=True, squeeze=False)
    for i, cfg in enumerate(configs):
        for j, kind in enumerate(KINDS):
            ax = axes[i, j]
            g = res[(res.config == cfg) & (res.kind == kind)].set_index("variant")
            for y, key in enumerate(ORDER):
                var = f"loko_{kind}" if key == "loko" else key
                if var not in g.index:
                    continue
                r = g.loc[var]
                ax.plot([r.within_lo, r.within_hi], [y, y], color=colours[key], lw=1.5, alpha=0.6)
                ax.plot(r.within, y, "o", color=colours[key], ms=6.5, mec=pl.SURFACE, mew=1)
            if "base" in g.index:
                ax.axvline(g.loc["base", "probe_within"], color=pl.BLUE, lw=1.5, ls="--",
                           label="Hidden-state probe" if i == j == 0 else None)
            ax.axvline(0.5, color=pl.MUTED, lw=0.8)
            ax.set_yticks(range(len(ORDER)), [names[k] for k in ORDER])
            ax.set_xlim(0.3, 1.0)
            if i == 0:
                ax.set_title(kind.capitalize())
            if j == 0:
                ax.set_ylabel(pl.MODEL_NAME[cfg])
                ax.invert_yaxis()
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Training the 'your answer' report on whether each answer was right", y=1.08, fontweight="bold")
    fig.supxlabel("Within-level AUROC of the reported confidence for the model's own answer being right",
                  fontsize=10, color=pl.INK2)
    pl.caption(fig, "LoRA on the training questions; tested on the test questions (150 per kind), 95% bootstrap "
                    "intervals. Dashed: the Phase 1B probe on the same answers. Held out: trained on the other "
                    "two kinds only.", y=-0.03)
    fig.tight_layout()
    an.save(fig, "e5_lora")


def main() -> None:
    rows = []
    for config in CONFIGS:
        d = load(config)
        if d is None:
            print("no results for", config)
            continue
        rows += summarise(d, config)
    res = pd.DataFrame(rows)
    res.to_csv(phase1.RESULTS / "e5_summary.csv", index=False)
    tr = transfer(res)
    tr.to_csv(phase1.RESULTS / "e5_transfer.csv", index=False)
    with pd.option_context("display.width", 230, "display.max_columns", 30, "display.max_rows", 100):
        cols = ["config", "variant", "kind", "n_own", "n_right", "within", "within_lo", "within_hi", "probe_within",
                "minus_probe", "minus_probe_lo", "minus_probe_hi", "pooled", "greedy_within", "ece", "sep_self", "sep_user"]
        print(res[cols].round(2).to_string(index=False))
        print(tr.round(2).to_string(index=False))
    figure(res)


if __name__ == "__main__":
    main()
