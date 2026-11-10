"""Step 3 and blog-post figures, from the tables analyze.py writes (run `analyze.py all` first).

Writes PNGs to results/figures/. Colours follow the data-viz reference palette, validated with
its script (2026-11-08): methods A, D, B, E use slots 1-4 in order (adjacent pairs pass for
lines); the three answerable kinds use slots 1-3 (all pairs pass, so safe in scatter plots).
Aqua and yellow are below 3:1 contrast on the surface, so their lines are also labelled directly.

    uv run --extra evals python experiments/epic2/phase1_confidence/plots.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = RES / "figures"

SURFACE, INK, INK2, MUTED, GRID = "#ffffff", "#0b0b0b", "#52514e", "#8a8984", "#e8e7e3"
BLUE, ORANGE, AQUA, YELLOW, RED, NEUTRAL = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#f0efec")
METHOD_COL = {"A": BLUE, "D": ORANGE, "B": AQUA, "E": YELLOW}
METHOD_NAME = {"A": "Stated, after (A)", "D": "Stated, before (D)",
               "B": "Sample agreement (B)", "E": "Token probability (E)", "C": "Threshold (C)"}
KIND_COL = {"recall": BLUE, "computation": ORANGE, "perception": AQUA}
KINDS = ("recall", "computation", "perception")
MODELS = ["qwen3-1.7b", "qwen3-8b", "qwen3-8b-think", "qwen3-14b", "qwen3-8b-base",
          "gpt-4o", "gpt-5-nano", "grok-4-1-fast", "claude-haiku"]
MODEL_NAME = {"qwen3-1.7b": "Qwen3-1.7B", "qwen3-8b": "Qwen3-8B", "qwen3-8b-think":
              "Qwen3-8B (thinking)", "qwen3-14b": "Qwen3-14B", "qwen3-8b-base": "Qwen3-8B-Base",
              "gpt-4o": "GPT-4o", "gpt-5-nano": "GPT-5 nano", "grok-4-1-fast": "Grok 4.1 Fast",
              "claude-haiku": "Claude Haiku 5.5"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
    "xtick.major.size": 0, "ytick.major.size": 0,
})


def caption(fig, text: str, y: float = -0.04, keep_lines: bool = False) -> None:
    """Small grey note under the figure, wrapped to the figure's width.

    keep_lines: wrap each "\n"-separated line on its own, so line breaks survive.
    """
    width = int(fig.get_figwidth() * 15)
    lines = text.split("\n") if keep_lines else [text]
    body = "\n".join(textwrap.fill(line, width) for line in lines)
    fig.text(0.01, y, body, fontsize=8.5, color=INK2, ha="left", va="top")


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / f"{name}.png")


def _models(present) -> list[str]:
    return [m for m in MODELS if m in set(present)]


def auroc_heatmap() -> None:
    """Within-level AUROC, model x (kind, method). Diverging around 0.5 (no information)."""
    d = pd.read_csv(RES / "discrimination.csv")
    d = d[d.kind.isin(KINDS)]
    methods = ["A", "D", "B", "E"]
    models = _models(d.config)
    cmap = LinearSegmentedColormap.from_list("div", [RED, NEUTRAL, BLUE])
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4), sharey=True)
    for ax, kind in zip(axes, KINDS):
        k = d[d.kind == kind].set_index(["config", "method"])
        grid = np.full((len(models), len(methods)), np.nan)
        for i, m in enumerate(models):
            for j, me in enumerate(methods):
                if (m, me) in k.index and k.loc[(m, me), "reportable_within"]:
                    grid[i, j] = k.loc[(m, me), "auroc_within"]
        ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0.2, vmax=1.0, aspect="auto")
        for i in range(len(models)):
            for j in range(len(methods)):
                v = grid[i, j]
                if np.isnan(v):
                    mark = "–" if (models[i], methods[j]) in k.index else "n/a"
                    ax.text(j, i, mark, ha="center", va="center", fontsize=8, color=MUTED)
                else:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                            color="white" if abs(v - 0.6) > 0.25 else INK)
        ax.set_xticks(range(len(methods)), methods)
        ax.set_yticks(range(len(models)), [MODEL_NAME[m] for m in models])
        ax.set_title(kind.capitalize())
        ax.grid(False)
        ax.set_xticks(np.arange(-0.5, len(methods)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(models)), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Does confidence pick out the wrong answers among equally hard questions?",
                 x=0.01, ha="left", fontweight="bold")
    caption(fig, "Within-level AUROC: 0.5 = no information, 1 = every right answer "
             "more confident than every wrong one. A stated after, D stated before, B sample "
             "agreement, E token probability.\nn/a = method not available for the model "
             "(no stated confidence from the base model; no token probabilities from GPT-5 nano or Claude Haiku).\n"
             "– = method ran, but fewer than 10 right or 10 wrong answers within levels "
             "(the model was nearly always right).", keep_lines=True)
    fig.tight_layout()
    save(fig, "auroc_within")


def regression_slopes() -> None:
    r = pd.read_csv(RES / "regression.csv")
    r = r[r.term.isin(KINDS)]
    methods = ["A", "D", "B", "E", "C"]
    fig, ax = plt.subplots(figsize=(8, 4))
    off = {"recall": -0.22, "computation": 0.0, "perception": 0.22}
    for kind in KINDS:
        k = r[r.term == kind].set_index("method").reindex(methods)
        x = np.arange(len(methods)) + off[kind]
        ax.errorbar(x, k.slope, yerr=[k.slope - k.lo, k.hi - k.slope], fmt="o", ms=7,
                    color=KIND_COL[kind], ecolor=KIND_COL[kind], elinewidth=2, capsize=0,
                    label=kind.capitalize(), markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks(range(len(methods)), [METHOD_NAME[m].replace(" (", "\n(") for m in methods])
    ax.set_ylabel("Confidence slope (log-odds)")
    ax.set_ylim(-3, 11)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncols=3)
    ax.set_title("With difficulty held fixed, stated confidence is blind on perception",
                 loc="left")
    caption(fig, "Slope = log-odds of being right for the model's most vs least confident "
            "answer within the same model × category × difficulty cell, pooled over models; "
            "0 = no signal. 95% bootstrap intervals over questions.")
    fig.tight_layout()
    save(fig, "regression_slopes")


def hard_easy_lines() -> None:
    """Accuracy and confidence across the five levels, for four contrasting models."""
    lv = pd.read_csv(RES / "hard_easy.csv")
    models = ["gpt-4o", "qwen3-8b", "qwen3-8b-base", "gpt-5-nano"]
    fig, axes = plt.subplots(len(models), 3, figsize=(10, 2.3 * len(models)), sharex=True,
                             sharey=True)
    for r, model in enumerate(models):
        for c, kind in enumerate(KINDS):
            ax = axes[r, c]
            g = lv[(lv.config == model) & (lv.kind == kind)]
            p_acc = g[g.method == "B"].set_index("level").acc
            ax.plot(p_acc.index, p_acc.values, color=INK, marker="o", ms=5, label="Accuracy")
            for me in ("A", "B", "E"):
                h = g[g.method == me].set_index("level")
                if len(h):
                    ax.plot(h.index, h.conf, color=METHOD_COL[me], marker="o", ms=5,
                            label=METHOD_NAME[me])
            ax.set_ylim(-0.03, 1.05)
            ax.set_xticks([1, 2, 3, 4, 5])
            if r == 0:
                ax.set_title(kind.capitalize())
            if c == 0:
                ax.set_ylabel(MODEL_NAME[model], fontweight="bold", color=INK)
            if r == len(models) - 1:
                ax.set_xlabel("Difficulty level")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=4, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("The hard–easy effect: stated confidence stays near 1 as accuracy falls",
                 x=0.01, y=1.07, ha="left", fontweight="bold")
    caption(fig, "Mean confidence at each difficulty level against the accuracy of the plain "
            "answer (black), on answered questions. The base model has no stated confidence; "
            "GPT-5 nano returns no token probabilities.", y=-0.005)
    fig.tight_layout()
    save(fig, "hard_easy")


def acc_vs_auroc() -> None:
    c = pd.read_csv(RES / "cells.csv").dropna(subset=["auroc"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, me in zip(axes, ("A", "B")):
        for kind in KINDS:
            k = c[(c.method == me) & (c.kind == kind)]
            ax.scatter(k.acc, k.auroc, s=40, color=KIND_COL[kind], edgecolor=SURFACE,
                       linewidth=1.5, label=f"{kind.capitalize()} ({len(k)})")
        ax.axhline(0.5, color=MUTED, linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0.1, 1.02)
        ax.set_xlabel("Accuracy in the cell")
        ax.set_title(METHOD_NAME[me], loc="left")
        ax.legend(loc="lower right")
    axes[0].set_ylabel("AUROC")
    fig.suptitle("At the same accuracy, stated confidence works on recall but not perception;"
                 " sample agreement works on both", x=0.01, ha="left", fontweight="bold")
    caption(fig, "Each point is one model × category × difficulty level with at least 5 right "
            "and 5 wrong answers (count in the legend). Many stated-confidence points sit at "
            "exactly 0.5: the model gave every answer in the cell the same confidence.")
    fig.tight_layout()
    save(fig, "acc_vs_auroc")


def d_vs_a() -> None:
    d = pd.read_csv(RES / "d_vs_a.csv")
    d = d[d.reportable]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharex=True)
    for ax, kind in zip(axes, KINDS):
        k = d[d.kind == kind].set_index("config").reindex(_models(d[d.kind == kind].config))
        y = np.arange(len(k))[::-1]
        ax.hlines(y, k.auroc_within_A, k.auroc_within_D, color=GRID, linewidth=3)
        ax.scatter(k.auroc_within_A, y, color=METHOD_COL["A"], s=50, zorder=3, label="After (A)")
        ax.scatter(k.auroc_within_D, y, color=METHOD_COL["D"], s=50, zorder=3, label="Before (D)")
        ax.set_yticks(y, [MODEL_NAME[m] for m in k.index])
        ax.axvline(0.5, color=MUTED, linewidth=1)
        ax.set_xlim(0.3, 0.95)
        ax.set_title(kind.capitalize())
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Within-level AUROC")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=2, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Confidence before vs after answering: no general winner", x=0.01, ha="left",
                 fontweight="bold")
    caption(fig, "Within-level AUROC on the same questions, both calls answered; Claude Haiku "
            "replies that worked the problem out before the JSON are left out. Models with "
            "fewer than 10 right or 10 wrong answers are omitted.")
    fig.tight_layout()
    save(fig, "d_vs_a")


def behaviour_rates() -> None:
    r = pd.read_csv(RES / "behaviour_rates.csv")
    r = r[r.method == "P"]
    titles = {"false_premise": "Corrects a false premise", "ambiguous": "Asks which work is meant",
              "unknowable": "Says it can't be known"}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    models = _models(r.config)
    y = np.arange(len(models))[::-1]
    for ax, kind in zip(axes, titles):
        k = r[r.kind == kind].set_index("config").reindex(models)
        ax.barh(y, k.right, height=0.62, color=BLUE)
        for yi, v in zip(y, k.right):
            ax.text(v + 0.02, yi, f"{v:.0%}", va="center", fontsize=8.5, color=INK2)
        ax.set_xlim(0, 1.15)
        ax.set_yticks(y, [MODEL_NAME[m] for m in models])
        ax.set_title(titles[kind])
        ax.grid(axis="y", visible=False)
        ax.set_xticks([0, 0.5, 1], ["0", "50%", "100%"])
    fig.suptitle("Questions without a proper answer: right behaviour on the plain answer",
                 x=0.01, ha="left", fontweight="bold")
    caption(fig, "40 false-premise, 40 ambiguous-title and 75 unknowable questions; "
             "labels by the validated judge.")
    fig.tight_layout()
    save(fig, "behaviour")


def kalai_answer_rates() -> None:
    k = pd.read_csv(RES / "kalai.csv")
    models = _models(k.config)
    ts = [0.0, 0.5, 0.75, 0.9]
    cmap = LinearSegmentedColormap.from_list("seq", ["#e3eefc", "#86b6ef", "#2a78d6", "#0d366b"])
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), sharey=True)
    for ax, kind in zip(axes, KINDS):
        g = k[k.kind == kind].pivot(index="config", columns="t", values="answer_rate")
        g = g.reindex(index=models, columns=ts)
        ax.imshow(g.to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(len(models)):
            for j in range(len(ts)):
                v = g.iloc[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if v > 0.6 else INK)
        ax.set_xticks(range(len(ts)), ["0", "0.5", "0.75", "0.9"])
        ax.set_yticks(range(len(models)), [MODEL_NAME[m] for m in models])
        ax.set_xlabel("Threshold t in the prompt")
        ax.set_title(kind.capitalize())
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Share of questions answered as the stated penalty rises: models react to a "
                 "penalty, not to its size", x=0.01, ha="left", fontweight="bold")
    caption(fig, "Prompt: answer only if more than t confident; a wrong answer costs "
             "t/(1−t) points, a right one earns 1, abstaining 0 (Kalai et al., 2025).")
    fig.tight_layout()
    save(fig, "kalai_answer_rates")


def forecast_brier() -> None:
    f = pd.read_csv(RES / "forecasts.csv")
    crowd = f.loc[f.config == "crowd", "brier"].item()
    base = f.loc[f.config == "base rate", "brier"].item()
    m = f[f.method == "F"].set_index("config")
    models = sorted(_models(m.index), key=lambda c: m.loc[c, "brier"])
    y = np.arange(len(models))[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.barh(y, m.loc[models, "brier"], height=0.62, color=BLUE)
    for yi, c in zip(y, models):
        ax.text(m.loc[c, "brier"] + 0.003, yi, f"{m.loc[c, 'brier']:.3f}", va="center",
                fontsize=8.5, color=INK2)
    for x, lab, ha in ((crowd, f"Crowd {crowd:.3f} ", "right"),
                       (base, f" Base rate {base:.3f}", "left")):
        ax.axvline(x, color=INK, linewidth=1.2)
        ax.text(x, len(models) - 0.35, lab, ha=ha, va="bottom", fontsize=8.5, color=INK)
    ax.set_yticks(y, [MODEL_NAME[c] for c in models])
    ax.set_xlim(0, 0.34)
    ax.set_xlabel("Brier score (lower is better)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Forecasting: no model beats always guessing the base rate", loc="left",
                 pad=22)
    caption(fig, "75 market questions resolved after every model's cutoff; the model's stated "
            "probability (F). Base rate = always predicting the share that resolved yes.")
    fig.tight_layout()
    save(fig, "forecasts")


# ---------------------------------------------------------------- figures for the post


def _rows() -> pd.DataFrame:
    return pd.read_parquet(RES / "rows.parquet")


def accuracy_levels() -> None:
    """Plain-answer accuracy per model, category and difficulty level: the difficulty knob."""
    d = _rows()
    d = d[(d.method == "P") & d.kind.isin(KINDS)]
    acc = d.groupby(["config", "kind", "level"]).correct.mean()
    models = _models(d.config)
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f8fe", "#86b6ef", "#2a78d6", "#0d366b"])
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.9), sharey=True)
    for ax, kind in zip(axes, KINDS):
        g = np.array([[acc.get((m, kind, lv), np.nan) for lv in range(1, 6)] for m in models])
        ax.imshow(g, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(len(models)):
            for j in range(5):
                ax.text(j, i, f"{g[i, j]:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if g[i, j] > 0.55 else INK)
        ax.set_xticks(range(5), range(1, 6))
        ax.set_yticks(range(len(models)), [MODEL_NAME[m] for m in models])
        ax.set_xlabel("Difficulty level")
        ax.set_title(kind.capitalize())
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle("The difficulty knob works: accuracy of the plain answer by level", x=0.01,
                 ha="left", fontweight="bold")
    caption(fig, "150 questions per category, 30 per level. Recall: entity popularity "
            "(Wikipedia page views). Computation: multiplication from 2 × 2 to 4 × 4 digits. "
            "Perception: count a letter, name the nth letter, or spell a word backwards, on "
            "words from 5 to 15 letters.")
    fig.tight_layout()
    save(fig, "accuracy_levels")


def judge_check() -> None:
    """How often the grading agrees with hand labels."""
    rows = [("Old judge, before the rework", 77, 120, MUTED),
            ("Rules + judge, gold set", 114, 120, BLUE),
            ("Same, judge thinking on", 112, 120, BLUE),
            ("Blind held-out, Qwen answers", 39, 40, AQUA),
            ("Blind held-out, API models", 43, 48, AQUA)]
    fig, ax = plt.subplots(figsize=(7.5, 2.9))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, k, n, col) in zip(y, rows):
        ax.barh(yi, k / n, height=0.6, color=col)
        ax.text(k / n + 0.01, yi, f"{k}/{n}  ({k / n:.0%})", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, 0.5, 1], ["0", "50%", "100%"])
    ax.grid(axis="y", visible=False)
    ax.set_title("Grading agreement with hand labels", loc="left")
    caption(fig, "Gold set: 120 responses drafted by Claude and reviewed by hand. Held-out "
            "sets were labelled blind, with the judge's label hidden; every API-model "
            "disagreement had the judge stricter than the hand label.", y=-0.06)
    fig.tight_layout()
    save(fig, "judge_check")


def within_level_explainer() -> None:
    """Synthetic: confidence that only tracks the level scores well on plain AUROC."""
    from synapses.calibration import auroc, stratified_auroc

    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, item_signal, title in ((axes[0], 0.0, "Knows which levels are hard"),
                                   (axes[1], 0.25, "Also knows which answers are wrong")):
        ps, ys, ls = [], [], []
        for lv, (acc, conf) in enumerate([(0.9, 0.85), (0.5, 0.6), (0.1, 0.35)], 1):
            y = (rng.random(40) < acc).astype(float)
            p = conf + item_signal * (y - acc) + rng.normal(0, 0.05, 40)
            jitter = lv + rng.uniform(-0.18, 0.18, 40)
            ax.scatter(jitter[y == 1], p[y == 1], s=22, color=BLUE, label="Right" if lv == 1
                       else None, edgecolor=SURFACE, linewidth=0.8)
            ax.scatter(jitter[y == 0], p[y == 0], s=22, color=ORANGE, label="Wrong" if lv == 1
                       else None, edgecolor=SURFACE, linewidth=0.8)
            ps += list(p)
            ys += list(y)
            ls += [lv] * 40
        a, w = auroc(ps, ys), stratified_auroc(ps, ys, ls)
        ax.set_title(f"{title}\nplain AUROC {a:.2f}, within-level {w:.2f}", loc="left")
        ax.set_xticks([1, 2, 3], ["Easy", "Medium", "Hard"])
        ax.set_ylim(0, 1.1)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Stated confidence")
    axes[0].legend(loc="lower left")
    fig.suptitle("Two kinds of self-knowledge", x=0.01, ha="left", fontweight="bold")
    caption(fig, "Synthetic data. Left: confidence follows the level but is identical for "
            "right and wrong answers within a level. Plain AUROC still rewards it; "
            "within-level AUROC (compare only within a level) does not.")
    fig.tight_layout()
    save(fig, "within_level_explainer")


def reliability_curves() -> None:
    """Calibration curves: an instruct model's stated confidence vs its samples vs the base
    model's token probability."""
    from synapses.calibration import reliability

    d = _rows()
    d = d[d.kind.isin(KINDS) & (d.abstained == 0) & d.correct.notna() & d.conf.notna()]
    series = [("qwen3-8b", "A", "Qwen3-8B, stated (A)", BLUE),
              ("qwen3-8b", "B", "Qwen3-8B, sample agreement (B)", AQUA),
              ("qwen3-8b-base", "E", "Qwen3-8B-Base, token probability (E)", YELLOW)]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7), sharey=True)
    for ax, kind in zip(axes, KINDS):
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1)
        for config, me, lab, col in series:
            g = d[(d.config == config) & (d.method == me) & (d.kind == kind)]
            p = np.exp(g.conf) if me == "E" else g.conf
            r = reliability(p.to_numpy(), g.correct.to_numpy(), n_bins=10)
            m = r.count >= 5
            ax.plot(r.confidence[m], r.accuracy[m], color=col, marker="o", ms=5, label=lab)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_aspect("equal")
        ax.set_title(kind.capitalize())
        ax.set_xlabel("Confidence")
    axes[0].set_ylabel("Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Reliability: the same weights, three ways of reading confidence", x=0.01,
                 y=1.12, ha="left", fontweight="bold")
    caption(fig, "Answered questions, binned by confidence (10 bins, bins with at least 5 "
            "answers); on the diagonal = calibrated. Qwen3-8B's stated confidence is a single "
            "point: nearly every answer is stated at 0.9 or more. The base model's token "
            "probability is the probability of the whole answer string, so a correct long "
            "number still gets a low value (the computation curve sits above the diagonal).")
    fig.tight_layout()
    save(fig, "reliability")


def says_vs_samples() -> None:
    c = pd.read_csv(RES / "disagree_counts.csv")
    models = [m for m in _models(c.config) if m != "gpt-5-nano"]
    sure = c[c.group == "says sure, samples scatter"].set_index("config")
    other = c[c.group == "other"].set_index("config")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    y = np.arange(len(models))[::-1]
    axes[0].barh(y, sure.reindex(models).share, height=0.6, color=BLUE)
    for yi, m in zip(y, models):
        axes[0].text(sure.loc[m, "share"] + 0.01, yi, f"{sure.loc[m, 'share']:.0%}",
                     va="center", fontsize=8.5, color=INK2)
    axes[0].set_xlim(0, 0.6)
    axes[0].set_xticks([0, 0.25, 0.5], ["0", "25%", "50%"])
    axes[0].set_title("Share of answers: says ≥ 0.9 sure,\nbut ≤ half its samples agree",
                      loc="left")
    axes[0].set_yticks(y, [MODEL_NAME[m] for m in models])
    ax = axes[1]
    ax.hlines(y, sure.reindex(models).p_acc, other.reindex(models).p_acc, color=GRID,
              linewidth=3)
    ax.scatter(sure.reindex(models).p_acc, y, color=ORANGE, s=50, zorder=3,
               label="Says sure, samples scatter")
    ax.scatter(other.reindex(models).p_acc, y, color=BLUE, s=50, zorder=3,
               label="All other answers")
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0", "50%", "100%"])
    ax.set_title("How often the answer is right", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=2)
    for a in axes:
        a.grid(axis="y", visible=False)
    fig.suptitle("When the words and the samples disagree, believe the samples", x=0.01,
                 ha="left", fontweight="bold")
    caption(fig, "Answered recall, computation and perception questions. GPT-5 nano is left "
            "out: it almost never says it is sure. The reverse case (says unsure, samples "
            "agree) is at most 1% for every model.")
    fig.tight_layout()
    save(fig, "says_vs_samples")


def haiku_working() -> None:
    d = _rows()
    d = d[(d.config == "claude-haiku") & d.method.str.match(r"^(P|C0)")
          & d.kind.isin(["computation", "perception"])].copy()
    d["working"] = d.answer.fillna("").str.len() > 25
    order = ["P", "C0.0", "C0.5", "C0.75", "C0.9"]
    labels = ["Plain", "t = 0", "t = 0.5", "t = 0.75", "t = 0.9"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharex=True)
    for kind, col in (("computation", ORANGE), ("perception", AQUA)):
        g = d[d.kind == kind].groupby("method").agg(w=("working", "mean"), a=("correct", "mean"))
        g = g.reindex(order)
        axes[0].plot(range(5), g.w, color=col, marker="o", ms=6, label=kind.capitalize())
        axes[1].plot(range(5), g.a, color=col, marker="o", ms=6, label=kind.capitalize())
        axes[0].text(4.1, g.w.iloc[-1], kind.capitalize(), va="center", fontsize=9, color=INK2)
        axes[1].text(4.1, g.a.iloc[-1], kind.capitalize(), va="center", fontsize=9, color=INK2)
    axes[0].set_title("Answers that show worked steps", loc="left")
    axes[1].set_title("Accuracy", loc="left")
    for ax, lo in ((axes[0], 0), (axes[1], 0.6)):
        ax.set_xticks(range(5), labels)
        ax.set_xlim(-0.3, 5.0)
        ax.set_ylim(lo, 1.02)
        ax.grid(axis="x", visible=False)
    axes[0].set_yticks([0, 0.5, 1], ["0", "50%", "100%"])
    fig.suptitle("Claude Haiku 5.5 under a stated penalty: it shows its work instead of "
                 "abstaining", x=0.01, ha="left", fontweight="bold")
    caption(fig, "Same questions, plain prompt vs the threshold prompt at each t. 'Worked "
            "steps' = a reply longer than 25 characters, e.g. '777 × 933 = 777×900 + 777×33 "
            "= … = 724,941'. Haiku answered every one of these questions at every t.")
    fig.tight_layout()
    save(fig, "haiku_working")


def unknowable_confidence() -> None:
    c = pd.read_csv(RES / "behaviour_conf.csv")
    c = c[c.kind == "unknowable"]
    models = _models(c.config)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    y = np.arange(len(models))[::-1]
    for ax, me, title in ((axes[0], "A", "Asked after the response (A)"),
                          (axes[1], "D", "Asked before answering (D)")):
        k = c[c.method == me].pivot_table(index="config", columns="label", values="conf")
        k = k.reindex(models)
        ab, co = k.get("ABSTAINS"), k.get("COMMITS")
        ax.hlines(y, ab, co, color=GRID, linewidth=3)
        ax.scatter(ab, y, color=BLUE, s=50, zorder=3, label="When it says it can't be known")
        ax.scatter(co, y, color=ORANGE, s=50, zorder=3, label="When it commits to an answer")
        ax.set_xlim(-0.03, 1.05)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_yticks(y, [MODEL_NAME[m] for m in models])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=2, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Unknowable questions: what does 'confidence' mean?", x=0.01, y=1.09,
                 ha="left", fontweight="bold")
    caption(fig, "Mean stated confidence on the 75 unknowable questions. After responding, "
            "GPT-4o, Grok and Qwen3-8B/14B rate the response itself, so a correct 'nobody "
            "knows' gets 0.85 to 1.0; Claude Haiku, GPT-5 nano and Qwen3-1.7B give it low "
            "confidence. Before answering, every model's confidence falls when it is about to "
            "abstain. Claude Haiku never committed.")
    fig.tight_layout()
    save(fig, "unknowable_confidence")


def main() -> None:
    auroc_heatmap()
    regression_slopes()
    hard_easy_lines()
    acc_vs_auroc()
    d_vs_a()
    behaviour_rates()
    kalai_answer_rates()
    forecast_brier()
    accuracy_levels()
    judge_check()
    within_level_explainer()
    reliability_curves()
    says_vs_samples()
    haiku_working()
    unknowable_confidence()


if __name__ == "__main__":
    main()
