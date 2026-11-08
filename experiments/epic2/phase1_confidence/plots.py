"""Step 3 figures, blog quality, from the tables analyze.py writes (run `analyze.py all` first).

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


def caption(fig, text: str, y: float = -0.04) -> None:
    """Small grey note under the figure, wrapped to the figure's width."""
    width = int(fig.get_figwidth() * 15)
    fig.text(0.01, y, textwrap.fill(text, width), fontsize=8.5, color=INK2, ha="left", va="top")


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
                    ax.text(j, i, "–", ha="center", va="center", color=MUTED)
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
             "agreement, E token probability.\n– = fewer than 10 right or 10 wrong answers, or "
             "the method isn't available for the model.")
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


def main() -> None:
    auroc_heatmap()
    regression_slopes()
    hard_easy_lines()
    acc_vs_auroc()
    d_vs_a()
    behaviour_rates()
    kalai_answer_rates()
    forecast_brier()


if __name__ == "__main__":
    main()
