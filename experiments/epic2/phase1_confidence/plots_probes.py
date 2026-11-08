"""Phase 1B figures, from the tables analyze_probes.py writes (and probe_layers.csv).

Same style and palette as plots.py. Colours: "before" (end of prompt) uses D's orange and
"after" (last answer token) A's blue, as in Phase 1 (D = before, A = after); the metadata
probe is grey. The abstention result (probe_abstain.csv) has no figure: too few questions on
one side (Qwen3-8B answers 4 of 75 unknowable questions) for a curve to mean anything. Writes PNGs to results/figures/probe_*.png.

    uv run --extra evals python experiments/epic2/phase1_confidence/plots_probes.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from plots import (
    AQUA,
    BLUE,
    GRID,
    INK,
    INK2,
    KINDS,
    METHOD_COL,
    MUTED,
    NEUTRAL,
    ORANGE,
    RED,
    RES,
    caption,
    plt,
    save,
)

BEFORE_COL, AFTER_COL, META_COL = ORANGE, BLUE, MUTED
CONFIGS = ["qwen3-8b", "qwen3-8b-base", "qwen3-1.7b"]
NAME = {"qwen3-8b": "Qwen3-8B", "qwen3-8b-base": "Qwen3-8B-Base", "qwen3-1.7b": "Qwen3-1.7B"}
POS_NAME = {"q_last": "End of question", "gen": "End of prompt", "ans_last": "Last answer token",
            "ans_mean": "Mean over answer"}
STRICT = "Within-level AUROC (strict: same level and task type / relation)"


def layers() -> None:
    """Validation within-level AUROC by layer, before vs after, Qwen3-8B."""
    d = pd.read_csv(RES / "probe_layers.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, kind in zip(axes, KINDS):
        for pos, col in (("gen", BEFORE_COL), ("ans_last", AFTER_COL)):
            g = d[(d.config == "qwen3-8b") & (d.train_kind == kind) & (d.position == pos)]
            ax.plot(g.layer, g.val_within, color=col, label=POS_NAME[pos] + (
                " (before)" if pos == "gen" else " (after)"))
            best = g.loc[g.val_within.idxmax()]
            ax.scatter([best.layer], [best.val_within], color=col, s=40, zorder=3,
                       edgecolor="white", linewidth=1.5)
        ax.axhline(0.5, color=MUTED, linewidth=1)
        ax.set_title(kind.capitalize())
        ax.set_xlabel("Layer (0 = embeddings, 36 = last)")
        ax.set_ylim(0.4, 1.0)
    axes[0].set_ylabel("Within-level AUROC (validation)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=2, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Where the signal sits: Qwen3-8B, probe at every layer", x=0.01, ha="left",
                 fontweight="bold")
    caption(fig, "Logistic probe per layer, trained on 80% of the training questions and scored "
            "on the other 20%; dots mark the chosen layer. Layer 0 is the token embedding alone "
            "(the floor).")
    fig.tight_layout()
    save(fig, "probe_layers")


def before_after() -> None:
    """The key figure: metadata vs before vs after, per model and kind (training set, OOF)."""
    a = pd.read_csv(RES / "probe_auroc.csv")
    a = a[a.strata == "strict"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    y = np.arange(len(CONFIGS))[::-1]
    for ax, kind in zip(axes, KINDS):
        k = a[a.kind == kind].set_index(["config", "position"])
        for yi, cfg in zip(y, CONFIGS):
            pts = {p: k.loc[(cfg, p)] for p in ("meta", "gen", "ans_last")}
            ax.hlines(yi, pts["meta"].auroc_within, pts["ans_last"].auroc_within, color=GRID,
                      linewidth=3)
            for p, col, lab in (("meta", META_COL, "Question details only"),
                                ("gen", BEFORE_COL, "Hidden state before answering"),
                                ("ans_last", AFTER_COL, "Hidden state after answering")):
                r = pts[p]
                ax.errorbar(r.auroc_within, yi, xerr=[[r.auroc_within - r.lo], [r.hi - r.auroc_within]],
                            fmt="o", color=col, ms=7, elinewidth=1.2, capsize=0, zorder=3,
                            label=lab if yi == y[0] else None)
        ax.axvline(0.5, color=MUTED, linewidth=1)
        ax.set_yticks(y, [NAME[c] for c in CONFIGS])
        ax.set_xlim(0.4, 1.0)
        ax.set_title(kind.capitalize())
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Within-level AUROC")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=3, bbox_to_anchor=(0.99, 1.02))
    fig.suptitle("Recall is settled before the answer; computation only after", x=0.01,
                 y=1.04, ha="left", fontweight="bold")
    caption(fig, "Out-of-fold probe scores on ~2,000 training questions per kind, compared "
            "within the same difficulty level and task type (perception) or relation (recall); "
            "95% bootstrap intervals. Question details: popularity, operand sizes, word length "
            "and frequency.")
    fig.tight_layout()
    save(fig, "probe_before_after")


def share_before() -> None:
    """Share of the hidden-state signal (beyond question details) present before answering."""
    a = pd.read_csv(RES / "probe_auroc.csv")
    a = a[a.strata == "strict"].set_index(["config", "kind", "position"]).auroc_within
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    w = 0.26
    x = np.arange(len(KINDS))
    cols = {"qwen3-8b": BLUE, "qwen3-8b-base": AQUA, "qwen3-1.7b": "#86b6ef"}
    for i, cfg in enumerate(CONFIGS):
        share = [(a[(cfg, k, "gen")] - a[(cfg, k, "meta")]) / (a[(cfg, k, "ans_last")] - a[(cfg, k, "meta")])
                 for k in KINDS]
        bars = ax.bar(x + (i - 1) * w, share, w * 0.92, color=cols[cfg], label=NAME[cfg])
        for b, v in zip(bars, share):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center", fontsize=8.5,
                    color=INK2)
    ax.set_xticks(x, [k.capitalize() for k in KINDS])
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.5, 1], ["0", "50%", "100%"])
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncols=3)
    ax.set_title("How much of what the model will know is there before it answers?", loc="left")
    caption(fig, "(before − question details) / (after − question details), strict within-level "
            "AUROC on the training set. 100% = the end-of-prompt state already holds everything "
            "the last answer token adds beyond question details. Point estimates.")
    fig.tight_layout()
    save(fig, "probe_share_before")


def vs_words() -> None:
    """Test set, Qwen3-8B: probe before / after vs A, D, B, E."""
    v = pd.read_csv(RES / "probe_vs_methods.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    rows = [("A", "Stated, after (A)"), ("D", "Stated, before (D)"), ("B", "Sample agreement (B)"),
            ("E", "Token probability (E)"), ("gen", "Probe, before answering"),
            ("ans_last", "Probe, after answering")]
    y = np.arange(len(rows))[::-1]
    for ax, kind in zip(axes, KINDS):
        r = v[(v.config == "qwen3-8b") & (v.kind == kind)].iloc[0]
        vals = [r[f"{m}_level"] for m, _ in rows]
        cols = [METHOD_COL["A"], METHOD_COL["D"], METHOD_COL["B"], METHOD_COL["E"], INK, INK]
        ax.hlines(y, 0.5, vals, color=GRID, linewidth=2)
        for yi, val, col, (m, _) in zip(y, vals, cols, rows):
            face = "white" if m == "gen" else col
            ax.scatter(val, yi, s=60, color=face, edgecolor=col, linewidth=1.8, zorder=3)
            ax.text(val + 0.015, yi, f"{val:.2f}", va="center", fontsize=8.5, color=INK2)
        ax.axvline(0.5, color=MUTED, linewidth=1)
        ax.set_xlim(0.4, 1.0)
        ax.set_yticks(y, [lab for _, lab in rows])
        ax.set_title(f"{kind.capitalize()} (n = {int(r.n)}, {int(r.n_right)} right)")
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Within-level AUROC")
    fig.suptitle("Qwen3-8B: the hidden state knows more than the words", x=0.01, ha="left",
                 fontweight="bold")
    caption(fig, "Phase 1 test questions the plain answer answered. Probe, B and E score the plain "
            "answer; A and D score their own answers. Probe vs stated after (A), before "
            "answering: perception +0.22 [0.10, 0.33], computation +0.19 [0.05, 0.33], recall "
            "+0.12 [−0.03, 0.26].")
    fig.tight_layout()
    save(fig, "probe_vs_words")


def base_vs_instruct() -> None:
    """Per position: base vs instruct probe (strict, training set), with Phase 1's outputs."""
    a = pd.read_csv(RES / "probe_auroc.csv")
    a = a[a.strata == "strict"]
    pos = ["q_last", "gen", "ans_last", "ans_mean"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    x = np.arange(len(pos))
    for ax, kind in zip(axes, KINDS):
        for cfg, col in (("qwen3-8b", BLUE), ("qwen3-8b-base", AQUA)):
            k = a[(a.config == cfg) & (a.kind == kind)].set_index("position").reindex(pos)
            ax.fill_between(x, k.lo, k.hi, color=col, alpha=0.15, linewidth=0)
            ax.plot(x, k.auroc_within, color=col, marker="o", ms=6, label=NAME[cfg])
        ax.set_xticks(x, ["End of\nquestion", "End of\nprompt", "Last answer\ntoken", "Mean over\nanswer"],
                      fontsize=8.5)
        ax.axvline(1.5, color=GRID, linewidth=1, linestyle=(0, (3, 3)))
        ax.text(0.5, 0.97, "before", ha="center", fontsize=8.5, color=INK2, transform=ax.get_xaxis_transform())
        ax.text(2.5, 0.97, "after", ha="center", fontsize=8.5, color=INK2, transform=ax.get_xaxis_transform())
        ax.set_ylim(0.5, 1.0)
        ax.set_title(kind.capitalize())
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Within-level AUROC (strict)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=2, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Post-training barely changes what the hidden states know", x=0.01, ha="left",
                 fontweight="bold")
    caption(fig, "Same training questions, each model against its own right/wrong answers; "
            "bands are 95% bootstrap intervals. In Phase 1 the two differed sharply in what they "
            "output (the base model's token probabilities were the best calibrated signal; the "
            "instruct model's stated confidence among the worst).")
    fig.tight_layout()
    save(fig, "probe_base_vs_instruct")


def transfer() -> None:
    t = pd.read_csv(RES / "probe_transfer.csv")
    t = t[(t.config == "qwen3-8b") & (t.train_kind != "pooled")]
    cmap = LinearSegmentedColormap.from_list("div", [RED, NEUTRAL, BLUE])
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    for ax, pos in zip(axes, ("gen", "ans_last")):
        g = t[t.position == pos].pivot(index="train_kind", columns="test_kind",
                                       values="auroc_within").reindex(index=KINDS, columns=KINDS)
        ax.imshow(g.to_numpy(), cmap=cmap, vmin=0.2, vmax=1.0)
        for i in range(3):
            for j in range(3):
                v = g.iloc[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10,
                        color="white" if abs(v - 0.6) > 0.25 else INK)
        ax.set_xticks(range(3), ["Recall", "Compu-\ntation", "Percep-\ntion"])
        ax.set_yticks(range(3), [k.capitalize() for k in KINDS])
        ax.set_xlabel("Tested on")
        ax.set_ylabel("Trained on")
        ax.set_title(POS_NAME[pos] + (" (before)" if pos == "gen" else " (after)"))
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Probes transfer poorly between kinds (Qwen3-8B)", x=0.01, ha="left",
                 fontweight="bold")
    caption(fig, "Within-level AUROC on the test questions of a probe trained on one kind and "
            "scored on another. 0.5 = no information; below 0.5 = the direction points the "
            "wrong way.")
    fig.tight_layout()
    save(fig, "probe_transfer")


def main() -> None:
    layers()
    before_after()
    share_before()
    vs_words()
    base_vs_instruct()
    transfer()


if __name__ == "__main__":
    main()
