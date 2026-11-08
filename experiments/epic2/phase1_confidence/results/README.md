# Step 3 results: Confidence Audit

Generated from the judged full-run logs (8 chat configs plus Qwen3-8B-Base on P/B/E) by `analyze.py all` and `plots.py`. No model calls. Definitions of every method row and column are in the `analyze.py` docstring. Written 2026-11-08; these are the working findings for Step 6 (RESULTS.md) and the "Inside …" post.

## Files

| File | What |
| --- | --- |
| `rows.parquet` | per (config, question, method) table (not committed; `analyze.py check --refresh` rebuilds it) |
| `discrimination.csv` | per (config, answerable kind, method): accuracy, abstention, AUROC and within-level AUROC with bootstrap 95% CIs, Brier, Murphy, ECE |
| `regression.csv` | confidence slope per kind and method, pooled over models with (model × kind × level) fixed effects; differences vs recall |
| `cells.csv` | accuracy and AUROC per (model, kind, level, method) |
| `hard_easy.csv`, `hard_easy_summary.csv` | confidence vs accuracy per level; sensitivity and over/underconfidence at the easiest and hardest level |
| `method_corr.csv` | Spearman correlation between methods per model |
| `d_vs_a.csv` | before (D) vs after (A), paired, within-level AUROC difference with CI; Haiku preamble replies dropped |
| `disagree_counts.csv`, `disagree_examples.csv` | questions where stated confidence (A) and sample agreement (B) disagree |
| `behaviour_rates.csv`, `behaviour_conf.csv` | ill-posed and unknowable: right-behaviour rate per call; stated confidence by label |
| `kalai.csv` | threshold calls on answerable questions: answer rate, accuracy, penalised score per t |
| `forecasts.csv` | forecast Brier, Murphy, AUROC per model, with crowd and base-rate baselines |
| `figures/` | blog-quality PNGs (see below) |

## Phase 1B tables

| File | What |
| --- | --- |
| `probe_layers.csv` | probe.py: validation and test within-level AUROC per (config, train kind, position, layer) |
| `probe_scores.parquet` | probe.py: test-set scores per (config, probe, train kind, position, question); probes: logreg, mlp, shuffled, meta |
| `probe_oof.parquet` | probe.py --oof: out-of-fold scores on the training set at the chosen layers |
| `probe_auroc.csv`, `probe_gaps.csv` | analyze_probes.py: training-set within-level AUROC (level / strict) and paired differences (after − before, before − details, base − instruct) |
| `probe_vs_methods.csv`, `probe_transfer.csv`, `probe_abstain.csv` | analyze_probes.py: test set, probe vs A/D/B/E, cross-kind transfer, abstention |

Findings: [`../RESULTS_1B.md`](../RESULTS_1B.md).

## Findings

1. **Recall vs perception is a category effect, not a difficulty effect.** With difficulty and model held fixed (regression with cell fixed effects), confidence predicts correctness much less on perception than on recall for every method: stated after (A) −6.5 log-odds [−10.6, −4.2], stated before (D) −6.1, sample agreement (B) −3.1, token probability (E) −1.8, threshold (C) −2.1, all intervals excluding 0. Stated confidence on perception carries no item-level signal at all (A slope +0.7 [−1.6, 2.4], D −0.4 [−1.3, 0.7]). Figures: `regression_slopes.png`, `acc_vs_auroc.png`, `auroc_within.png`.
2. **What models do knows more than what they say.** Sample agreement and token probabilities discriminate (within-level AUROC mostly 0.6–0.9) where stated confidence is flat. "Says sure, samples scatter" (A ≥ 0.9, B ≤ 0.5) covers 14–50% of answered questions for most models, and the plain answer in that group is right only 2–28% of the time; the reverse ("says unsure, samples agree") almost never occurs (≤ 1%).
3. **Much of the computation signal is knowing which *levels* are hard.** Plain → within-level AUROC drops sharply on computation (e.g. Qwen3-8B D 0.94 → 0.72, GPT-4o D 0.85 → 0.64, GPT-4o B 0.97 → 0.84) and hardly at all on recall (Claude Haiku A 0.90 → 0.87). After answering a computation question, no model's stated confidence tells its right answers from its wrong ones within a level (A ≈ 0.5).
4. **The hard–easy effect is one-sided and limited to what models say.** Instruction-tuned models state ≈ 1.0 whatever the difficulty (sensitivity of A to accuracy ≈ 0 on computation and perception for GPT-4o, Grok, Qwen3-8B/14B), so they are calibrated on easy items only because they are right there, and overconfident by up to +1.0 on the hardest (GPT-4o and Grok on the hardest products, 4 digits × 4 digits: 0% accuracy, confidence 1.0). Sample agreement and token probabilities track difficulty (sensitivity mostly 0.5–1.0). GPT-5 nano is the exception: mildly underconfident (−0.1 to −0.2) on computation and perception. Figure: `hard_easy.png`.
5. **The base model is the best-calibrated model in the set.** Qwen3-8B-Base (few-shot, P/B/E only): token-probability AUROC 0.98 on recall, 0.99 on computation (within-level 0.95 / 0.91), sensitivity 0.73–0.88, overconfidence between −0.15 and +0.05 at both ends. This is Kalai et al.'s prediction (pre-training gives calibrated probabilities; post-training erodes them).
6. **Before vs after answering (D vs A): no general winner.** Paired, within-level, Haiku's worked-out replies dropped: D beats A on computation for Qwen3-8B (+0.21 [0.08, 0.34]) and GPT-4o (+0.14 [0.02, 0.24]), but loses for Qwen3-14B (−0.21) and, on perception, for Qwen3-8B thinking (−0.24); on recall A is as good or better (Haiku −0.15, Qwen3-1.7B −0.17, Qwen3-8B −0.13). The part 2 impression that "before beats after on computation" was mostly level tracking. Figure: `d_vs_a.png`.
7. **Models react to a penalty, not to its size.** With the Kalai threshold rule in the prompt, answer rates drop from t = 0 to t = 0.5 and then stay flat through 0.75 and 0.9 (e.g. GPT-4o recall 0.67 → 0.62 → 0.62 → 0.62; GPT-5 nano and Claude Haiku answer every computation item at every t). A model following the rule would abstain more as t rises. Figure: `kalai_answer_rates.png`.
8. **A stated penalty changes behaviour in other ways.** Claude Haiku shows its working instead of abstaining: 12% of computation answers at t = 0 contain worked steps, 76% at t = 0.9, and accuracy rises from 0.82 to 1.00. On false premises, the penalty turns corrections into refusals (Haiku abstains on 0% of plain answers, 60% at t = 0.9; most Qwen configs fall to 0% premise correction under any threshold).
9. **Ill-posed questions separate the models most.** Plain-answer right behaviour: Claude Haiku corrects 100% of false premises and asks which work is meant on 75% of ambiguous titles; every other model picks one work on most ambiguous titles (GPT-5 nano 42%, GPT-4o 22%, the rest ≤ 8%). Unknowable questions are easy for all (84–100% abstain). Figure: `behaviour.png`.
10. **Stated confidence means different things on behaviour items.** After a response (A), GPT-4o, Grok and Qwen3-8B/14B rate the response itself: they give 0.85 to 1.0 when they abstain on an unknowable question ("confident it can't be known"); Claude Haiku, GPT-5 nano and Qwen3-1.7B give a correct abstention low confidence instead (0.07 to 0.37). Before (D), confidence means "can I answer": GPT-4o's D is 0.12 when it then abstains and 0.80 when it commits. This answers the office report's open question about abstentions at 0.99.
11. **Forecasting: no model beats always guessing the base rate.** Brier on 75 post-cutoff market questions: Claude Haiku 0.243, GPT-4o 0.252, the rest 0.26–0.31, against the base rate 0.227 and the crowd 0.190. The loss is mostly calibration (Murphy reliability 0.05–0.12 vs the crowd's 0.03); AUROC 0.50–0.61 vs the crowd's 0.74. Averaging the 10 samples (FB) helps slightly (Haiku 0.232). Figure: `forecasts.png`.

## Caveats

- **Claude Haiku works problems out before its JSON** in A and D (86% / 73% of computation replies), against "Reply with only a JSON object". Its A/D accuracy on computation is therefore 1.00 versus 0.80 for the plain answer, its D is not confidence *before* answering there, and on ill-posed items a clarification in the preamble is not seen by the judge (its A/D ill-posed rates are understated; its P rates are not affected). Flagged per row (`preamble`); excluded from D vs A.
- **Cells with too few errors are not reported** (fewer than 10 right or 10 wrong answers): GPT-5 nano and Qwen3-8B thinking on computation and perception, Haiku A/D on computation.
- **Grok and GPT-4o often give one confidence value to every answer in a cell**, which makes AUROC exactly 0.5 and the A slope on computation unstable (interval +0.16 to +9.71).
- **The judge** agrees with hand labels on 114/120 (gold set), 39/40 (Qwen held-out) and 43/48 (API held-out, all disagreements stricter than the hand label).
