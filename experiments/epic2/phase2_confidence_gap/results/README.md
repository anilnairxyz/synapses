# Phase 2 results

## E1 — Stereotype analysis (2026-11-14)

`analyze_stereotype.py` on Phase 1's test-set runs (stated confidence A and D, all eight configs that state one) and Phase 1B's probe scores (Qwen3-8B, Qwen3-1.7B). Tables: `e1_stereotype.csv` (within-level Spearman correlations of stated confidence with the question's look, being right, and the probes, with bootstrap intervals and paired differences), `e1_regression.csv` (open models: standardised betas of stated confidence on the look and both probes), `e1_null_check.csv`. Figure: `figures/e1_stereotype.png`.

**Method notes.** All comparisons use only variation inside strict strata (level × task family or relation). The question's look is one detail per kind, fixed in advance: subject popularity (recall), minus number size (computation), minus word length (perception). A fitted, cross-validated version was tried first and dropped because out-of-fold predictions anti-correlate with the target under no signal (shuffle check, mean −0.09); the fixed detail's shuffle check centres on 0.001. Only strata where stated confidence varies are used, so cells shrink a lot for the models that use few values (Qwen3-8B: 2 to 5 distinct values; 20 to 120 of 150 answers usable).

**Findings.**

- **Recall: not a stereotype.** Stated confidence follows being right (0.2 to 0.5 for most models, e.g. GPT-4o A 0.40 [0.14, 0.54], Grok A 0.49 [0.27, 0.70], Qwen3-8B thinking A 0.38 [0.23, 0.52]) more than subject popularity (−0.14 to 0.37, mostly near 0); for four configs the look-minus-right difference excludes 0 in favour of being right. One exception: Qwen3-1.7B stated before (D) follows popularity (0.39) and not being right (0.07).
- **Computation: largely a stereotype.** Stated confidence follows number size about as much as, or more than, being right: Claude Haiku D 0.37 [0.21, 0.48] (being right not measurable: nearly all right), Qwen3-14B A 0.30 vs 0.24, GPT-4o D 0.27 vs 0.28, Qwen3-1.7B D 0.31 vs 0.10 (difference 0.21 [0.05, 0.39]).
- **Perception: neither, reliably.** Being right ranges from −0.17 to 0.30 with wide intervals; few models vary their stated confidence within levels at all.
- **Stated confidence does read the hidden state, coarsely.** For the open models it correlates with the probe at the same timing, most clearly when stated before answering, against the before-answer probe (0.32 to 0.52 on recall and computation; Qwen3-8B D on computation 0.52 [0.35, 0.64]), weakly or not at all for Qwen3-8B stated after (−0.13 to 0.18). For Qwen3-8B D, the regression keeps the before-answer probe with number size in the model: standardised beta 0.43 [0.23, 0.61] on computation (number size 0.16 [−0.05, 0.35]) and 0.38 [0.15, 0.55] on recall. So the words are not cut off from the state: they follow it, with very few distinct values.

**Against the prediction** (written before the run: "within a level, stated confidence is predicted at least as well by the question's details as by the hidden-state probe; on perception, by neither"): **not supported on recall** (the probe and being right lead; Qwen3-1.7B A recall, look − probe −0.30 [−0.53, −0.04]); **supported on computation** for the after-answer probe (differences include 0) but the before-answer probe leads for Qwen3-8B D; **supported on perception** (neither, though cells are small).

**What it means for the story.** Hypothesis 6 holds where the model has no internal access to its error before answering (computation), not where it does (recall). Stated confidence is a low-resolution readout of the state rather than a disconnected stereotype, which points E4 at resolution (how many distinct values, which direction they follow) and keeps E2 and E3 focused on whether any readout recovers the within-level information the probe has.

## E2 and E3 — Swap and elicitation (2026-11-15)

`analyze_swaps.py` on the full 4090 run (`extract_<config>.parquet`, the swap logs) with Phase 1's methods and Phase 1B's out-of-fold probe scores. Tables: `e2_means.csv`, `e2_separation.csv`, `e3_auroc.csv`, `e3_calibration.csv`, `tail.csv`, `tail_quarter.csv`, `ptrue_excluded.csv`. Figures: `figures/e2_separation.png` (key figure), `figures/e2_means.png`, `figures/e3_readouts.png`.

**Run checks.** Every swap reply parsed (2,658 and 2,676, no errors). The refitted probes reproduce Phase 1B on the model's own answers (r = 0.9999 for Qwen3-8B and the base model on both positions; 0.9959 before / 0.9997 after for Qwen3-1.7B). P(True) answers left out under the rule fixed before analysis (< 0.5 probability on the two letters): 16 of 7,075 (Qwen3-8B) and 236 of 7,084 (Qwen3-1.7B), all in the neutral zero-shot variant.

**E2: the same question, a correct and a plausible wrong answer.** Separation = share of the 150 questions per kind where the correct answer gets more confidence (0.5 = none), 95% bootstrap intervals.

- **"Your answer": no separation at all.** With the answer in the assistant turn, verbal confidence is 0.50 to 0.55 on every kind for both instruct models; Qwen3-8B gives 0.97 to 1.00 on average to planted wrong answers and puts 0.9 or more on every one of them (100% on all three kinds). The model is certain of whatever it is made to have said.
- **"My answer" (the user's): partial separation, except perception.** The same answers attributed to the user: Qwen3-8B 0.64 [0.59, 0.70] recall, 0.67 [0.62, 0.72] computation, 0.52 [0.50, 0.54] perception; Qwen3-1.7B 0.54, 0.63, 0.56. Confidence in a planted wrong answer drops by 0.21 (recall) and 0.37 (computation) for Qwen3-8B and 0.31 to 0.59 for Qwen3-1.7B when it is the user's: Sanz-Guerrero et al.'s ownership bias, replicated, and here it does not only inflate confidence, it removes its ability to tell right from wrong.
- **P(True) separates moderately in both placements** (0.53 to 0.74 for the instruct models, 0.65 to 0.93 for the base model). Placing the answer in the assistant turn raises P(True) a lot (Qwen3-8B 8-shot: +0.26 to +0.33 recall, +0.41 to +0.60 computation, +0.66 to +0.73 perception) but changes separation much less.
- **The probe separates, but much of it is familiarity.** On computation the probe after the answer separates 0.95 (8B), 0.85 (1.7B), 0.99 (base), but the answer's own likelihood under the model does almost as well (0.93, 0.83, 0.97): a planted number is one the model would not have written. Comparing only answers of similar likelihood, the probe keeps 0.80 (8B) and 0.74 (base) on computation, 0.57 for 1.7B; on recall and perception it falls to 0.53 to 0.58, about where P(True) is (0.48 to 0.71). The layer-0 control is near 0.5 on computation (0.50 to 0.53), 0.46 to 0.60 elsewhere. So with planted answers, "the state knows it is wrong" is mostly "the state knows it would not say this", except on computation for the 8B models.

**E3: confidence in the model's own answers, within levels.** Test set (150 per kind, wide intervals) and the ~5,700 training questions (P(True) and out-of-fold probe only, tight intervals).

- **The probe after the answer is the ceiling everywhere:** training questions 0.86 to 0.91 on recall and computation, 0.70 to 0.72 on perception, for all three models.
- **P(True) is the best way of asking, still below the probe.** Training questions, best variant per cell: Qwen3-8B 0.81 recall, 0.79 to 0.80 computation, 0.69 to 0.70 perception (0.02 to 0.10 below the probe; on perception within 0.02, intervals including 0); Qwen3-1.7B 0.69, 0.76, 0.59 (0.07 to 0.21 below).
- **What the model says is far below.** Test set: stated with the answer (A) 0.50 to 0.68; post-hoc "your answer" 0.49 to 0.68 (0.20 to 0.32 below the probe, intervals excluding 0); user framing 0.52 to 0.77, better than post-hoc on recall and computation, not perception.
- **Neutral beats own placement on recall.** Training questions: Qwen3-8B 0.81 neutral vs 0.74 own (8-shot), Qwen3-1.7B 0.69 vs 0.57; computation level; perception 0.70 vs 0.65.
- **The base model.** Its P(True) is not better at discrimination than the instruct model's (training: 0.77 / 0.71 to 0.74 / 0.54 vs Qwen3-8B's 0.81 / 0.79 / 0.70), but its token probability (E) remains the best output-level readout in the study (test 0.95 recall, 0.91 computation), as in Phase 1.
- **Calibration** (ECE, training questions): the instruct models' own-placement P(True) is badly overconfident (0.27 to 0.96); neutral 8-shot is the best calibrated readout for Qwen3-8B on computation and perception (0.06, 0.05), not recall (0.47); the base model's P(True) 0.07 to 0.40.

**The confident tail.** Qwen3-8B states 0.9 or more (A) on every test answer, and 60% to 76% of those answers are wrong. The pre-specified matched-share comparison is therefore degenerate for Qwen3-8B (matching a 100% share flags everything); the secondary measure, added after seeing that, is the error rate in each readout's most confident quarter (`tail_quarter.csv`): the probe's top quarter has the fewest errors in every cell (Qwen3-8B training computation 3%, against 10% to 19% for P(True) and 66% overall; recall 52% vs 60% to 66% and 86% overall), while stated confidence's top quarter is about as wrong as the average answer (test: A 58% vs 60% overall on computation, 62% vs 63% on perception).

**Predictions** (written in the plan before the runs):

| Prediction | Outcome |
| --- | --- |
| E1: stated confidence follows the question's details at least as well as the probe; perception: neither | Not supported on recall; supported on computation and perception (see E1) |
| E2: about an answer in its own mouth, nearly as sure of wrong as right | **Supported, strongly**: separation 0.50 to 0.55; 100% of planted wrong answers at ≥ 0.9 (Qwen3-8B) |
| E2: the probe on the same answers tells them apart | Supported before the familiarity control; after it, only on computation for the 8B models (0.80, 0.74) |
| E2: ownership bias replicates | **Supported**: +0.2 to +0.6 on wrong answers (verbal); +0.1 to +0.7 (P(True)) |
| E2: ownership bias largest on perception | Mixed: verbal, no (Qwen3-8B is at ceiling either way on perception); P(True), yes for Qwen3-8B (+0.66 to +0.73) |
| E3: P(True) and user framing beat stated confidence within levels on every kind | P(True): yes for Qwen3-8B, not on perception for Qwen3-1.7B; user framing: recall and computation, not perception |
| E3: both stay below the probe | **Supported** (training-set intervals exclude 0, except Qwen3-8B perception) |
| E3: the gap to the probe is largest on perception | Not supported (smallest on perception for Qwen3-8B) |
| E3: own placement raises P(True) but changes AUROC by < 0.05 | Raises it: yes; AUROC: computation yes, recall no (−0.07, −0.12) |
| E3: base better calibrated, AUROC within 0.05 of instruct | Calibration partly (better than own placement, not than neutral 8-shot); AUROC not supported: the instruct model's P(True) discriminates better (up to +0.16, perception) |
| Tail: most wrong answers at ≥ 0.9 (Qwen3-8B, computation and recall) | **Supported**: all of them |
| Tail: the probe at the same flag rate gives high scores to fewer wrong answers | Not testable as specified (rate 100%); secondary measure: the probe's top quarter has the fewest errors everywhere |

**What it means for the story.** The model's verbal confidence in an answer depends on whose answer it is, not on whether it is right: in its own turn, separation vanishes (0.50 to 0.55) while the same model, judging the same answer as the user's, partly separates on recall and computation. The information is in the model (P(True) recovers much of it; the probe more), so the failure is in the act of reporting on one's own answer (hypothesis 5, rationalisation, for verbal reports), on top of a coarse readout (E1). Usable result: for these models, do not ask "how sure are you of your answer"; ask P(True) about the answer framed neutrally, with worked examples, or use the probe; never trust a stated 0.9.

## E2b — Beyond Qwen: Claude Haiku 5.5 (2026-11-15)

Verbal self vs user on Claude Haiku 5.5 (API, from the Mac; Phase 1's settings: no temperature, reasoning effort none; max_tokens 1,024 because under "my answer" it often works the problem in the reply). 2,638 replies, none failed, unparsed or truncated. Same swap construction: its own Phase 1 answers, the correct answer and a plausible wrong one in its format (all injected answers grade as built). Key figure: `figures/e2_ownership.png`.

| Separation (150 questions per kind) | "Your answer" | "My answer" |
| --- | --- | --- |
| Computation | 0.58 [0.52, 0.63] | 0.99 [0.97, 1.00] |
| Perception | 0.60 [0.55, 0.64] | 0.86 [0.81, 0.90] |
| Recall | 0.79 [0.73, 0.85] | 0.85 [0.80, 0.91] |

- **The ownership effect is larger than in Qwen.** A planted wrong product gets 0.9 or more in 78% of cases as "your answer" and in none as the user's; moving the same wrong answer to the user's turn lowers confidence by 0.80 [0.74, 0.85] on computation and 0.77 [0.70, 0.82] on perception.
- **Mechanism visible in the replies:** under "my answer" Haiku recomputes the answer in its reply ("85 × 98 = 8500 − 170 = 8330. The answer 8830 is incorrect."); under "your answer" it states a confidence without checking. Owning the answer appears to switch off verification. Recall, where there is nothing to recompute, shows the smallest gap.
- **Prediction** ("Haiku shows the same pattern: separation under 'your answer' within 0.05 of 0.5 on at least two kinds, and higher under 'my answer'"): **partly supported.** Higher under "my answer" on every kind (much higher on computation and perception), but "your answer" is above chance (0.58, 0.60, 0.79), not within 0.05 of 0.5 on any kind: Haiku keeps some separation in its own turn, Qwen almost none.

## E2b — Qwen3-8B with thinking on (2026-11-15)

Verbal self vs user on Qwen3-8B with thinking on (4090, vLLM; Phase 1's settings: temperature 0.6, max_tokens 6,000; the thinking trace stripped before parsing). 2,670 replies, no errors; 3 cut off at the token limit and so unparsed, left out (0.1%).

| Separation (150 questions per kind) | "Your answer", no thinking | "Your answer", thinking | "My answer", thinking |
| --- | --- | --- | --- |
| Computation | 0.51 [0.50, 0.52] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| Perception | 0.50 [0.50, 0.51] | 0.88 [0.84, 0.91] | 0.93 [0.91, 0.96] |
| Recall | 0.55 [0.51, 0.60] | 0.72 [0.66, 0.78] | 0.66 [0.59, 0.73] |

- **Thinking removes the effect.** With thinking on, Qwen3-8B gives planted wrong products 0.00 and correct ones 1.00 even as "your answer"; planted wrong perception answers get 0.9 or more in 25% of cases (100% without thinking). The ownership difference on wrong answers is 0.00 on computation and 0.15 on recall and perception, against 0.21 to 0.37 without thinking.
- **What it means.** The gap is not about ownership as such but about checking: without reasoning, a model rates an answer in its own turn without checking it; Claude Haiku checks the user's answer in its visible reply, and thinking makes the model check its own as well. This matches Tsui's finding that reasoning-trained models lack the self-correction blind spot.
- **Caveat.** On computation and perception the thinking model recomputes the answer, so its confidence becomes verification, not introspection about its earlier state. Recall cannot be recomputed; there thinking helps less (0.72 vs 0.55 without).
- **Prediction** ("thinking reduces but does not remove it: separation under 'your answer' above 0.5 but below 'my answer'"): **not supported.** Above 0.5 on every kind, but not below "my answer": equal on computation, within the intervals on perception, and slightly above on recall. Thinking removes the gap.

**Updated story.** Without reasoning first, a model's confidence in an answer placed in its own turn barely depends on whether the answer is right (Qwen3-8B and 1.7B: 0.50 to 0.55; Claude Haiku: 0.58 to 0.79), while the same model, judging the same answer as the user's, separates far better (Haiku computation 0.99). The difference is checking: the user's answer is checked, the model's own is not, unless the model reasons before stating its confidence, in which case its own answer is checked too.

## E4 Part A — Which report can be read from the answer's state (2026-11-15)

`directions.py` on the PC (CPU), on the E2 states of every given answer (own, correct, wrong), at the layer the Phase 1B probe chose per kind (`ans_last`). The refitted probes reproduce `extract_answers.py`'s scores exactly (r = 1.0000 for all six). Table: `e4_directions.csv`; directions for steering: `e4_probe_dirs.npz` (pushed in commit c22e8c3, whose message says "Part B" but which holds these Part A outputs). Within-stratum Spearman, 95% intervals over questions.

| | "My answer": readable | "My answer" vs probe | "Your answer": readable | "Your answer" vs probe |
| --- | --- | --- | --- | --- |
| Qwen3-8B recall | 0.46 [0.38, 0.54] | 0.49 [0.40, 0.57] | 0.51 [0.41, 0.58] | 0.50 [0.40, 0.57] |
| Qwen3-8B computation | 0.66 [0.58, 0.73] | 0.39 [0.28, 0.48] | pinned (4 values, all ~1.0) | |
| Qwen3-8B perception | 0.02 [−0.07, 0.13] | 0.15 [0.05, 0.23] | pinned | |
| Qwen3-1.7B recall | 0.59 [0.49, 0.66] | 0.51 [0.40, 0.58] | 0.48 [0.36, 0.55] | 0.50 [0.41, 0.58] |
| Qwen3-1.7B computation | 0.48 [0.37, 0.57] | 0.38 [0.28, 0.47] | 0.47 [0.39, 0.55] | 0.15 [0.05, 0.26] |
| Qwen3-1.7B perception | 0.48 [0.37, 0.55] | 0.21 [0.08, 0.29] | 0.21 [0.10, 0.30] | 0.14 [0.02, 0.23] |

"Readable": the out-of-fold ridge readout from the state against the stated confidence. "Vs probe": the correctness probe's score against the stated confidence. A (stated with the answer) barely varies within levels for either model (nothing to read, except Qwen3-1.7B recall 0.18 and computation 0.29).

- **"My answer" confidence is in the answer's state** on recall and computation (0.46 to 0.66), and follows the correctness probe there (0.38 to 0.51); on perception much less.
- **On computation, Qwen3-1.7B's "your answer" confidence is as readable from the state as its "my answer" confidence (0.47 vs 0.48) but follows the correctness probe far less (0.15 vs 0.38):** the report on its own answer reads something in the state, but not mainly the correctness signal. On recall both reports follow the probe equally (about 0.50).
- **Raw-space cosines are uninformative.** Qwen3-8B computation: −0.94 against the correctness probe, but −0.96 against a probe trained on shuffled labels; fitted directions share the states' dominant axes. Elsewhere 0.06 to 0.27 against −0.16 to 0.04 for the shuffled baseline: small.
- **Prediction (A)** ("'my answer' readable, Spearman ≥ 0.3 on recall and computation; cosine 0.2 to 0.5"): **readability supported** (0.46 to 0.66); **the cosine part is not testable in the raw state space** (the shuffled baseline is as large where the cosine is large).
