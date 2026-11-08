# Epic 2 · Phase 1B — Probing Internal Confidence (Detailed Plan)

Follow-on to [Phase 1](epic2_phase1_plan_03-11-2026.md) ([results](../experiments/epic2/phase1_confidence/RESULTS.md)). Budget: about two days, mostly analysis; GPU work runs on the 4090. No API cost.

> **Status (2026-11-08): plan drafted, awaiting approval. Step 1 (study) done.**

## Question

Phase 1 found that what a model *does* (sample agreement B, token probability E) tracks correctness better than what it *says* (stated confidence A, D), and that stated confidence carries no signal at all on perception, with difficulty held fixed. Phase 1B asks the same of the model's hidden states:

1. **When does the signal appear?** A linear probe on the state *before* the model writes its answer vs *after*. Prediction: on **recall**, before ≈ after (whether the fact is stored is settled before generation); on **computation**, before ≪ after (whether the product comes out right is decided during generation). Perception: no firm prediction; if the probe works there while stated confidence does not, the model "knows" more than it says.
2. **Does the representation know more than the words?** Probe vs A, D, B, E on the same test questions for the same model (Kalai et al.'s latent-vs-output point).
3. **Does post-training damage the representation or only the output?** Qwen3-8B vs Qwen3-8B-Base. Phase 1 found the base model's token probabilities best calibrated; does the base model's probe also beat the instruct model's?
4. **Questions without a proper answer (secondary).** Does the state before answering predict whether the model will abstain on unknowable and ill-posed questions?

## Study: Ni et al. (ACL 2025)

[*Towards Fully Exploiting LLM Internal States to Enhance Knowledge Boundary Perception*](https://aclanthology.org/2025.acl-long.1184/), [code](https://github.com/Trustworthy-Information-Access/LLM-Knowledge-Boundary-Perception-via-Internal-States).

- **Models:** Llama2-7B-Chat, Llama3-8B-Instruct, Qwen2-7B-Instruct, Llama2-13B-Chat; greedy answers.
- **Data:** NQ and HotpotQA (factual, free-form), MMLU (multiple choice). Probe trained on 1,000 correct + 1,000 wrong answers (balanced by sampling), a 10,000-example run as a data-size check.
- **States:** one middle layer (16 of 32 for Llama2-7B, after Azaria & Mitchell 2023), no layer sweep. *Pre-State* = last token of the question. *Last State* = last generated token. *Avg State* = mean over the generated tokens.
- **Probe:** MLP 512 → 64 → 32 → 2, ReLU, Adam, lr 5e-5, 30 epochs, three seeds averaged.
- **Labels:** correct if the gold answer string is contained in the response.
- **Metric:** "alignment", the share of questions where the binary confident/not-confident prediction matches correctness. No AUROC, no calibration.
- **Result:** before is nearly as good as after. Alignment gap (after minus before): NQ −0.04 to +2.5 points, HotpotQA −1.6 to +2.0, MMLU +2.0 to +5.9. More data raises both levels but leaves the gap unchanged. With 2,000 examples the Qwen2-7B pre-state probe collapsed to "never confident"; 10,000 fixed it.
- **C3:** asks the model for 10 candidate answers, rewrites the question as multiple choice with 2 to 8 options, and lowers the confidence when the probe's confidence on the original and multiple-choice versions disagree. It needs the probe (internal states), so it does not carry over to API models.

**What we change, and why.**

| Ni et al. | Phase 1B | Why |
| --- | --- | --- |
| Factual QA (+ MMLU) | Recall, computation, perception with a difficulty knob | Their gap is largest on MMLU, the one set with reasoning, but they never test computation; that is our key test |
| Alignment (accuracy of a binary call) | AUROC and within-level AUROC, bootstrap CIs | Alignment moves with the base rate; within-level AUROC controls difficulty, as in Phase 1. Their "better perception on harder questions" may be a base-rate effect |
| One middle layer | Every layer, chosen on a validation split | Qwen3 has 36 layers; we report the curve, which also shows *where* the signal sits |
| MLP | Logistic regression (L2, tuned); their MLP as a check at the chosen layer | A linear probe is the standard reading of "linearly available"; 4,096 inputs and ~2,000 examples favour a regularised linear model |
| Balanced 1,000 / 1,000 | All training answers, class-weighted | Keeps every example; avoids the collapse they saw |
| Containment match | Phase 1 grading (rules, then the validated judge) | Same labels as Phase 1, so the comparison with A/B/D/E is exact |

## Design

**Models (4090, HF transformers, bf16).** Qwen3-8B (thinking off; the Phase 1 anchor), Qwen3-8B-Base (5-shot prompt as in Phase 1), Qwen3-1.7B (size comparison, cheap). Qwen3-14B does not fit in bf16 and the thinking config has no clean "after" position; both are out unless time allows.

**Questions.**

- **Test set:** Phase 1's answerable questions (450: 150 per kind, 30 per level) plus unknowable and ill-posed (155). Unchanged, so every probe score sits next to the Phase 1 numbers for the same question.
- **Training set (new):** about 2,000 per answerable kind from the same generators with the balance rules relaxed, none overlapping the test set. Computation and perception: as many as needed at each level. Recall: the same five relations, all ~4,000 remaining PopQA candidates available, but thin at the ends of the popularity scale (level 1 has ~27 left, level 5 ~200), so the training level mix is uneven. That is acceptable: the training set teaches the probe; the test set measures it.
- **Labels for the training set:** a P-only run of the Phase 1 task (`-T methods=p`) on the local models, graded with the same rules and judge.

**States extracted (per question, every layer).**

- *Before (question):* last token of the user message (Ni et al.'s Pre-State).
- *Before (generation start):* last token of the full prompt, after the chat template's assistant header, the position from which the first answer token is predicted.
- *After (last):* last token of the model's logged P answer, teacher-forced.
- *After (mean):* mean over the answer tokens.

The logged answer is fed back rather than regenerated, so the state belongs to exactly the answer that was graded.

**Probe.** Per model, kind, position and layer: standardise, logistic regression with L2, strength chosen by 5-fold CV on the training set; the best layer per (model, position) chosen on a held-out 20% of the training set, never on the test set. Trained per kind and pooled over kinds; also train on one kind, test on another (is there one shared "I know this" direction?).

**Scores.** On the test set: AUROC and within-level AUROC per kind with bootstrap CIs (`stratified_auroc`, `bootstrap_ci`); paired bootstrap for before vs after and for probe vs A/D/B/E on the same questions; the Phase 1 cell-fixed-effects regression with the probe added as a method.

**Controls.**

- *Metadata probe:* logistic regression on what the question builder knows (log popularity, digit counts, word length, word frequency). If the hidden-state probe barely beats this within levels, it is reading difficulty, not self-knowledge.
- *Embedding layer (layer 0):* at the "before" position this holds only the last token's identity; it is the floor.
- *Shuffled labels:* AUROC should be 0.5.

**Where the work runs.** Hidden states are large (Qwen3-8B: 37 layers × 4 positions × 4,096 values × 2 bytes ≈ 1.2 MB per question, ~8 GB for 6,600 questions), so extraction *and* probe fitting run on the PC; only per-question probe scores and summary tables come back to the Mac (committed). The Mac runs the analysis, figures and unit tests on synthetic arrays; no model is loaded on the Mac.

## Steps

1. **Study.** Ni et al. (above). *Done 2026-11-08.*
2. **Training questions.** `build_train_questions.py`: reuse the Phase 1 generators with exclusion of test items and relaxed balance; write `data/epic2/confidence/train_questions.jsonl`; report counts per kind × level. Mac, no model.
3. **Training labels.** `run_4090.sh train`: P-only runs for the three configs, then `judge_4090.sh` on them. Check accuracy per kind × level and the class balance (each kind needs a few hundred wrong *and* right answers per model).
4. **Extraction.** `extract_states.py`: rebuild each P prompt exactly as in the log, check the token IDs, run one forward pass on prompt + logged answer, save the four positions at every layer (float16, gitignored). `--limit 20` smoke run first on the 4090: prompt reproduces, HF greedy answer matches the logged P answer on nearly all items, shapes as expected. Then the full run.
5. **Probes.** `probe.py` on the PC: layer sweep, chosen layer, per-question test scores, controls; writes `results/probe_scores.parquet` and `results/probe_layers.csv`. Unit tests (synthetic arrays) on the Mac.
6. **Analysis.** `analyze.py probes` (merge with `rows.parquet`): the tables above; figures (AUROC by layer, before vs after per kind, probe vs A/D/B/E, base vs instruct, cross-kind transfer).
7. **Close-out.** `RESULTS_1B.md` in the Phase 1 folder; the second "Inside …" post on lifeandtimes, figures first.

## Predictions (written before the runs)

- Recall: before-probe within-level AUROC ≥ 0.7 and within 0.05 of after.
- Computation: before clearly below after (difference CI excluding 0).
- Probe ≥ stated confidence (A, D) on every kind; probe ≈ B, E on recall.
- Base model's probe at least as good as the instruct model's.
- Cross-kind transfer weak (recall-trained probe near chance on computation).

## Out of scope

- C3 and other confidence-calibration methods (would need extra generations).
- Steering or editing along the probe direction (a candidate sprint extension).
- Thinking mode and models that don't fit the 4090 in bf16.
