# Epic 2 · Phase 2 — Why Confidence Reports Miss What the Model Knows (Sprint Plan)

Follows [Phase 1](epic2_phase1_plan_03-11-2026.md) ([results](../experiments/epic2/phase1_confidence/RESULTS.md)) and [Phase 1B](epic2_phase1b_plan_08-11-2026.md) ([results](../experiments/epic2/phase1_confidence/RESULTS_1B.md)). This is the project for the Apart AI x Epistemics Research Sprint, 13 to 15 November 2026; submissions close Monday 16 November, 11:59 UTC. Rules, report requirements and talks: [sprint brief](epic2_sprint_brief_13-11-2026.md). It replaces the earlier Phase 2 (sycophancy), which moves to later; see the [epic plan](epic2_plan_03-11-2026.md).

> **Status (2026-11-12): draft, awaiting approval.** No code until approved; Step 1 (study) comes first.

## Contribution

*Whether an LLM can know it is wrong depends on how the answer is produced, and its confidence reports fail to reflect what it knows; we test why.*

The first half is measured (Phases 1 and 1B): recall is known before answering, computation only after, perception barely; hidden-state probes beat stated confidence on every kind, and base and instruct models hold the same internal knowledge. The sprint adds the second half with three controlled pieces, not four or five partial ones: a stereotype analysis on existing data, a swap experiment, and an elicitation comparison. Steering and fine-tuning are stretch goals only.

The lesson from two past winners (runaway-agent detection; omission attacks): one sharp claim, something usable, honest scope, a short report. Our edge is rigour; the risk is too much material.

## Hypotheses

Candidate explanations for the gap between what the model represents and what it says. They are not exclusive: 4 is a root cause, 2 says which training, and 1, 3, 5, 6 are mechanisms.

1. **Decoding failure.** The information is internal but the generation does not expose it.
2. **Instruction tuning.** Post-training pushes towards confident answers over the latent uncertainty.
3. **Elicitation failure.** The model could expose it, but "give your confidence" is a poor readout.
4. **Objective mismatch.** Nothing in training rewards turning internal uncertainty into an honest number.
5. **Rationalisation.** Once committed, the model rates the answer it gave, not its uncertainty.
6. **Stereotype.** Stated confidence comes from what the question looks like (long numbers, an obscure name), not from the model's state about this answer.

| Experiment | Tests | Main prediction (written before the runs) |
| --- | --- | --- |
| E1 Stereotype analysis | 6 | Within a level, stated confidence (A, D) is predicted at least as well by the question's details as by the hidden-state probe; on perception, by neither |
| E2 Swap | 5 (and 1) | Asked about an answer placed in its own mouth, the model is nearly as sure of a wrong answer as of the right one, while the probe on the same answer tells them apart. Ownership bias (Sanz-Guerrero et al.) replicates and is largest on perception |
| E3 Elicitation | 3 (and 2, via the base model) | P(True) and user framing beat stated confidence within levels on every kind but stay below the probe; the gap to the probe is largest on perception |

## Study (Step 1)

Before any code, check what is already known, and record it here:

- **Ji et al. 2025**, *Calibrating Verbal Uncertainty as a Linear Feature to Reduce Hallucinations*: verbal uncertainty as one direction, separate from the model's actual uncertainty, steered. Closest prior work to hypotheses 1 and 3; confirm the claim and what it leaves open.
- **Kadavath et al. 2022**, [*Language Models (Mostly) Know What They Know*](https://arxiv.org/abs/2207.05221): P(True) exactly as they defined it (prompt, few-shot, which token), so E3 reproduces it. **Read 2026-11-14** (notes below).
- **Lin, Hilton & Evans 2022**, *Teaching Models to Express Their Uncertainty in Words*: the fine-tuning evidence for hypothesis 4 (stretch).
- **Sanz-Guerrero, Mager & von der Wense 2026**, [*Large Language Models Are Overconfident in Their Own Responses*](https://arxiv.org/abs/2606.03437) (Findings of ACL 2026). Found by the user, 2026-11-12; **read in full 2026-11-14** (notes below). Ownership bias, instruction tuning and the user-framing fix. **This covers E2's attribution factor and part of hypothesis 2.**
- A search for further prior work that puts a different answer in the model's mouth and asks for its confidence (prefill, self-preference), so E2's novelty is stated, not assumed.

Provenance of this list: Sanz-Guerrero et al. was found by the user; Kadavath et al. was checked by search and then read; Ji et al. and Lin et al. are from memory and not yet checked against the source. The search in the last bullet has not been done, and Sanz-Guerrero et al. cite no prefill, self-preference or self-recognition work, so it cannot be borrowed from them.

### Sanz-Guerrero et al.: full read (2026-11-14)

Read from the arXiv HTML (v1). The prompt templates are images and were not readable; dataset sizes and checkpoint IDs are not stated.

- **Models:** six base/instruct pairs (Llama 3.1 8B/70B, Qwen3 4B/30B, Gemma 3 4B/27B), instruct run with and without the chat template; GPT-5.2 on MMLU only.
- **Benchmarks:** MMLU (four options, the main one), GSM8K (numeric maths), TruthfulQA, open-ended MMLU.
- **Readouts:** P(True) (after Kadavath et al.), a verbalised percentage, a seven-point verbal scale ("very low" to "very high").
- **Own vs user:** two prompts that differ only in whether the answer sits in the assistant or the user turn. On MMLU they ask for confidence in each of the four options in turn, so most "own" answers are options the model did not choose, placed in its turn.
- **Metrics:** ECE, Brier, mean confidence; Wilcoxon and bootstrap tests. **No AUROC or other discrimination measure, and no hidden-state analysis.**
- **Instruction tuning:** on MMLU, averaged over the six models, accuracy +3.7 points but ECE +13.1 points; the chat template adds another 2.7. Asking for confidence directly does not remove the gap (Gemma 3 4B, P(True): ECE 0.02 base vs 0.65 instruct with chat).
- **Ownership bias:** confidence higher in the assistant turn by 0.16 (P(True)), 0.18 (verbal %), 0.27 (verbal scale) on average. The headline "up to 26%" is the verbal-scale *average*, an absolute difference, not a maximum or relative; the largest single case is Qwen3-30B at 0.46. On GSM8K, +0.20 (P(True)).
- **Right vs wrong:** one qualitative sentence from a plot, confidence is "largely independent of accuracy". In the assistant turn the four options' confidences sum to roughly 2 to 3, where calibrated would be 1.
- **The fix** (answer in the user turn) cuts ECE by 0.10, 0.18, 0.26 but recovers base-model calibration only partly: better than base for the verbal scale, slightly worse for verbal %, about twice base's ECE for P(True).
- **Their conclusion**, that reliable confidence needs "decoupling the generation of an answer from its evaluation" (an "observer" role), is close to our hypothesis 5.

**What changes after Sanz-Guerrero et al.**

- *Self vs user attribution* is a direct replication, not a contribution: same answer text, only the turn changes, and Qwen3 is in both studies.
- *Correctness by construction* is only partly new: their four MMLU options already put right and wrong answers in the assistant turn. What is new is measuring **discrimination** (within-level AUROC and the paired right − wrong difference) where they report calibration only, and using the model's real free-text answer against a matched plausible wrong one rather than fixed options.
- *By kind of question:* GSM8K covers computation for calibration, so their fix is not untested there; **perception is not covered**, and neither is difficulty held fixed. The prediction that ownership matters most where the model has no internal access (perception) stays.
- *The probe on the same answers* is new: it says whether the model's state knows the answer is wrong, whoever is said to have given it.
- *E3:* P(True) and user framing are both among their readouts, for calibration; what is new is comparing them within levels against the probe ceiling.
- *Hypothesis 2 via the base model:* they show instruction tuning hurts calibration, so our base-model P(True) mostly replicates that. The new question is whether it also hurts *discrimination*, given Phase 1B found base and instruct hold the same internal knowledge.

The sharpest new angle: they measured how confident models are, never whether that confidence tells a right answer from a wrong one; their own plot suggests it does not, but unquantified. E2 and E3 measure exactly that, with the probe as the reference. Cite their 26% as an average absolute difference.

### Kadavath et al.: read (2026-11-14)

Read from the arXiv HTML (v4): all of the main text and the start of the appendices. Figures are images, so per-task, per-size P(True) numbers (AUROC, Brier, ECE) are not available, only the trends stated in the text.

- **Models:** Anthropic pretrained models of 0.8B, 3B, 12B, 52B; RLHF policies appear only in the temperature section.
- **Datasets:** multiple choice (BIG-Bench, MMLU, TruthfulQA, LogiQA, QuALITY); free answers (TriviaQA, LAMBADA, HumanEval, GSM8K, arithmetic, Python functions). Few-shot examples are drawn at random from the test set itself.
- **P(True) prompt** (the model first answers at T = 1, then sees):

  ```
  Question: Who was the first president of the United States?
  Proposed Answer: George Washington was the first president.
  Is the proposed answer:
  (A) True
  (B) False
  The proposed answer is:
  ```

  The readout is the probability on " (A)" vs " (B)"; whether they renormalise over the two is not stated. Either 20-shot or a fixed six-example prompt ("simply the first prompt we tried"). A *brainstorm* variant first shows five of the model's own samples, then asks about one; it "improves significantly" on short-answer tasks, less on GSM8K and code.
- **Multiple choice:** large models are well calibrated if they see lettered options; better with size and with few-shot. "None of the above" hurts badly.
- **P(True)** separates right from wrong better with size, but zero-shot it sits near 50% and is badly calibrated: "models do a good job separating correct and incorrect samples, but their calibration for P(True) is quite poor". Few-shot is needed for calibration.
- **Own answers are harder to judge**, "since models tend to find their own samples more plausible"; a model has no explicit way to know whether tokens are its own or a third party's. This is the 2022 precursor of Sanz-Guerrero et al.'s ownership bias. Samples from smaller models are easier for every model to judge.
- **P(IK):** a value head, with the whole model fine-tuned to predict the fraction of 30 T = 1 samples it gets right. 52B, AUROC on tasks outside training (TriviaQA only → all but GSM8K): arithmetic 0.93 → 0.99, GSM8K 0.62 → 0.75, LAMBADA 0.61 → 0.85. Responds to context: a Wikipedia article raised one question from 18% to 78%; bad or irrelevant hints lower it. All pooled across difficulty.
- **RLHF:** policies "naively appear very miscalibrated", but one temperature, T = 2.5, "largely fixes" it on three evaluations; they caution that heavier RL might distort calibration beyond what temperature can fix.
- **Entropy** of the answer is "not a robust measure" of knowing, especially for large models (relevant to our E).

**What changes after Kadavath et al.**

- *P(True) is specified* (see E3): their prompt verbatim; zero-shot and few-shot both reported; few-shot examples from the **training** questions with known labels, not from the test set as they did; probability renormalised over " (A)" and " (B)", one forward pass.
- *Their format is already the observer framing:* "Proposed Answer" is plain text owned by no one. So P(True) is run with the prompt in the user turn (neutral) and, for the instruct models, with the model's answer in the assistant turn before it (own): an ownership test with the same readout.
- *Calibration vs discrimination:* a temperature rescales confidence without changing its ranking, so calibration gaps like Sanz-Guerrero et al.'s can be partly cosmetic; discrimination cannot be fixed that way. This is why E2 and E3 lead with within-level AUROC; it also fits Phase 1B (base and instruct hold the same internal knowledge). ECE and Brier are reported alongside so the results compare with Sanz-Guerrero et al.
- *Related work:* P(IK) is the trained precursor of our probe (correctness predicted before answering, from the model). Ours is a linear readout on frozen states, and is measured within levels; their AUROCs are pooled across difficulty, and arithmetic at 0.93 to 0.99 is very likely the coarse easy-vs-hard signal that Phase 1 showed shrinks within levels. "Own samples more plausible" is cited with Sanz-Guerrero et al. as the origin of the ownership idea.
- *Optional readout:* brainstorm P(True), using the Phase 1 samples already drawn for B; tests whether showing the samples adds anything beyond B. Optional, so E3 does not grow.

Output: a short "what we add" paragraph, and any change to E1 to E3.

## Design

**Models.** Qwen3-8B and Qwen3-1.7B (thinking off) and Qwen3-8B-Base, on the 4090, as in Phase 1B: probes, labels and states already exist. API models for the behavioural parts of E2 and E3 only if time allows (GPT-4o via Foundry, run elsewhere; Claude Haiku 5.5), to answer "small open models only".

**Questions.** The Phase 1 test set's answerable questions (150 per kind, 30 per level), so every number sits next to Phases 1 and 1B and is scored by probes that never saw the question. E3's cheap readouts also run on the 5,995 training questions, where the out-of-fold probe scores give tight intervals.

### E1 — Stereotype analysis (Mac, existing data)

For every config with stated confidence: within level × task type, how well does stated confidence (A, D) track (a) the question-details probe score, (b) the hidden-state probe score (open models), (c) correctness? Spearman correlations with bootstrap intervals, and a regression of stated confidence on details and probe score with cell fixed effects. Caveat to report: Qwen3-8B states 0.9 or more almost always, so ties are heavy; the analysis uses rank methods and says how many distinct values each config used.

### E2 — Swap

Each question is asked as a fresh conversation in which the answer is already given, then the model is asked for its confidence. Two factors, crossed:

- **Answer:** the model's own Phase 1 P answer; the **correct** answer; a **plausible wrong** answer, different from the model's own.
- **Attribution:** *self* (the answer sits in the assistant turn, then the user asks "What is the probability, from 0 to 1, that your answer is correct?") vs *user* (the user says "My answer is X. What is the probability that it is correct?"), matching Sanz-Guerrero et al.'s framing so the ownership effect is a direct replication.

Wrong answers are built to be plausible and certainly wrong:

- *Computation:* change one middle digit, so the length and the leading digits stay right, like the models' real errors.
- *Perception:* count ±1; for nth letter, the neighbouring letter in the word; for reverse, two adjacent letters swapped.
- *Recall:* another model's Phase 1 answer to the same question that was graded wrong; failing that, the answer to another question with the same relation (another director, another author).

Right and wrong are known by construction, so only the model's own answers need Phase 1's labels; no new judging.

**Measures.** Mean confidence per cell; paired difference correct − wrong per question, and within-level AUROC of confidence for correct vs wrong answers. Rationalisation predicts a small difference under *self* and a larger one under *user*.

**Probe on the same answers.** Teacher-force the P prompt plus the correct or the wrong answer in the assistant turn: the same format the after-answer probe was trained on, with only the answer's source changed. Score with the Phase 1B after-answer probe. If the probe separates correct from wrong where the words do not, that is the knowing-vs-saying gap in one controlled experiment. Caveat (from Phase 1B): after-answer states encode the answer itself, so the layer-0 control is reported alongside.

### E3 — Elicitation

Readouts of confidence in the model's own P answer, compared within levels against the probe (ceiling) and B, E (behaviour):

- **A**: stated with the answer (Phase 1, existing).
- **Post-hoc verbal**: the *self / own answer* cell of E2.
- **P(True)**: Kadavath et al.'s prompt verbatim ("Question: … / Proposed Answer: … / Is the proposed answer: (A) True (B) False / The proposed answer is:"), probability on " (A)" renormalised over " (A)" and " (B)", one forward pass (open models, and API models with logprobs). Two placements: *neutral* (the whole prompt in the user turn) and *own* (the model's answer in the assistant turn first). Zero-shot and few-shot, the few-shot examples drawn from the training questions with known labels.
- **User framing** (Sanz-Guerrero et al.'s fix): the *user / own answer* cell of E2.
- **P(True), base model**: few-shot (needed for calibration, per Kadavath et al.), which tests hypothesis 2 against the instruct model's P(True), for discrimination as well as calibration.
- *Optional:* **brainstorm P(True)** (five Phase 1 samples shown first, as in Kadavath et al.), to see whether it adds anything beyond B.

Within-level AUROC leads; ECE and Brier are reported alongside for comparison with Sanz-Guerrero et al.

If one readout closes most of the gap to the probe, that is the usable result: which readout to trust, by kind of question. If none does, the information is there but asking cannot reach it.

### From the sprint talks

Proposed additions after Navaie's talk (2026-11-13; notes in the [sprint brief](epic2_sprint_brief_13-11-2026.md#talk-notes)), to review at the study step:

- **Report the tail as well as AUROC.** AUROC is an average; the safety-relevant number is how often a wrong answer comes with high confidence. For every readout in E2 and E3 (and Phase 1's A, D, B, E for reference): the share of wrong answers with confidence ≥ 0.9, and the share of answers given ≥ 0.9 that are wrong, by kind of question, with bootstrap intervals. For the probe, the same at the threshold that matches the readout's rate of ≥ 0.9 answers, so the comparison is fair. Phase 1 suggests this will be stark for Qwen3-8B, which states ≥ 0.9 almost always.
- **State reliance conditions.** The "which readout to trust" result is stated with its limits: which models (small open Qwen3, thinking off; API models only if run), which kinds of question, which difficulty range, and what would change the conclusion.
- **Say how we tried to break the claim.** A short section in the report: every pre-registered prediction with its outcome, failed ones included; the controls (layer-0 probe, difficulty held fixed, wrong answers by construction, the question-details probe); and the checks that could have overturned the main claim.

## Steps

1. **Study** (Thu 12 Nov). Above. *Gate: approve the plan after the study.*
2. **E1** (Mac, Thu 12 Nov). `analyze_stereotype.py`, table and figure. Free, and a finding to bring to the sprint.
3. **Build** (Mac). New folder `experiments/epic2/phase2_confidence_gap/`: `build_swaps.py` (wrong answers by construction, written to `data/epic2/confidence/swaps.jsonl`), an Inspect task for E2's conversations and P(True) (reusing Phase 1's prompts and grading), and `extract_states.py --answers` to teacher-force given answers. Unit tests on the wrong-answer builders (always wrong, always different from the correct and the own answer, plausible length).
4. **4090 smoke runs** (`--limit`), then the full runs: E2 for three configs (base: P(True) only), E3 P(True) on test and training sets, extraction for E2's answers.
5. **Analysis and figures.** Paired bootstrap CIs; the tail measures (confident wrong answers); a `verify.py` that recomputes every number in the report from the committed tables.
6. **Report.** On the official template, 4 to 8 pages (hard cap) plus references and appendices, including the required Limitations and Dual-Use / Ethical Considerations appendix. Preexisting code is marked in the README (done for `phase1_confidence/`) and in the report. Also: key figure on page 1, related work stating exactly what is new, limitations, LLM usage statement, pre-registered predictions with outcomes, a "how we tried to break it" section and reliance conditions (see *From the sprint talks*), public code and data. State plainly that Phases 1 and 1B were done before the sprint. Then a third lifeandtimes post.

**Stretch, only once E1 to E3 are done:** steering along the probe direction to see whether stated confidence follows (hypothesis 1); a small LoRA on Qwen3-1.7B to state confidence against correctness labels (hypothesis 4).

## Out of scope

- Thinking-mode probing (a cheap transfer check of the existing probes onto thinking states is a candidate if E1 to E3 finish early).
- New question categories or models beyond those above.
- Sycophancy and belief-vs-answer (the earlier Phases 2 and 3), deferred.
