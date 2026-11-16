# Report outline (for the team to write from)

Template: `apart_template.docx` (Word / Google Docs), exported to PDF. Target 4 pages excluding references and appendix (hard cap 8). Abstract 150 to 250 words. Every number below is traced to a file in `../results/`; `verify.py` (to come) recomputes them all. Numbers in [brackets] are 95% bootstrap intervals over questions. E5 numbers added 2026-11-15.

## The one claim

**A language model's confidence in an answer tracks whether the answer is right only when the model checks the answer. Without reasoning first, models do not check an answer that sits in their own turn, so their confidence in it is close to chance, even though their hidden states, and other ways of asking, carry the information.** Checking is switched on by reasoning (Qwen3-8B with thinking, GPT-5 nano) and, for some models, by framing the same answer as the user's (Claude Haiku recomputes it in the reply).

Why it matters (theory of change, one or two sentences in the intro, again in the discussion): stated confidence is used as evidence (by users, by evaluators, by monitors and agents rating their own earlier output). Our result says when that evidence is worth something and when it is not, and which readouts to use instead. It is an instance of Navaie's point that unverified evidence should not carry weight: here the model itself does not verify.

## Title

**Words in the model's mouth**, subtitle *Without reasoning, models do not check an answer in their own turn, so their confidence in it is close to chance* (chosen 2026-11-16; replaces the working title "LLMs are too certain of whatever they said"). The reasoning models are the exception, so the abstract and introduction should say early that the claim is about models that answer without reasoning first.

## Abstract (150 to 250 words; write last)

Five beats, one or two sentences each:

1. Problem: LLM confidence is used as evidence of correctness, but stated confidence often misses what the model knows (prior work, and our preexisting Phase 1 and 1B: probes beat stated confidence on every kind).
2. Method: we give seven models the same question with a correct and a plausible wrong answer, built by construction in the model's own format, placed either in the model's own turn ("your answer") or the user's ("my answer"), across recall, computation and perception; we compare their confidence with P(True), the answer's likelihood and a hidden-state probe on the same answers.
3. Finding: without reasoning, confidence in an answer in the model's own turn barely separates right from wrong (0.50 to 0.55 for Qwen3-8B/1.7B and Grok 4.1 Fast on computation and perception); Grok rates every such answer 1.00. Reasoning models separate under both framings (0.88 to 1.00). Claude Haiku separates the user's answer (computation 0.99) because it recomputes it; where it writes no working, its separation is at chance.
4. Mechanism tests: the report's information is in the state, but steering along the correctness direction does not move it; a small LoRA trained on per-answer correctness connects the own-turn report to it (computation 0.57 to 0.97, at or above the probe), but only for the kinds it was trained on and only for answers the model wrote itself.
5. Implication: do not ask a non-reasoning model how sure it is of its own answer; ask it to check, or use P(True) or a probe.

## 1. Introduction (about 0.6 page with the figure; about 350 words)

Paragraph 1, the problem. Confidence reports are used to decide when to trust a model. Known: verbal confidence is overconfident and poorly discriminating (Kadavath et al. 2022; Lin et al. 2022; Sanz-Guerrero et al. 2026; Zhao et al. 2026). Our own earlier work (preexisting, Phases 1 and 1B, 3 to 12 November): across nine model configurations and 680 questions, what models *do* (sample agreement, token probability) tracks correctness within a difficulty level on every kind, what they *say* tracks it on recall only; hidden-state probes beat stated confidence on every kind. Open question: why?

Paragraph 2, our approach in one sentence each: the swap (correct vs wrong answer by construction, own turn vs user's turn), several readouts on the same answers, and two mechanism tests (steering, fine-tuning).

Paragraph 3, the finding (the one claim above) and the theory of change.

**Figure 1 (page 1, full width): `results/figures/e2_ownership.png`.** Caption: the separation for seven models × three kinds × two framings; 0.5 = none. One-line takeaway in the caption: "Unless it reasons first, a model barely checks an answer placed in its own turn."

Contributions (numbered list, as the template asks):

1. A controlled swap experiment: correct and plausible wrong answers built in each model's own format, on three kinds of question at fixed difficulty, seven models (two open families, three API providers, two reasoning models). Without reasoning, confidence in an answer in the model's own turn is near chance.
2. Evidence that the cause is *checking*, not ownership as such: reasoning removes the gap; Claude Haiku's large ownership effect comes with written recomputation and disappears without it; GPT-4o and Grok, which never write working, mostly lower confidence wholesale for the user's answer.
3. A comparison of readouts on the same answers: the hidden-state probe > P(True) > what the model says; with a familiarity control showing that, on planted answers, much of the probe's advantage is the answer's likelihood.
4. Two mechanism tests: the correctness direction is readable but steering it does not move the report (E4); a small LoRA on per-answer correctness makes the own-turn report as good as the probe, kind by kind, while calibration-only training does not (E5).

## 2. Related work (about 0.4 page; about 250 words)

Three short paragraphs. Cite only what is in the plan's study notes; the starred ones were read only at abstract level, so cite them for their headline claim only.

- **Verbal confidence and P(True).** Kadavath et al. 2022 (P(True); models mostly know what they know); Lin, Hilton & Evans 2022 (fine-tuning verbal confidence on sub-task accuracy: our Lin-style control in E5); Ji et al. 2025 (verbal uncertainty as a linear feature, steered; their reference is sample disagreement, ours is correctness by construction). Probes vs words: Ghosh et al. 2026*, Cacioli 2026*.
- **Answers in the model's mouth.** Sanz-Guerrero et al. 2026 (ownership bias; user framing as a fix; no perception, no difficulty control); Zhao et al. 2026* (closest: injected answers, "confident in what it said, not in what is true"; factual only, no user arm, no P(True), no probe on the same answers); Tsui 2026* (Self-Correction Bench: models fix the user's error, miss their own; reasoning models and "Wait" close it: our reasoning result is the confidence analogue); Khullar et al. 2026* (self-attribution bias in monitors).
- **What is new here** (one sentence each, these are the Impact points): discrimination within difficulty, not calibration; correctness known by construction; three kinds of question; words, P(True) and a probe on identical answers; the reply evidence that separation follows checking.

## 3. Methods (about 1 page; about 550 words plus one small table)

**Preexisting vs new (say this explicitly, the rules require it).** Preexisting (Phases 1 and 1B, 3 to 12 November; git tag `sprint-start`): the 680 questions, grading (code rules plus a Qwen3-14B judge, 114/120 against hand labels), each model's own answers, and the hidden-state probes. New in the sprint: everything below.

**Questions.** From Phase 1: recall (PopQA by page views), computation (multiplication, 2×2 to 4×4 digits), perception (letter count, nth letter, reversal); five difficulty levels each. Test set 150 per kind; Phase 1B's 5,995 training questions for the probe and for E5.

**Models.** Qwen3-8B and Qwen3-1.7B (thinking off), Qwen3-8B-Base (P(True) and probe only), Qwen3-8B with thinking on (4090, vLLM); Claude Haiku 5.5 (API); GPT-4o, Grok 4.1 Fast (non-reasoning), GPT-5 nano (reasoning effort low) on Azure AI Foundry. Settings as in Phase 1.

**The swap (E2).** For each test question, three answers: the model's own Phase 1 answer, the correct answer, and a plausible wrong one, rendered in the model's own format (separators, number words, punctuation). Wrong answers: computation changes a middle digit; perception ±1 count, a neighbouring letter or an adjacent swap; recall another model's wrong answer to the same question, else another question's answer. Every injected answer is graded and grades as built. Two framings: **"your answer"** (the answer sits in the assistant turn; the user asks for the probability it is correct) and **"my answer"** (the user writes "My answer: X"). Reply: `{"confidence": p}`. About 2,600 to 2,700 replies per model, all parsed except 3 thinking replies cut off at the token limit.

**Readouts on the same answers (Qwen).** Verbal confidence (both framings); P(True) as in Kadavath et al. (four variants: neutral or own placement, zero or eight-shot); the answer's log-probability (familiarity); Phase 1B's after-answer probe (logistic, layer chosen on validation; refits reproduce Phase 1B at r ≥ 0.996), with a layer-0 control.

**Metrics.** *Separation*: share of questions where the correct answer gets more confidence than the wrong one (ties half; 0.5 = none). *Within-level AUROC* on the model's own answers (strata = kind × level × task type or relation). 95% intervals from 1,000 bootstrap resamples of questions. Familiarity control: separation within quintile bands of the answer's log-probability.

**Mechanism tests.** E4: a ridge readout of each report from the answer's state (out-of-fold, grouped by question), and steering: the probe's direction (or a random one of the same size) added after the probe's layer, ±2 to ±8 SDs of the probe axis and ±0.25 to ±0.5 of the state norm. E5: LoRA (rank 16) on Qwen3-1.7B, training the "your answer" report on whether each training answer was right; controls: Lin-style targets (the stratum's accuracy) and shuffled labels; leave-one-kind-out for transfer.

**Pre-registration.** Every prediction was written in the plan before its run; outcomes are in Appendix A (supported or not).

Optional small Table 1: models × settings × number of replies (could go to the appendix if space is short).

## 4. Results (about 1.5 pages)

### 4.1 Without reasoning, a model does not check an answer in its own turn (Figure 1)

- Qwen3-8B "your answer": 0.55 [0.51, 0.60] recall, 0.51 [0.50, 0.52] computation, 0.50 [0.50, 0.51] perception; it gives 0.9 or more to every planted wrong answer (100%, all three kinds). Qwen3-1.7B 0.51 to 0.52. (`e2_separation.csv`, `e2_means.csv`)
- Grok 4.1 Fast: 1.00 to every computation and perception answer in its own turn (separation exactly 0.50); GPT-4o rates planted wrong products 0.97 on average (97% at ≥ 0.9), separation 0.53 [0.51, 0.56].
- Recall is the exception for the stronger models: GPT-4o 0.93 [0.90, 0.96] in its own turn; a fact can be checked by recalling it.

### 4.2 Checking, not ownership

- Reasoning removes the gap: Qwen3-8B with thinking 1.00 computation, 0.88 [0.84, 0.91] perception in its own turn (0.51 and 0.50 without); GPT-5 nano 0.98 and 0.95; under "my answer" both are equal or within the intervals.
- Claude Haiku: "your answer" 0.58 [0.52, 0.63] / 0.60 [0.55, 0.64] / 0.79 [0.73, 0.85] (computation / perception / recall); "my answer" 0.99 [0.97, 1.00] / 0.86 [0.81, 0.90] / 0.85 [0.80, 0.91]. Under "my answer" it writes out working in 97% of computation replies (18% under "your answer"). With working for both answers of a pair, separation 1.00 (144 pairs); with none, 0.45 for computation answers in its own turn (107 pairs). (`e2_working.csv`, `e2_working_haiku.csv`)
- GPT-4o and Grok never write working. "My answer" lowers their confidence in a planted wrong product by 0.77 [0.70, 0.83] and 0.63 [0.55, 0.71], but in the correct product too (by 0.52), so separation rises only to 0.63 and 0.57.
- Qwen without thinking: "my answer" 0.64 / 0.67 / 0.52 (Qwen3-8B recall / computation / perception): Sanz-Guerrero et al.'s ownership bias replicates, and here it costs discrimination, not only calibration.

### 4.3 The information is there: other readouts on the same answers

- On the model's own answers (Qwen; ~5,700 training questions, out-of-fold): probe 0.86 to 0.91 on recall and computation, 0.70 to 0.72 on perception; best P(True) variant 0.70 to 0.81 for Qwen3-8B (0.02 to 0.10 below the probe) and 0.59 to 0.76 for Qwen3-1.7B; verbal (test set) 0.49 to 0.68, 0.20 to 0.32 below the probe, intervals excluding 0 in five of six cells (Qwen3-1.7B recall, with 15 right answers, reaches 0). (`e3_auroc.csv`; Figure 2 optional: `figures/e3_readouts.png`)
- On planted answers, the probe separates computation at 0.95 (8B), but the answer's likelihood does almost as well (0.93): within likelihood bands the probe keeps 0.80 (8B) on computation and drops to 0.53 to 0.58 on recall and perception. Say this plainly: much of "the state knows it is wrong" on planted answers is "the state knows it would not have said this". (How we tried to break our own claim.)
- The confident tail: Qwen3-8B states ≥ 0.9 on every test answer, and 60% to 76% of them are wrong; the probe's most confident quarter has the fewest errors in every cell (training computation: 3% vs 66% overall). (`tail.csv`, `tail_quarter.csv`)

### 4.4 Mechanism tests

- E4 readable but not used: "my answer" confidence can be read from the answer's state (Spearman 0.46 to 0.66 on recall and computation) and follows the probe (0.38 to 0.51) (`e4_directions.csv`); steering along the probe's direction moves neither report more than a random direction of the same size (Qwen3-8B: no shift above 0.02 at up to half the state norm; Qwen3-1.7B: inconsistent sign, matched by random pushes, or collapse to one constant reply) (`e4_steer_summary.csv`). One sentence, figure to the appendix.
- E5 (Qwen3-1.7B; `e5_summary.csv`, `e5_transfer.csv`; Figure 3 `figures/e5_lora.png`): training the "your answer" report on whether each training answer was right lifts own-answer within-level AUROC from 0.57 to 0.97 [0.93, 1.00] on computation (probe 0.90) and from 0.58 to 0.76 [0.63, 0.88] on perception (probe 0.80). Controls: Lin-style targets calibrate the report (ECE 0.24 to 0.03 on computation) but barely change discrimination (0.67, 0.60); shuffled labels give nothing (0.57). The gain is kind-specific (held out: computation 0.74, 42% of the gain; perception 0.58, none) and does not teach checking: separation of injected answers in its own turn only 0.59 to 0.69, far below the reasoning models. Read: the default report fails not for lack of information or capacity but because the model is not trained to use what it has; training gives a reading of its own answers, not a check.


## 5. Discussion and limitations (about 0.5 page; about 300 words)

- **What it means.** The model has the information (probe, P(True)); what fails is the act of reporting on its own answer without checking it. A confidence report is only as good as the check behind it. Practical rule: for non-reasoning models, do not ask "how sure are you of your answer"; ask it to verify (reasoning), or use P(True) on the answer framed neutrally, or a probe; never trust a stated 0.9 (Qwen3-8B: 60 to 76% of those are wrong).
- **For monitoring and agents.** An agent or monitor rating content in its own earlier turn is in the "your answer" condition (cf. Khullar et al.). One sentence.
- **Limitations** (the main ones here; the rest in Appendix B): planted answers are off-policy text, so the probe was reported on own and planted answers separately (Kirch et al.); with reasoning, confidence becomes verification, not introspection (we do not claim the reasoning models introspect better); one run per model, one prompt wording, 150 questions per kind; the Foundry and Haiku models differ in many ways besides reasoning; steering tested one layer and one direction per kind; E5 on one small model.
- **Future work.** Does a checking instruction ("verify, then rate") close the gap without reasoning? Do agents' self-evaluations in multi-turn settings show the same blind spot?

## 6. Conclusion (3 sentences)

Restate the claim, the evidence in one line, the rule for practitioners.

## Code and data

Three links (repo is public):

- Sprint work: https://github.com/anilnairxyz/synapses/tree/sprint-submission/experiments/epic2/phase2_confidence_gap (its README explains the files, how to rerun, and what is reused from Phase 1).
- Preexisting work (Phases 1 and 1B, 3 to 12 November): https://github.com/anilnairxyz/synapses/tree/sprint-submission/experiments/epic2/phase1_confidence
- Exactly what changed during the sprint: https://github.com/anilnairxyz/synapses/compare/sprint-start...sprint-submission

`sprint-submission` is a tag to create just before submitting (I will do it), so the links show the code as submitted even if the repo changes later. Say in one sentence that `verify.py` recomputes every number in the report.

## Author contributions (optional)

One line if solo.

## References

Consistent style (author, year, title, venue, arXiv link). Only papers we cite: Kadavath et al. 2022; Lin, Hilton & Evans 2022; Ji et al. 2025; Sanz-Guerrero, Mager & von der Wense 2026; Zhao et al. 2026; Tsui 2026; Khullar et al. 2026; Kirch et al. 2025; Ghosh et al. 2026 and Cacioli 2026 if cited; Navaie 2026 (Artificial Intelligence Review, doi 10.1007/s10462-026-11666-8). I will assemble the list with links from the plan's study notes.

## Appendix (does not count towards the page limit)

- **A. Pre-registered predictions and outcomes**: the table from `results/README.md`, plus E2b, E4, E5 rows.
- **B. Limitations and dual-use / ethical considerations (required).** Draft by me, for you to edit. Points: what the result does and does not establish (behavioural for API models; mechanistic tests on Qwen only; verification vs introspection); misuse: the ownership effect could be exploited by putting false content in a model's own turn (prefill) to have it vouch for it, and E5-style training could make reports *look* calibrated without improving discrimination (which is why we report discrimination); personal data: none (public questions, synthetic arithmetic and words).
- **C. Further figures**: `e2_separation.png` (all readouts), `e2_means.png`, `e4_steer.png`, E1 (`e1_stereotype.png`) with a two-line summary: stated confidence follows the state coarsely; on computation it follows number size as much as correctness.
- **D. LLM usage statement** (required by the template). Draft by me: Claude Code (Claude Opus 5.5) wrote and ran the analysis code, made the figures, searched the literature (every cited paper checked against its arXiv page) and drafted this outline, the appendix and the statement; the authors set the questions and hypotheses, decided every design change, ran the GPU and Foundry jobs, and wrote the report text; every number was checked by `verify.py`.

## What stays out of the main text (space)

E1 in detail; the P(True) variant comparison; calibration (ECE); the base model beyond one sentence; per-level tables. All in the results README and the repo.
