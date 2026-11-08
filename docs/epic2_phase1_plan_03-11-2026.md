# Epic 2 · Phase 1 — Confidence Audit (Detailed Plan)

Detail for **Phase 1** of [`epic2_plan_03-11-2026.md`](epic2_plan_03-11-2026.md). Budget: a few days, followed straight away by Phase 1B (internal-state probe on the 4090).

> **Status (2026-11-07): Step 1 done except post-cutoff; model line-up and budget agreed; Step 2 (task) built, pilot next on the 4090.**

## Working framing (in progress)

**Question.** Does a model's self-knowledge differ by *what it takes to answer the question*, and does each failure mechanism leave a distinct pattern in the confidence results? We approach this from the test side: categorise questions by what answering demands, then grade calibration and discrimination per category rather than as one overall score. A single score can hide errors that cancel out, e.g. a model that is overconfident on long-tail facts and underconfident on arithmetic can look calibrated on average.

**Benchmarks carry a theory of what they test.** Current LLM confidence benchmarks are built around known LLM failure modes (long-tail facts, tokenisation blind spots, unknowables). So they mostly measure *known* weaknesses, and they go stale once models are trained on those weaknesses. Human psychometrics starts from a theory of cognition; ARC-AGI starts from a theory of intelligence (efficient skill acquisition).

**Categories independent of the model, mechanisms specific to it.** The categories should come from what *any* agent needs to answer, so they last and allow a human baseline:

- recall of stored information (from common to long-tail)
- computation (multi-step arithmetic or logic)
- perception / representation (e.g. character-level questions)
- information beyond the agent's horizon (events after the training cutoff that have already happened)
- no answer exists, which splits three ways because the right behaviour differs:

| Sub-type | Example | Right behaviour |
| --- | --- | --- |
| Ill-posed | Underspecified, false premise | Abstain or ask for clarification |
| Forecastable | "Will X happen by June?" | Give a calibrated probability; abstaining is *wrong* |
| Unknowable in principle | "What will vermouth taste like in 2150?" | Abstain |

LLM-specific mechanisms, mostly from Kalai et al., are then *predictions for each category*: singleton rate for long-tail recall, model-class limits for perception, errors snowballing during generation for computation, distribution shift for the post-cutoff horizon, binary-grading incentives against abstention for the ill-posed and unknowable sub-types. Kalai's "I don't know" incentive does *not* apply to the forecastable sub-type: there, a scoring rule that rewards honest probabilities (Brier) is the right tool. Open issues:

- difficulty within a category, so differences between categories aren't just differences in difficulty
- who labels what a question demands (building questions so their demand is known by construction, e.g. arithmetic with controlled digits, PopQA popularity, templated unknowables, is cleaner than labelling by hand)
- questions with more than one demand

**Confidence about itself vs confidence about the world.** Two different constructs that are easy to mix up. *Self-knowledge* is P(my answer is correct): uncertainty about the model itself, which comes from what it doesn't know (recall, computation, perception categories). *Forecasting* is P(event happens): uncertainty about the world (the forecastable sub-type, ForecastBench). A model can be excellent at one and poor at the other, so results should always say which is being measured. The main question of this phase is self-knowledge.

**Measure self-knowledge separately from accuracy (meta-d′).** AUROC and Brier move with accuracy, so they can't tell poor self-knowledge from a category that is simply harder. Borrow **meta-d′** from human metacognition research (Maniscalco & Lau, 2012): with signal detection theory it estimates how well confidence tracks correctness independent of task performance. **meta-d′/d′** (metacognitive efficiency) is comparable across categories with different accuracy. Also test the **hard–easy effect** (Lichtenstein & Fischhoff): overconfident on hard items, underconfident on easy ones, checked per category.

**Ways of eliciting confidence, compared on the same questions.** Each method assumes a theory of where confidence lives:

| Method | Human version | LLM version | Assumes confidence lives in |
| --- | --- | --- | --- |
| Verbal report | Confidence rating | Stated probability (ConfidenceBench) | Self-report: the model can put it into words |
| Betting | Post-decision wagering | Threshold in the prompt (Kalai's behavioural calibration) | Behaviour under stakes |
| Implicit | Response time, consistency | Token probabilities, agreement across samples | The output distribution |
| Before vs after | Feeling-of-knowing vs confidence judgment | Confidence asked before vs after the answer | Whether confidence comes before or after the answer |
| Internal state *(open weights only)* | Neural measures of confidence (EEG/fMRI) | Linear probe on hidden states, before or after generation (Ni et al.) | The representation: the model may "know" more than it says |

**Gap vs ConfidenceBench** (from the ChatGPT reading list, not the sprint brief). It reports no per-category results and gives no theoretical justification for its four categories (spatial reasoning, high-precision maths, word lookup, unknowable). Per-category analysis, theory-grounded categories and a measure that holds up across accuracy levels are what would set this work apart. The approach is to cover the ground ourselves first and compare against more rigorous team-backed benchmarks later.

**Extension: probing internal states (Ni et al.; needs open weights on the 4090).** Ni et al. probe LLM hidden states and find a usable signal about what the model knows *before* generation, which is refined after it. Their tests (NQ, HotpotQA, MMLU) are almost all recall. Closed API models give no internal access, but open-weights models of 7–9B parameters fit on the 4090 in bf16, so this is a GPU handoff, not out of reach. It is also the same tool as Epic 1: a linear probe on frozen representations (`synapses.metrics.linear_probe`), with TransformerLens already in the `interp` extra. This is where the phase meets mechanistic interpretability.

- **Comparing across models without a transferable probe.** A probe is specific to one model and one layer, so it can't be carried across models. Instead, standardise the *protocol*: the same questions, train/test split, probe type and layer sweep, with one probe per model, judged by a score that applies to any model (AUROC, meta-d′). That makes "how much each model knows internally about what it knows" comparable.
- **Internal signal vs the model's outputs.** For each open model, compare the probe's confidence with its stated confidence and its agreement across samples on the same questions. This measures Kalai's latent-vs-output point directly: the knowledge is in the representation, and the output overrides it.
- **Base vs instruct pairs.** Open families release both. Measuring the probe signal and the stated calibration on both tests whether post-training damages calibration in the representation or only in the output. Closed APIs can't support this.
- **Closed models: behavioural proxies only.** Token log-probabilities (OpenAI exposes them; as far as we know Anthropic does not) and C3-style consistency across rephrasings (check whether C3 itself needs internal states or only outputs; if only outputs, it carries over to API models as-is).
- **Prediction by category (the key test).** A probe before generation should work well on **recall** (whether the fact is stored is settled before the model writes anything) and much worse on **computation** (whether the answer is right is decided *during* generation, so the signal can only appear after it). **No answer exists** should be where it flags uncertainty most cleanly. If that holds, it shows *when* a model can know what it knows, which neither behavioural evals nor recall-only probing can show alone.

**ForecastBench takeaways** (leaderboard data from `forecastingresearch/forecastbench-datasets`, updated 2026-11-02; scores are the difficulty-adjusted Brier Index, higher is better):

- *Tournament* (teams may add tools, retrieval, context, ensembling): eight entries sit above the superforecaster median (68.4), three of them Google DeepMind (best 68.9). None of the differences is significant (overlapping CIs; p = 0.5–0.9 for "superforecasters better"). The lead is mostly on dataset questions; on market questions superforecasters (74.2) still beat most entries. This is parity, not superiority.
- *Baseline* (models out of the box): superforecasters 67.3 vs best model 61.3, still significantly ahead (p < 0.001); frontier models are roughly level with the general public.
- Caveats: superforecasters were surveyed in July 2024 on a different, smaller question set; the comparison relies on a difficulty adjustment.
- **Lesson 1: aggregation is where the gain comes from.** The 6–8 point gap between baseline and tournament comes from the scaffolding: retrieval, ensembling, aggregating forecasts. That supports agreement across samples as a confidence signal.
- **Lesson 2: contamination-free by construction.** Ask questions whose answers didn't exist at test time. Waiting for resolution is impractical for a one-day phase; questions about events after the training cutoff give the same protection now.
- Forecasting measures confidence about the world, not self-knowledge (see above), so ForecastBench is a neighbour of this phase, not its core.

**Reading complete** (2026-11-05): Kalai et al., ConfidenceBench, Ni et al., ForecastBench.

**Phase question:** can a model reliably tell the cases where its own answer is solid from the cases where it is on shaky ground? This is stronger than accuracy, and stronger than calibration alone. A model that is 80% accurate and says "95%" on every answer is usable as an answerer but useless epistemically. So we measure two separate things: **calibration** (does $P(\text{correct}\mid p=0.8)\approx 0.8$?) and **discrimination** (is confidence higher on the answers that turn out right?). Brier score mixes the two; AUROC isolates the second, which is the one the phase question is really about.

## Background reading (verified 2026-11-03)

*Provenance: the Apart Track 1 description names only "ForecastBench / Confidence Audit". The four papers below came from a ChatGPT-drafted reading list; I checked each link and its abstract, not the full papers. None of them is endorsed by the sprint. What Apart means by "Confidence Audit" is still to be established and should take precedence once known.*

- **ForecastBench**, Karger et al. 2024, [arXiv 2409.19839](https://arxiv.org/abs/2409.19839). Questions about future events, so the answers don't exist yet and there's no contamination. Useful for its evaluation machinery; far too broad to reproduce in a day.
- **Why Language Models Hallucinate**, Kalai et al. 2025, [arXiv 2509.04664](https://arxiv.org/abs/2509.04664). Argues that binary-graded benchmarks reward guessing over abstaining, and proposes scoring with an explicit confidence threshold so that a wrong answer costs more than an abstention.
- **ConfidenceBench**, ffrench-Constant et al. 2026, [arXiv 2607.20526](https://arxiv.org/abs/2607.20526). Prompted verbal confidence across 15 frontier models, scored with Brier. Its 200 multiple-choice questions span four categories, one of them unknowable; the set is private. Best Brier is 0.103 against a calibrated-random baseline of 0.1875, and the most accurate model is not the best calibrated. *Not referenced by Apart; probably suggested for its name's similarity to "Confidence Audit".* On reading (2026-11-04): no per-category results, no theoretical justification for its categories, and lightweight in rigour. Useful as a point of comparison, not as a reference design.
- **Exploiting LLM Internal States for Knowledge Boundary Perception**, Ni et al., ACL 2025, [aclanthology 2025.acl-long.1184](https://aclanthology.org/2025.acl-long.1184/). Internal states carry a confidence signal *before* generation. Its C3 method scores confidence by how consistent the answers are across rephrasings of the question. This paper is the bridge back to Epic 1's probing and mech-interp direction.

## Scope (agreed 2026-11-06)

**Phase question:** does a model's self-knowledge differ by what it takes to answer the question, once difficulty is controlled for, and does each way of eliciting confidence tell the same story?

### Categories (all eight)

Answerable categories carry a right/wrong answer, so they get calibration and discrimination metrics. Each has a difficulty knob so categories can be compared at matched accuracy and the hard–easy effect can be checked.

| Category | Source (candidate) | Difficulty knob | Prediction (mechanism) |
| --- | --- | --- | --- |
| Recall | PopQA | Entity popularity (Wikipedia pageviews) | Accuracy falls with obscurity; does confidence follow? (singletons) |
| Computation | Generated arithmetic | Digits / number of steps | Confidence stays high as accuracy falls (errors snowball during generation) |
| Perception / representation | Generated character tasks (letter counts, reversal, nth letter) | Word length / rarity | Blind spot the model may not know it has (tokenisation) |
| Multi-hop (**deferred 2026-11-06**, see below) | Composed two-fact questions (e.g. from PopQA / Wikidata relations), or an existing multi-hop set | Number of hops | Each fact known, composition fails; does confidence reflect the weak link? |

Categories without a right answer carry a *right behaviour* instead, so they are scored on behaviour (abstention, clarification, calibrated probability), and reported separately.

| Category | Source (candidate) | Right behaviour | Scored by |
| --- | --- | --- | --- |
| After the training cutoff | Questions about events between the models' cutoffs and now | "That's after my knowledge ends" | Abstention rate; confidence on committed answers |
| Ill-posed | Built from PopQA (2026-11-06): false premises (a famous work with the wrong creator swapped in, 5 popularity levels) and ambiguous titles (several works share a title, comparable popularity) | Correct the premise / clarify | Premise-correction and clarification rates |
| Unknowable | Built 2026-11-06: unrecorded details about famous people (templated), curated open problems, far-future specifics (templated). KUQ / SelfAware dropped: mostly opinion, speculation or "would you rather" | Abstain | Abstention rate; confidence on committed answers |
| Forecastable | ForecastBench question sets resolved *after* every model's cutoff (unresolved as far as the model knows, resolved for us). Built 2026-11-06: 75 resolved market questions (Polymarket 35, Metaculus 20, Manifold 20) asked as of their late-June to September 2026 freeze dates, resolving July to November 2026; crowd probability 5–95%, one question per family of near-duplicates; crowd forecast kept as a human baseline (Brier 0.19 vs 0.23 for the base rate) | Calibrated probability | Brier on P(event). This is confidence about the world, so it is reported apart from self-knowledge |

**Multi-hop is deferred (2026-11-06).** No existing set fits closed-book use. MuSiQue's steps are often reading-comprehension lookups that only make sense with the passage, some are time-dependent, and some gold answers are wrong. MQuAKE's long chains converge on a few guessable answers (19 distinct answers across 729 four-hop items) and lean on time-sensitive facts (CEOs, heads of government, spouses); after filtering, no four-hop items are left. The fix that would work is our own Wikidata chains through stable relations that end at a specific entity, with the same starting entity asked at 1–4 hops (popularity matched by construction, and the single hops included so "each fact known, composition fails" can be tested directly). That's about a day of work, set aside for now.

Sizes (initial): about 150 per answerable category (spread across 5 difficulty levels) and about 75 per behaviour category; about 750 questions in all with multi-hop deferred. Generated categories are contamination-free by construction; cutoff-sensitive categories need each model's cutoff recorded.

### Elicitation methods (a–d, plus e where available)

| | Method | How | Gives |
| --- | --- | --- | --- |
| a | Stated probability (after) | Answer, then P(correct) | One number per question |
| b | Agreement across samples | about 10 samples at T=1, share agreeing with the main answer | Behavioural confidence; 10× cost |
| c | Threshold in the prompt (Kalai) | Scoring rule stated, t ∈ {0, 0.5, 0.75, 0.9} | Answer/abstain per t, so an implied confidence range per question |
| d | Stated probability (before) | P(will answer correctly) before answering | "Feeling of knowing"; behavioural twin of the 1B probe |
| e | Token probabilities | Log-probability of the answer tokens | Only where the provider exposes them (OpenAI-family, local models) |

The interesting results are where methods disagree on the same question.

### Measures

Per category and per difficulty level: accuracy, Brier (+ Murphy decomposition), ECE, AUROC, ~~**meta-d′/d′**~~, abstention rate. **meta-d′ dropped (2026-11-08):** it needs a two-choice task, where type-1 d′ comes from which option was picked; our questions are open-ended, so there is no d′ (recall accuracy of 0.10 would give a negative one). Accuracy is controlled instead by comparing AUROC across categories at matched accuracy (difficulty levels give the spread) and by a regression, correct ~ confidence × category + level, per method and model. AUROC is reported only with at least 10 right and 10 wrong answers, with bootstrap confidence intervals; method e is summarised by the sum of answer-token logprobs. Plots: reliability per category, accuracy vs confidence across difficulty (hard–easy effect), stated vs implied (method c) confidence, before vs after.

### Models (agreed 2026-11-07)

The phase tests our methods, not the models, so each model is there for what it exercises. Budget: Foundry about $50, personal Claude about $5, personal OpenAI about $10 (kept in reserve); the 4090 costs time only. Connection checks: `experiments/epic2/phase1_confidence/checks/` (4090) and the office repo's `outputs/check_models_2026-11-07.txt` (Foundry).

| Model | Where | What it tests |
| --- | --- | --- |
| Qwen3-8B, thinking off | 4090 | anchor: every method including e; the Phase 1B model |
| Qwen3-8B, thinking on | 4090 | reasoning on vs off, same weights (main calls at T=0.6, Qwen's advice against greedy thinking) |
| Qwen3-1.7B, Qwen3-14B-FP8, thinking off | 4090 | size and accuracy spread within one family |
| Qwen3-8B-Base (P and B only, 5-shot) | 4090 | base vs instruct calibration of token probabilities (Kalai's prediction) |
| grok-4-1-fast-non-reasoning | Foundry | another vendor, with logprobs |
| gpt-5-nano | Foundry | closed reasoning model, no logprobs |
| ~~Phi-4-mini-instruct~~ | Foundry | dropped after the pilot (2026-11-08): too slow on Foundry (4 of 60 questions in 11 minutes) |
| gpt-4o | Foundry | strong OpenAI non-reasoning, with logprobs (replaced gpt-4.1-mini on 2026-11-08: real cost is about $4.70 a run, and the judge is now Qwen3-14B on the 4090, validated on a hand-labelled gold set) |
| Claude Haiku 5.5, low effort | personal | no temperature control, no logprobs |

Dropped for cost: Claude Sonnet. (gpt-4o was dropped on a $15 estimate, then brought back once the 4090 runs gave real token counts.) Full-run cost per model: about 12k calls; Foundry models $0.40–3, judge about $0.50 per model. Costs are re-estimated from the pilot's real token counts before any full run; Azure budget alerts at $25/$40/$50.

Cutoffs matter for forecastable (a model must not have seen the outcome: filter by `resolved_on`) and post-cutoff (to design).

### Step 2 design (agreed 2026-11-07)

One Inspect task (`confidence_task.py`); every method on every question, each call a fresh conversation, so methods line up per question: P (answer only, logprobs: method e), B ×10 at T=1 (method b, agreement with P), A (answer then confidence), D (confidence field first), C at t ∈ {0, 0.5, 0.75, 0.9} (Kalai rule). Forecastable: P(yes) with background, criteria and as-of date, plus 10 samples. Main calls at T=0, samples at T=1. Every prompt carries the same neutral line allowing the model to say a question can't or shouldn't be answered. Base model: 5-shot, answerable categories, P and B only.

Grading: exact for computation and perception; alias match then judge for recall; judge labels for the behaviour categories (corrects premise / clarifies / abstains vs answers or commits); Brier for forecasts. The judge runs as a separate `inspect score` pass, so generation can run where the judge isn't reachable. Judge choice: Qwen3-14B (4090, free) vs gpt-4.1-mini (Foundry), both run on the pilot and kept by agreement with about 30 hand-checked decisions per judged category.

---

## Steps

1. **Question set.** Build the categories (seven, with multi-hop deferred) with difficulty levels; record sources, gold answers / right behaviours, and difficulty level per item. Sanity-read a sample of every category and level.
2. **Elicitation + grading.** Prompts for methods a–d (e where available); grading = exact match for generated categories, string match then LLM judge for recall, judge for behaviour categories (abstain / clarify / correct premise / commit). Hand-check about 30 judge decisions per judged category.
3. **Metrics.** Extend `synapses.calibration` (bootstrap intervals, accuracy-matched comparison; meta-d′ dropped, see Measures) with tests; per-category / per-level analysis; rewrite `analyze.py`. Analysis of the existing logs only, no model calls. **Done 2026-11-08:** `analyze.py` + `plots.py`; tables, figures and findings in `experiments/epic2/phase1_confidence/results/` (README there).
4. **Pilot.** One cheap model, about 10 questions per category, every method: check parsing, judge quality, difficulty ranges (each answerable category should span from high to low accuracy; adjust knobs if not).
5. **Full runs.** The model line-up across tiers.
6. **Close-out.** `experiments/epic2/phase1_confidence/RESULTS.md`: per-category tables, where methods disagree, hard–easy effect, abstention behaviour, the forecasting result kept apart; a go/no-go for the sprint; a full "Inside …" post on lifeandtimes, in the style of the Epic 1 posts (inside-a-vae, inside-simclr, …), so Step 3's plots are made to blog quality.

**Phase 1B (straight after):** linear probes on the open models' hidden states before and after generation, on the same question set; the key test is recall vs computation.

## Out of scope for Phase 1

- Fine-tuning for calibration.
- Paraphrase-stability sweeps (C3-style) unless 1B needs them.
- Reasoning on vs off, except where a local model offers it for free.
