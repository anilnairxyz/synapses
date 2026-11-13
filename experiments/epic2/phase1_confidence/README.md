# Epic 2 · Phase 1 — Confidence Audit

> **Preexisting work (AI x Epistemics sprint disclosure).** All code, data and results in this folder (Phases 1 and 1B) were written between 3 and 12 November 2026, before the sprint began on 13 November. Work done during the sprint lives in `experiments/epic2/phase2_confidence_gap/`.

Can a model tell when its own answer is solid from when it is on shaky ground? Five ways of eliciting confidence (stated after, sample agreement, Kalai threshold, stated before, token probabilities) compared on the same questions, across seven question categories with difficulty levels; calibration, discrimination and abstention per category.

Plan: [`docs/epic2_phase1_plan_03-11-2026.md`](../../../docs/epic2_phase1_plan_03-11-2026.md).

## Files

- `questions/`, `build_questions.py`: Step 1. Builds `data/epic2/confidence/questions.jsonl` (recall, computation, perception, ill-posed, unknowable, forecastable; post-cutoff to come). The file is committed with `git add -f` (`data/` is ignored) so every machine runs the same questions.
- `check_models.py`: connection check per model (connect, JSON, sampling, logprobs). Results kept in `checks/`.
- `confidence_task.py`: Step 2. The Inspect task: every method on every question (calls P, B×k, A, D, C×4; forecastable F + FB×k), deterministic grading in the run, judge grading in a second pass. Its docstring lists the options.
- `chat_template_raw.jinja`: pass-through chat template so base models can be served by vLLM and prompted few-shot.
- `run_4090.sh`: the 4090 driver: per configuration, start vLLM, run the task, stop vLLM. Logs in `runs/4090/<mode>/<config>/`.
- `analyze.py`: Step 3, analysis of the judged full-run logs (no model calls). `load()` builds one row per (config, question, method), cached in `results/rows.parquet` (not committed; `--refresh` rebuilds it). Commands: `check`, `discrimination`, `difficulty`, `hard-easy`, `methods`, `behaviour`, `forecasts`, or `all` (every table, about 3 minutes). Its docstring defines every method row and analysis. Tests in `tests/test_phase1_analyze.py`.
- `plots.py`: Step 3 figures (blog quality) from the tables, into `results/figures/`.
- `results/`: tables, figures and the Step 3 findings summary ([`results/README.md`](results/README.md)).
- [`RESULTS.md`](RESULTS.md): Step 6 close-out (findings, caveats, go/no-go). Blog post: lifeandtimes `docs/blog/posts/2026-11-08-inside-a-confidence-audit.md` (figures copied there as `docs/blog/images/confidence_*.png`).
- Metrics live in `src/synapses/calibration.py` (tested in `tests/test_calibration.py`); task tests in `tests/test_phase1_task.py`.

## Running

**4090 (WSL, repo root):**

```bash
experiments/epic2/phase1_confidence/run_4090.sh pilot        # 10 questions per category, every config
experiments/epic2/phase1_confidence/run_4090.sh full         # overnight
```

**Judge pass on the 4090:** `experiments/epic2/phase1_confidence/judge_4090.sh [full|pilot|full/<config>]` serves Qwen3-14B-FP8 (thinking off; `THINK=1` for on) and re-scores every log in place.

**Judge validation:** `checks/gold_set.jsonl` holds 120 hand-labelled responses (drafted by Claude, reviewed by Anil, 2026-11-08; built by `make_goldset.py`, readable in `checks/gold_set_review.md`). `check_judge.py` runs the code rules and the judge on it and reports agreement; `judge_4090.sh check` does this on the 4090 with thinking off and on, saving results in `checks/judge_check_*.txt`. Re-run the check whenever a judge prompt or rule changes. Held-out checks (labelled blind, judge label hidden): `checks/heldout_set.jsonl`, 40 decisions on the Qwen runs, 39/40; `checks/heldout_api_set.jsonl`, 48 decisions on the Claude and Foundry runs (3 per model per kind, `make_goldset.py --heldout 12 --balance --runs claude/full foundry/full --out ...`), 43/48. Every API disagreement was the judge being stricter than the hand label (4 of 5 borderline, all on recall but one); the one clear error ("Question is ambiguous" on recall → INCORRECT) is now a code rule (ABSTAINS).

**Judge pass, by hand** (anywhere the grader model is reachable; overwrites the scores in the log):

```bash
uv run --extra evals inspect score <log.eval> \
    --scorer experiments/epic2/phase1_confidence/confidence_task.py@confidence_scorer \
    -S judge=true --model-role grader=<model> --action overwrite --overwrite
```

**Foundry (office):** `run_foundry.sh`, `confidence_task.py` and `questions.jsonl` are copied into the office repo's `confidence/` folder (the synapses copies are the source; re-copy after changes). There: `confidence/run_foundry.sh pilot|full [configs]` runs grok-4-1-fast, gpt-4o and gpt-5-nano (generation only; phi-4-mini was dropped after the pilot as too slow on Foundry) into `outputs/confidence/<mode>/<config>/`. The logs are copied into `runs/foundry/<mode>/<config>/` here and judged on the 4090 with `judge_4090.sh foundry`, so every model has the same validated judge.

**Dry run (Mac):** `uv run --extra evals inspect eval experiments/epic2/phase1_confidence/confidence_task.py --model mockllm/model -T per_category=2 -T k_samples=2`.

## Phase 1B: internal-state probes

Plan: [`docs/epic2_phase1b_plan_08-11-2026.md`](../../../docs/epic2_phase1b_plan_08-11-2026.md). The Phase 1 questions are the test set; the probes train on a separate set.

- `build_train_questions.py`: Step 2. Builds `data/epic2/confidence/train_questions.jsonl` (5,995 answerable questions from the same generators, no overlap with the test set; committed with `git add -f`).
- `run_4090.sh train`: Step 3. Plain answer (P) only on the training set for qwen3-8b, qwen3-1.7b and qwen3-8b-base; logs in `runs/4090/train/<config>/`. Then `judge_4090.sh 4090/train`.
- `extract_states.py`: Step 4. Hidden states at four positions (end of question, end of prompt, last answer token, mean over the answer) at every layer, from one forward pass on prompt + logged answer. Writes `states/<config>/<split>.npy|.parquet` (gitignored; about 8 GB per model). `--limit 20` is the smoke test (prints the positions and checks HF greedy answers against the logged ones, saves nothing).
- `probe.py`: Step 5, on the PC (CPU). Per config, position and training kind (recall, computation, perception, pooled): logistic-regression probe at every layer, layer chosen on a 20% validation split of the training set, C by 5-fold CV, then scores for every test question; controls (shuffled labels, metadata-only probe) and, with `--mlp`, Ni et al.'s MLP. Writes `results/probe_layers.csv` and `results/probe_scores.parquet` (small; commit them). Tests: `tests/test_phase1b_probe.py` (synthetic states).

- `probe.py --oof`: out-of-fold probe scores on the training set at the chosen layers (`results/probe_oof.parquet`), for the probe-only comparisons.
- `analyze_probes.py`: Step 6, on the Mac (no states needed): `results/probe_auroc.csv`, `probe_gaps.csv`, `probe_vs_methods.csv`, `probe_transfer.csv`, `probe_abstain.csv`. About 5 minutes (bootstrap).
- `plots_probes.py`: Phase 1B figures, `results/figures/probe_*.png`.
- [`RESULTS_1B.md`](RESULTS_1B.md): Step 7 close-out. Blog post (draft): lifeandtimes `docs/blog/posts/2026-11-08-probing-confidence.md` (figures as `docs/blog/images/confidence_probe_*.png`).

**4090 order (WSL, repo root):**

```bash
uv run --extra evals --extra interp python experiments/epic2/phase1_confidence/extract_states.py qwen3-8b --splits test --limit 20   # smoke test
experiments/epic2/phase1_confidence/run_4090.sh train
experiments/epic2/phase1_confidence/judge_4090.sh 4090/train
uv run --extra evals --extra interp python experiments/epic2/phase1_confidence/extract_states.py qwen3-8b qwen3-1.7b qwen3-8b-base
uv run --extra evals python experiments/epic2/phase1_confidence/probe.py qwen3-8b qwen3-1.7b qwen3-8b-base --mlp
uv run --extra evals python experiments/epic2/phase1_confidence/probe.py qwen3-8b qwen3-1.7b qwen3-8b-base --oof
```

Then on the Mac: `analyze_probes.py`, `plots_probes.py`.
