# Epic 2 · Phase 1 — Confidence Audit

Can a model tell when its own answer is solid from when it is on shaky ground? Five ways of eliciting confidence (stated after, sample agreement, Kalai threshold, stated before, token probabilities) compared on the same questions, across seven question categories with difficulty levels; calibration, discrimination and abstention per category.

Plan: [`docs/epic2_phase1_plan_03-11-2026.md`](../../../docs/epic2_phase1_plan_03-11-2026.md).

## Files

- `questions/`, `build_questions.py`: Step 1. Builds `data/epic2/confidence/questions.jsonl` (recall, computation, perception, ill-posed, unknowable, forecastable; post-cutoff to come). The file is committed with `git add -f` (`data/` is ignored) so every machine runs the same questions.
- `check_models.py`: connection check per model (connect, JSON, sampling, logprobs). Results kept in `checks/`.
- `confidence_task.py`: Step 2. The Inspect task: every method on every question (calls P, B×k, A, D, C×4; forecastable F + FB×k), deterministic grading in the run, judge grading in a second pass. Its docstring lists the options.
- `chat_template_raw.jinja`: pass-through chat template so base models can be served by vLLM and prompted few-shot.
- `run_4090.sh`: the 4090 driver: per configuration, start vLLM, run the task, stop vLLM. Logs in `runs/4090/<mode>/<config>/`.
- `analyze.py`: Step 3, still written for the old three-tier set; to be reworked with meta-d′.
- Metrics live in `src/synapses/calibration.py` (tested in `tests/test_calibration.py`); task tests in `tests/test_phase1_task.py`.

## Running

**4090 (WSL, repo root):**

```bash
experiments/epic2/phase1_confidence/run_4090.sh pilot        # 10 questions per category, every config
experiments/epic2/phase1_confidence/run_4090.sh full         # overnight
```

**Judge pass on the 4090:** `experiments/epic2/phase1_confidence/judge_4090.sh [full|pilot|full/<config>]` serves Qwen3-14B-FP8 (thinking off; `THINK=1` for on) and re-scores every log in place.

**Judge validation:** `checks/gold_set.jsonl` holds 120 hand-labelled responses (drafted by Claude, reviewed by Anil, 2026-11-08; built by `make_goldset.py`, readable in `checks/gold_set_review.md`). `check_judge.py` runs the code rules and the judge on it and reports agreement; `judge_4090.sh check` does this on the 4090 with thinking off and on, saving results in `checks/judge_check_*.txt`. Re-run the check whenever a judge prompt or rule changes.

**Judge pass, by hand** (anywhere the grader model is reachable; overwrites the scores in the log):

```bash
uv run --extra evals inspect score <log.eval> \
    --scorer experiments/epic2/phase1_confidence/confidence_task.py@confidence_scorer \
    -S judge=true --model-role grader=<model> --action overwrite --overwrite
```

**Foundry (office):** `run_foundry.sh`, `confidence_task.py` and `questions.jsonl` are copied into the office repo's `confidence/` folder (the synapses copies are the source; re-copy after changes). There: `confidence/run_foundry.sh pilot|full [configs]` runs grok-4-1-fast, gpt-4o, phi-4-mini and gpt-5-nano (generation only) into `outputs/confidence/<mode>/<config>/`. The logs are copied into `runs/foundry/<mode>/<config>/` here and judged on the 4090 with `judge_4090.sh foundry`, so every model has the same validated judge.

**Dry run (Mac):** `uv run --extra evals inspect eval experiments/epic2/phase1_confidence/confidence_task.py --model mockllm/model -T per_category=2 -T k_samples=2`.
