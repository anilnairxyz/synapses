# Epic 2 · Phase 1 — Confidence Audit

Can a model tell when its own answer is solid from when it is on shaky ground? Verbal confidence vs sample consistency on a three-tier (easy / hard / unknowable) question set; calibration, discrimination and abstention.

Plan: [`docs/epic2_phase1_plan_03-11-2026.md`](../../../docs/epic2_phase1_plan_03-11-2026.md).

## Files

- `build_dataset.py`: Step 1. Builds `data/epic2/confidence/questions.jsonl` (100 TriviaQA / 100 SimpleQA / 100 unknowable from KUQ + SelfAware, seed 0).
- `confidence_task.py`: Steps 2 + 4. The Inspect task: JSON answer + confidence (or ABSTAIN), string-match-then-LLM-judge grading, optional k samples for consistency confidence.
- `analyze.py`: Step 3. Per run × tier: accuracy, abstention, Brier (+ Murphy), ECE, AUROC for verbal and consistency confidence; reliability and selective/penalised-score plots.
- Metrics live in `src/synapses/calibration.py` (tested in `tests/test_calibration.py`).

## Running

API-only, so this runs on the Mac. Put keys in `.env` (copy `.env.example`), then `uv sync --extra evals`.

```bash
TASK=experiments/epic2/phase1_confidence/confidence_task.py
LOGS=outputs/epic2/confidence
GRADER=anthropic/claude-haiku-5-5   # use a grader from the other provider when grading its own family, if you want to rule out self-preference

uv run python experiments/epic2/phase1_confidence/build_dataset.py

# 1. smoke on real models: check parsing and judge verdicts in `uv run inspect view`
uv run inspect eval $TASK --model anthropic/claude-haiku-5-5 -T limit_per_tier=5 \
    --model-role grader=$GRADER --log-dir outputs/epic2/smoke-real

# 2. baseline + consistency (k=10) per model
uv run inspect eval $TASK --model <provider/model> -T k_samples=10 --model-role grader=$GRADER --log-dir $LOGS

# 3. stretch: confidence before the answer
uv run inspect eval $TASK --model <provider/model> -T elicit=pre --model-role grader=$GRADER --log-dir $LOGS

uv run python experiments/epic2/phase1_confidence/analyze.py --log-dir $LOGS
```

Cost per model at k=10: about 300 × 11 short generations plus about 300–600 grader calls.
