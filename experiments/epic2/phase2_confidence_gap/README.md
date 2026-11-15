# Epic 2 · Phase 2 — Why Confidence Reports Miss What the Model Knows

> **Sprint work (AI x Epistemics Research Sprint, 13 to 16 November 2026).** Everything in this folder, and in `data/epic2/confidence/phase2/`, was written during the sprint. It builds on preexisting work, Phases 1 and 1B in [`../phase1_confidence/`](../phase1_confidence/) and the metrics in [`src/synapses/calibration.py`](../../../src/synapses/calibration.py), written 3 to 12 November 2026. The boundary is the git tag [`sprint-start`](https://github.com/anilnairxyz/synapses/tree/sprint-start) (commit `8ae8824`, the last commit before the sprint began on 13 November, 14:15 UTC). Preexisting code is not edited during the sprint; any fix to it is a separate, labelled commit listed below.

Plan: [`docs/epic2_phase2_plan_12-11-2026.md`](../../../docs/epic2_phase2_plan_12-11-2026.md).

Phases 1 and 1B found that whether a model can know it is wrong depends on how the answer is produced (recall is known before answering, computation only after, perception barely), and that probes of the hidden states know far more than stated confidence. Phase 2 tests why the confidence reports miss what the model knows: a stereotype analysis (E1), a swap experiment with right and wrong answers by construction (E2), a comparison of ways of asking (E3), the directions behind stated confidence and correctness with steering (E4), and training stated confidence on correctness (E5).

## What is reused from Phase 1

Phase 2 reaches Phase 1 only through [`phase1.py`](phase1.py), which lists every module and function used (question set and prompts, grading, state extraction, probes, calibration metrics) and the Phase 1 outputs read. Nothing is copied. A test (`tests/test_phase2_boundary.py`) fails if any other Phase 2 file imports Phase 1 directly.

Inputs read from Phase 1 (unchanged; `../phase1_confidence/results/rows.parquet` is a gitignored cache, so on a fresh machine build it first with `uv run --extra evals python experiments/epic2/phase1_confidence/analyze.py check --refresh`): the question sets in `data/epic2/confidence/`, the judged run logs in `../phase1_confidence/runs/`, the probe results in `../phase1_confidence/results/`, and the hidden states on the PC (`../phase1_confidence/states/`, not committed).

## Files

- `phase1.py`: the reuse surface (above).
- `analyze_stereotype.py`: E1, the stereotype analysis (Mac, Phase 1 outputs only). `--null` runs the shuffle check.
- `build_swaps.py`: E2 / E3 inputs (Mac): for each Phase 1B config, the model's own answer, the correct answer and a plausible wrong one built by construction, the injected ones written in the model's own format; plus the training questions' own answers for E3. Checks that every injected answer grades as built with Phase 1's grader. Writes `data/epic2/confidence/phase2/swaps.jsonl` (committed with `git add -f`). Tests: `tests/test_phase2_swaps.py`.
- `swap_task.py`: E2's verbal confidence (Inspect, vLLM): each given answer in the assistant turn ("your answer") or from the user ("my answer").
- `extract_answers.py`: on the 4090 (HF): states, answer log-probability and P(True) (user and assistant placement, 0- and 8-shot) for every given answer, then Phase 1B's probes refitted and scored on the new states (with a check that own answers reproduce Phase 1B's scores). Writes `results/extract_<config>.parquet`; states under `states/` (gitignored).
- `directions.py`: E4 Part A (PC, CPU): which verbal confidence can be read linearly from the answer's state, and how its direction relates to the correctness probe's; saves the probe directions for steering.
- `steer.py`: E4 Part B on the 4090 (HF): the E2 confidence conversations with the correctness probe's direction (or a random one) added at the probe's layer.
- `analyze_steer.py`: E4 Part B analysis (Mac): shift per condition, probe vs random, separation and output collapse under steering.
- `lora_confidence.py`: E5 on the 4090 (HF + peft): LoRAs training the "your answer" report on per-answer correctness, with Lin-style (stratum accuracy) and shuffled controls and leave-one-kind-out transfer; evaluated on the E2 test answers. Adapters in `lora_runs/` (gitignored).
- `analyze_lora.py`: E5 analysis (Mac): within-level and pooled AUROC of the trained report on own answers against the probe and the untrained model, separation on injected answers, transfer to a held-out kind.
- `analyze_replies.py`: E2b follow-up (Mac): share of replies with written working per model and framing, and Claude Haiku's separation with and without working.
- `verify.py`: checks every number quoted in the report against the results files (`--rerun` recomputes them first from the logs); exits non-zero if any claim fails.
- `office/` and `make_office_bundle.py`: E2b on Azure AI Foundry (GPT-4o, GPT-5 nano, Grok 4.1 Fast): a self-contained bundle for the office; see `office/README.md`.
- `run_4090.sh`: `smoke` or `full`, optionally only `extract` or `swap`.
- `results/`: tables, figures, and the findings so far ([`results/README.md`](results/README.md)).

- `analyze_swaps.py`: E2 and E3 analysis (Mac): separation of correct vs planted wrong answers per readout and attribution, with the familiarity control; readouts of confidence in own answers against the probe; calibration; the confident tail. Findings in [`results/README.md`](results/README.md).

To come, per the plan: `directions.py`, `steer.py` (E4); `lora/` (E5); `verify.py`; `plots.py`; `run_4090.sh`; `results/`, `runs/`.

## Fixes to preexisting code during the sprint

None so far.
