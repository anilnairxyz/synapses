# Epic 2 · Phase 2 — Why Confidence Reports Miss What the Model Knows

> **Sprint work (AI x Epistemics Research Sprint, 13 to 16 November 2026).** Everything in this folder, and in `data/epic2/confidence/phase2/`, was written during the sprint. It builds on preexisting work, Phases 1 and 1B in [`../phase1_confidence/`](../phase1_confidence/) and the metrics in [`src/synapses/calibration.py`](../../../src/synapses/calibration.py), written 3 to 12 November 2026. The boundary is the git tag [`sprint-start`](https://github.com/anilnairxyz/synapses/tree/sprint-start) (commit `8ae8824`, the last commit before the sprint began on 13 November, 14:15 UTC). Preexisting code is not edited during the sprint; any fix to it is a separate, labelled commit listed below.

Plan: [`docs/epic2_phase2_plan_12-11-2026.md`](../../../docs/epic2_phase2_plan_12-11-2026.md).

Phases 1 and 1B found that whether a model can know it is wrong depends on how the answer is produced (recall is known before answering, computation only after, perception barely), and that probes of the hidden states know far more than stated confidence. Phase 2 tests why the confidence reports miss what the model knows: a stereotype analysis (E1), a swap experiment with right and wrong answers by construction (E2), a comparison of ways of asking (E3), the directions behind stated confidence and correctness with steering (E4), and training stated confidence on correctness (E5).

## What is reused from Phase 1

Phase 2 reaches Phase 1 only through [`phase1.py`](phase1.py), which lists every module and function used (question set and prompts, grading, state extraction, probes, calibration metrics) and the Phase 1 outputs read. Nothing is copied. A test (`tests/test_phase2_boundary.py`) fails if any other Phase 2 file imports Phase 1 directly.

Inputs read from Phase 1 (unchanged): the question sets in `data/epic2/confidence/`, the judged run logs in `../phase1_confidence/runs/`, the probe results in `../phase1_confidence/results/`, and the hidden states on the PC (`../phase1_confidence/states/`, not committed).

## Files

- `phase1.py`: the reuse surface (above).
- `analyze_stereotype.py`: E1, the stereotype analysis (Mac, Phase 1 outputs only). `--null` runs the shuffle check.
- `results/`: tables, figures, and the findings so far ([`results/README.md`](results/README.md)).

To come, per the plan: `build_swaps.py`, `swap_task.py`, `extract_answers.py` (E2, E3); `analyze.py`; `directions.py`, `steer.py` (E4); `lora/` (E5); `verify.py`; `plots.py`; `run_4090.sh`; `results/`, `runs/`.

## Fixes to preexisting code during the sprint

None so far.
