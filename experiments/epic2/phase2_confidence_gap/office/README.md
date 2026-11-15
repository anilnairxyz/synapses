# E2b on Azure AI Foundry (office bundle)

Self-contained copy of the E2b swap task for the Foundry models (GPT-4o, GPT-5 nano, Grok 4.1 Fast), which are not reachable from the Mac or the 4090. Sprint work (AI x Epistemics sprint, 15 November 2026). The source of truth is `experiments/epic2/phase2_confidence_gap/` in synapses; `swap_task_foundry.py` carries copies of `swap_task.py`'s and Phase 1's prompts and parsers so it needs only Inspect and the standard library (`tests/test_phase2_office.py` checks the copies are identical).

- `swaps_foundry.jsonl`: the answers to rate (each model's own Phase 1 answer, the correct answer, a plausible wrong one in the model's format), with the question text; built by `make_office_bundle.py`.
- `swap_task_foundry.py`: the Inspect task ("your answer" vs "my answer").
- `run_foundry.sh`: per-model settings as in Phase 1 (Grok non-reasoning, GPT-4o; GPT-5 nano at its default temperature with reasoning effort low), max_tokens 512 (4,000 for GPT-5 nano).

**At the office:** copy this folder into the office repo (for example as `swap/`), then from the office repo root run `swap/run_foundry.sh smoke`, check that the summary shows `parsed` near 1, then `swap/run_foundry.sh full`. Bring `outputs/swap/full/<config>/*.eval` back to synapses as `experiments/epic2/phase2_confidence_gap/runs/foundry/full/<config>/`, and commit them there.
