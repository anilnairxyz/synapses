# synapses

A personal exploration of frontier AI through small, hands-on reproductions of mainline techniques. Each experiment builds the thing small, makes its insides visible, then varies or breaks it on purpose; the results become posts on the [lifeandtimes blog](https://anilnairxyz.github.io/lifeandtimes/) (the "Inside …" series). The goal is to learn deeply and start contributing to research, not to build products.

## Epics

| Epic | Theme | Status | Plan |
| ---- | ----- | ------ | ---- |
| **Epic 2** | Epistemic honesty evals: calibration, sycophancy, belief-vs-answer honesty (for an Apart Research sprint, Track 1) | **Active** | [`docs/epic2_plan_03-11-2026.md`](docs/epic2_plan_03-11-2026.md) |
| Epic 1 | From Representation to Features: Representation Learning → Mechanistic Interpretability | Paused (2026-11-03) | [`docs/epic1_plan_12-07-2026.md`](docs/epic1_plan_12-07-2026.md) |

**Epic 1, where it stopped.** Phase 1 (VAE) is complete with [`RESULTS.md`](experiments/epic1/phase1_vae/RESULTS.md). Phase 2 (SSL) has SimCLR, SimSiam and VICReg built, analysed and posted. Not done: the Phase 2 side-by-side comparison (Step 6), I-JEPA (Step 7), the Phase 2 close-out, and Phases 3–4 (SAEs, mech interp). Code, plans and checkpoints stay in place so it can be resumed.

## Hardware & Dev Setup

I work across two machines, with a third compute tier available for larger evals:

| Tier                              | Role                          | Details                                                                                                                                          |
| --------------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Mac (M3 Pro)** — *this machine* | Primary dev environment       | Where I write and code. Runs Claude / Codex. My preferred machine for typing and editing.                                                        |
| **PC (Windows)**                  | GPU compute                   | AMD Ryzen 9 7950X, NVIDIA RTX 4090. Runs **WSL Ubuntu** for training.                                                                            |
| **Azure AI Foundry**              | Hosted model evals (overflow) | Foundry-hosted models I can evaluate against when the 4090 is a bottleneck (large models, throughput). Accessible via my work at Stockport Labs. |

### How it fits together

- I do all my **coding on the Mac** (this machine) — editing, agents, git, planning.
- Actual **GPU workloads run on the PC** inside WSL Ubuntu.
- The two are wired together so I can drive the PC's GPU from the Mac via **VS Code Remote / Terminal (SSH)** — editing files and running training remotely as if local.
- When a task exceeds the 4090 (models too large to fit, or eval throughput at scale), I can run certain **evaluations against Azure AI Foundry models** instead. This is an occasional overflow path for eval workloads — not the primary training environment.

### ⚠️ Important constraint for AI agents

**Agents running on this Mac cannot execute GPU code.** The 4090 lives on the PC, and while I can reach it interactively over SSH / VS Code Remote, agents here have no direct path to run GPU workloads.

Implications for any agent working in this repo:
- Write code that is **runnable on the remote PC/WSL GPU**, but don't assume you can execute it here.
- CPU-only smoke tests, unit tests, linting, data prep, and visualization scaffolding **can** be validated on the Mac.
- Anything requiring CUDA / the 4090 must be **handed off to run manually** on the PC. Flag these steps clearly rather than attempting them locally.
- **Epic 2 is mostly API-driven** (evals against hosted models), so much of it *can* run on the Mac. Open-weights models served on the 4090 are still a handoff.

### Markdown conventions (for AI agents)

- **Do not hard-wrap prose** in Markdown files (`README.md`, `docs/*.md`, `RESULTS.md`, experiment `README.md`s). Write each paragraph as a single line and let the editor soft-wrap. Tables, code fences, and list items follow normal Markdown.

## Environment (uv + conda, "Option B")

Dependencies are managed with **uv** (`pyproject.toml` + `uv.lock`). On the PC, uv installs *into* the existing conda env rather than creating a `.venv`; on the Mac it uses a plain `.venv`. Torch is pulled from the CUDA 12.4 index on Linux and from PyPI (CPU/MPS) on macOS — see `[tool.uv.sources]` in `pyproject.toml`.

**On the PC (WSL Ubuntu, RTX 4090):**
```bash
conda activate sapphire                       # your conda env
export UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX"  # make uv target the conda env
uv sync --extra interp --extra viz            # add --extra diffusers for the SD test
```

**On the Mac (M3 Pro — this machine):**
```bash
uv sync            # creates .venv with CPU/MPS torch, for smoke tests & scaffolding
```

> The PC conda env is named `sapphire` (Python 3.12.11). Tip: add `export UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX"` to `$CONDA_PREFIX/etc/conda/activate.d/uv.sh` so it's set automatically on activation.

### Setup smoke test

```bash
uv run python scripts/check_setup.py             # env report + tiny matmul on best device
uv run python scripts/check_setup.py --diffusers # Stable Diffusion gen test — run on the PC
```

The basic check runs anywhere (CUDA / MPS / CPU). The `--diffusers` test needs the GPU and the `diffusers` extra, so run it on the PC.

## Repo Layout

```
docs/                     Planning docs (one master plan per epic, plus per-phase plans)
src/synapses/             Shared library (device helpers, models, viz — grows over time)
scripts/                  Runnable utilities (check_setup.py)
experiments/
  epic1/                  Epic 1 — "From Representation to Features"
    phase1_vae/           Phase 1 — VAE / representation learning
    phase2_ssl/           Phase 2 — self-supervised learning
    phase3_sae/           Phase 3 — sparse autoencoders
    phase4_mech_interp/   Phase 4 — mechanistic interpretability
  epic2/                  Epic 2 — epistemic honesty evals (Apart sprint, Track 1)
notebooks/                Exploration notebooks
data/                     Datasets (gitignored)
outputs/                  Generated artifacts: images, checkpoints, plots (gitignored)
```

Experiment scripts import shared code from `synapses`, keeping each epic DRY across its phases (Epic 1's longitudinal chain was train repr → extract activations → train SAE → analyze). Shared code is grouped by kind, never by epic or phase.
