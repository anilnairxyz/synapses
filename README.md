# synapses

A personal exploration of frontier AI — **Representation Learning** and **Mechanistic Interpretability**. The goal is to learn deeply and start contributing to research, not to build products.

See [`docs/plan_12-07-2026.md`](docs/plan_12-07-2026.md) for the full 3–4 week plan.

## Hardware & Dev Setup

I work across two machines:

| Machine | Role | Specs |
|---------|------|-------|
| **Mac (M3 Pro)** — *this machine* | Primary dev environment | Where I write and code. Runs Claude / Codex. My preferred machine for typing and editing. |
| **PC (Windows)** | GPU compute | AMD Ryzen 9 7950X, NVIDIA RTX 4090. Runs **WSL Ubuntu** for training. |

### How it fits together

- I do all my **coding on the Mac** (this machine) — editing, agents, git, planning.
- Actual **GPU workloads run on the PC** inside WSL Ubuntu.
- The two are wired together so I can drive the PC's GPU from the Mac via **VS Code Remote / Terminal (SSH)** — editing files and running training remotely as if local.

### ⚠️ Important constraint for AI agents

**Agents running on this Mac cannot execute GPU code.** The 4090 lives on the PC, and while I can reach it interactively over SSH / VS Code Remote, agents here have no direct path to run GPU workloads.

Implications for any agent working in this repo:
- Write code that is **runnable on the remote PC/WSL GPU**, but don't assume you can execute it here.
- CPU-only smoke tests, unit tests, linting, data prep, and visualization scaffolding **can** be validated on the Mac.
- Anything requiring CUDA / the 4090 must be **handed off to run manually** on the PC. Flag these steps clearly rather than attempting them locally.

## Environment (uv + conda, "Option B")

Dependencies are managed with **uv** (`pyproject.toml` + `uv.lock`). On the PC, uv installs *into* the existing conda env rather than creating a `.venv`; on the Mac it uses a plain `.venv`. Torch is pulled from the CUDA 12.4 index on Linux and from PyPI (CPU/MPS) on macOS — see `[tool.uv.sources]` in `pyproject.toml`.

**On the PC (WSL Ubuntu, RTX 4090):**
```bash
conda activate synapses                       # your conda env
export UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX"  # make uv target the conda env
uv sync --extra interp --extra viz            # add --extra diffusers for the SD test
```

**On the Mac (M3 Pro — this machine):**
```bash
uv sync            # creates .venv with CPU/MPS torch, for smoke tests & scaffolding
```

> Replace `synapses` with your actual conda env name if it differs.

### Setup smoke test

```bash
uv run python scripts/check_setup.py             # env report + tiny matmul on best device
uv run python scripts/check_setup.py --diffusers # Stable Diffusion gen test — run on the PC
```

The basic check runs anywhere (CUDA / MPS / CPU). The `--diffusers` test needs the GPU and the `diffusers` extra, so run it on the PC.

## Repo Layout

```
docs/                     Planning docs (the exploration plan)
src/synapses/             Shared library (device helpers, models, viz — grows over time)
scripts/                  Runnable utilities (check_setup.py)
experiments/
  week1_vae/              Week 1 — VAE / representation learning
  week2_simclr/           Week 2 — self-supervised learning
  week3_sae/              Week 3 — sparse autoencoders
  week4_mech_interp/      Week 4 — mechanistic interpretability
notebooks/                Exploration notebooks
data/                     Datasets (gitignored)
outputs/                  Generated artifacts: images, checkpoints, plots (gitignored)
```

Experiment scripts import shared code from `synapses`, keeping the longitudinal "From Representation to Features" project (train repr → extract activations → train SAE → analyze) DRY across weeks.
