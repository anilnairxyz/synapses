# Phase 1 — Results

Close-out for Phase 1 (VAE / representation learning), covering Steps 1–7 of [`docs/epic1_phase1_plan_23-07-2026.md`](../../../docs/epic1_phase1_plan_23-07-2026.md). All runs: MLP VAE on MNIST (784–400–L), dynamic-binarized train / fixed-binarized test, Adam 1e-3, 30 epochs, seed 0, on the RTX 4090. Numbers below are from those runs; plots live under `outputs/…/plots/` (gitignored).

## Per-step findings

**Step 1 — baseline (L=2, β=1).** Test ELBO −151.5 (recon 145.5 + KL 6.0), both latent dims active, KL rising 5.2 → 6.0 over training (rate flowing into the latent). Init sanity check recon ≈ 553 nats ≈ 784·ln2 confirmed the loss normalization.

**Step 2 — latent scatter.** The ten digit classes separate into regions the model was never told about (labels used only to colour the finished plot). 0/1 well separated; 3/5/8 and 4/9 overlap (those digits *look* alike). Points spill past the 2σ prior circle — the **prior hole** (aggregate posterior wider than N(0,I)).

**Step 3 — manifold.** Inverse-CDF-spaced grid decode: digits morph smoothly with no hard boundaries. Clean canonical digits at the periphery (cluster interiors); ambiguous blends at the centre (the crossroads where clusters meet) — the counterintuitive consequence of the aggregate posterior being pushed outward.

**Step 4 — sample near a class.** Independent SmallCNN judge (98.9%). Targeted hit rates (N(μ̄_c, 0.1·I)): 0→98%, 1→100%, 2→92%, 3→69%, **4→0%, 5→0%**, 6→100%, 7→100%, 8→47%, 9→100%; mean **70.7%**. The 0% cases are the lesson: **the mean of a class need not be a member of the class** — μ̄_4 lands in 9-territory (crescent-shaped, overlapping cluster), μ̄_5 in 3/8-territory. Prior-baseline sampling: 38% of N(0,I) samples were low-confidence (the prior-hole garbage rate).

**Step 5 — latent-dim sweep (L ∈ {2,8,32,128}, β=1).**

| L | ELBO | recon | KL | active | probe |
|---|---|---|---|---|---|
| 2 | −151.4 | 145.4 | 6.0 | 2/2 | 66.4% |
| 8 | −108.7 | 92.1 | 16.6 | 8/8 | 88.4% |
| 32 | −98.8 | 68.7 | 30.1 | 31/32 | 88.6% |
| 128 | −99.0 | 69.7 | 29.3 | **30/128** | 94.0% |

Raw-pixel probe baseline: 90.9%. ELBO plateaus at L=32 (L=128 ≈ L=32). **Active units saturate at ~30** regardless of capacity — the model chose its own dimensionality and left ~98 dims of L=128 at zero KL (the information-bottleneck tail). Probe below raw pixels at L=2 (compression lost class info), crossing it by L=128. (Single seed — the L=32→128 probe rise, despite fewer active units, is intriguing but not seed-verified; reconstruction quality itself saturated at L=32.)

**Step 6 — β sweep (L=32, β ∈ {0,0.1,1,5,20}).**

| β | D (recon) | R (KL) | ELBO | active |
|---|---|---|---|---|
| 0 | 47.4 | **633.6** | −681 | 32/32 |
| 0.1 | 49.9 | 78.5 | −128 | 32/32 |
| 1 | 68.6 | 30.1 | **−98.7** | 31/32 |
| 5 | 114.4 | 8.0 | −122 | 11/32 |
| 20 | 191.0 | 0.6 | −191 | 2/32 |

Convex rate–distortion frontier; β selects the operating point. β=1 is ELBO-optimal (it is the only model minimizing D+R). **β=0 uses 634 nats ≈ 914 bits to represent a 784-bit image — expansion, not compression** (no KL brake → σ→0 → precise, prior-mismatched per-instance codes). β≥5 shows **posterior collapse** (active 31 → 11 → 2), the same failure mode as dead SAE features in Phase 3.

**Step 7 — feature emergence (L=2 checkpoints).** Latent scatter across epochs {1,2,4,8,16,30}: 1 & 7 resolve at epoch 1 and stay put (the most visually distinctive digits, learned first); 4 & 9 appear separated early then re-entangle as the model reorganizes globally under the 2D constraint (a *transient* early feature — emergence is non-monotonic); the crowded middle (0/2/3/5/6/8) starts entangled and resolves slowly and only partially. Active units / rate / probe accuracy vs epoch: _[fill from `emergence_curves.png`]_.

## Answers to the phase's four questions

1. **How many nats through the latent, and how distributed?** ~30 nats at β=1/L=32, concentrated in ~30 active dimensions (per-dim KL spectrum: a handful of high-KL dims, the rest at zero).
2. **Why doesn't L=128 beat L=32?** Active units saturate at ~30 — the model uses only the intrinsic dimensionality MNIST needs and discards extra capacity (zero-KL tail). Reconstruction and ELBO plateau accordingly.
3. **What does β control, in R–D terms?** The price of rate. It slides the model along the convex rate–distortion frontier: low β → high-rate/low-distortion, high β → low-rate/high-distortion (and eventually collapse).
4. **What was discarded, and was that right?** Per-instance pixel detail beyond the ~30-dim shared structure. At β=1 this is the right call — a compressed, generalizable, prior-matching code. β=0 keeping everything (expansion) and β=20 discarding almost everything (collapse) are the two failure modes it sits between.

**In my own words:** _[your synthesis here]_

## For Phase 3

The longitudinal feature-emergence study consumes the checkpoint series saved here:
- Checkpoints: `outputs/phase1_L{L}_beta{β}_s{seed}/ckpt/epoch_{NNNN}.pt` (log-spaced {1,2,4,8,16,30}; carry model + optimizer + epoch + RNG + config).
- Frozen held-out batch: `outputs/<run>/eval_batch.pt` (comparable activations across checkpoints).
- `outputs/` is gitignored — these live on the PC only.
