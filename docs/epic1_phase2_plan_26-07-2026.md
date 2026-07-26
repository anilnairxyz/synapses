# Phase 2 — Self-Supervised Learning (Detailed Plan)

Detail for **Phase 2** of [`epic1_plan_12-07-2026.md`](epic1_plan_12-07-2026.md). Same structure as the [Phase 1 detailed plan](epic1_phase1_plan_23-07-2026.md): the numbered **Steps** are the units of work. The first two steps are deliberately *study* — Phase 1 leaned on your separate fundamentals refresh, but the SSL landscape is the thing to understand *before* writing a contrastive loss, so the reading is folded in here as Steps 1–2.

**Phase question:** what can a model learn about images from *no labels at all* — only the constraint that two augmented views of the same image should map to nearby representations? Which invariances does that instill, how good is the resulting representation, and how is the trivial solution (map everything to one point — *collapse*) avoided?

This is the discriminative mirror of Phase 1. Phase 1 asked what a *generative* objective (reconstruct the input) learns; Phase 2 asks what an *invariance* objective (be robust to augmentation) learns. The **linear probe** is the shared yardstick — the exact protocol from Phase 1 Step 5 (`synapses.metrics.linear_probe`) — so a VAE latent and a SimCLR embedding land on one comparable axis. Keep that comparison in view throughout.

## Compute — GPU-first, unlike Phase 1

Phase 1 ran on the Mac. **Phase 2 does not.** Contrastive learning wants a real backbone (ResNet-18), large batches (256–512, for negatives), and many epochs (~200+), so training is a genuine **4090 handoff** — a few hours on the PC. What the Mac (and the agent) *can* validate: the augmentation pipeline, the NT-Xent loss (unit tests + a tiny `--smoke` run), and — importantly — **all post-hoc analysis on saved embeddings** (linear probe, t-SNE/UMAP, alignment/uniformity). Pattern: **train on the PC, analyze anywhere.** No new dependencies — `torchvision` (ResNet, CIFAR-10) is a base dep; `umap-learn`/`seaborn` are in the already-provisioned `viz` extra.

## Reuse from Phase 1

- `synapses.metrics.linear_probe` — the frozen-features probe, unchanged. The headline SSL metric and the cross-phase comparison axis.
- `synapses.run` — `set_seed`, `RunLogger`, `save_checkpoint`/`load_checkpoint`, `log_spaced_epochs`, `list_checkpoints`. The experiment plumbing is phase-agnostic; reuse wholesale.
- `synapses.viz` — `latent_scatter` (to plot 2-D t-SNE/UMAP embeddings colored by label), `line_plot`, `image_grid` (to show augmentation pairs).
- `synapses.device.get_device`, and the `RESULTS.md` close-out pattern.

New code (SimCLR model, NT-Xent loss, augmentation pipeline, CIFAR data) is the natural moment to introduce **domain grouping in `src/`** — e.g. `src/synapses/models/` for `vae.py` + the new `simclr.py`, per the layout discussion. Decide at build time; the plan assumes reusable pieces still land in `src/synapses/`, phase-agnostic.

---

## Step 1 — Study: the SSL landscape and the collapse problem

No code. The goal is one organizing idea: **every SSL method is a different answer to "how do I avoid representation collapse?"** (the degenerate solution where the encoder maps every input to the same vector, trivially satisfying "views of the same image are close"). Read with that lens and the zoo becomes a taxonomy.

The four families, by anti-collapse mechanism:

- **Contrastive — push negatives apart.** SimCLR (Chen et al. 2020) and MoCo (He et al. 2020). Positives = two augmentations of one image; negatives = other images. Collapse is impossible because negatives are forced apart. Cost: needs many negatives (SimCLR → huge batches; MoCo → a momentum queue to decouple negatives from batch size).
- **Non-contrastive — architectural asymmetry.** BYOL (Grill et al. 2020) and SimSiam (Chen & He 2021). *No negatives.* Collapse is avoided by a predictor head on one branch + stop-gradient (BYOL adds an EMA target network). SimSiam is the one to study hardest: it strips away everything and shows **stop-gradient alone** prevents collapse — a genuinely surprising, clarifying result.
- **Regularization — explicit anti-collapse terms.** VICReg (Bardes et al. 2022) and Barlow Twins (Zbontar et al. 2021). VICReg's three terms — **V**ariance (hinge each dim's std above a floor), **I**nvariance (MSE between views), **C**ovariance (decorrelate dims) — attack collapse directly and are the most interpretable, connecting straight back to Phase 1's per-dimension variance / active-units analysis.
- **Clustering / distillation / predictive.** SwAV (Caron et al. 2020, online clustering), DINO (Caron et al. 2021, self-distillation with centering+sharpening; produces ViT attention maps that segment objects), and I-JEPA (Assran et al. 2023, predict *representations* of masked regions in latent space — LeCun's joint-embedding predictive framework, the bridge toward Phase 3/4 thinking).

Plus one analysis paper to read carefully: **Wang & Isola (2020), "Alignment and Uniformity"** — good representations are *aligned* (positives close) and *uniform* (features spread evenly on the unit hypersphere). This gives two quantitative, collapse-sensitive metrics used in Step 6.

**Deliverable:** short notes answering, for each method, "what is its anti-collapse mechanism, and what does it cost?" — and one sentence connecting collapse here to *posterior collapse* in Phase 1 (both are the model refusing to use its representational capacity).

---

## Step 2 — Study→build: the SimCLR objective and the augmentation recipe

Still mostly study, but now aimed at what you'll implement. Two things to derive/internalize before coding, exactly as `vae_loss` was understood before Step 1 was written:

- **NT-Xent (normalized temperature-scaled cross-entropy).** For a batch of N images, make 2N augmented views. For each view, its positive is the other view of the same image; the other 2N−2 are negatives. Loss per pair: `−log[ exp(sim(zᵢ,zⱼ)/τ) / Σ_k exp(sim(zᵢ,zₖ)/τ) ]`, with `sim` = cosine similarity, τ the temperature (~0.5 for CIFAR). Derive it as a 2N-way classification "which of the others is my positive?" — it's InfoNCE, and it is the object of study, so it gets written by hand and unit-tested (like the KL was).
- **The augmentation recipe *is* the method.** SimCLR's power comes from strong augmentation: random-resized-crop, horizontal flip, color jitter (strong), random grayscale, and Gaussian blur. The composition defines *which invariances the representation is forced to learn* — crop → some translation/scale invariance, color jitter → color invariance, etc. Understand each augmentation as "the invariance I am imposing." This is what Step 5 ablates.

**Deliverable:** the NT-Xent math worked out by hand, and a written list mapping each augmentation to the invariance it induces (the hypothesis Step 5 tests).

---

## Step 3 — Build: SimCLR on CIFAR-10

The baseline training run. Components:

- **Backbone:** ResNet-18, CIFAR-adapted (replace the 7×7 stride-2 stem with 3×3 stride-1 and drop the initial maxpool — standard for 32×32; the ImageNet stem throws away too much on tiny images). Backbone output (512-D) is the representation kept for evaluation.
- **Projection head:** small MLP (512 → 512 → 128) with BN+ReLU, trained but **discarded at eval** (SimCLR's key finding: probe the backbone, not the projection). Understand *why* the head helps training but hurts as a final representation.
- **NT-Xent loss** (from Step 2), temperature τ≈0.5.
- **Two-view augmentation pipeline** (from Step 2).
- **Data:** CIFAR-10 (32×32, 50k train; well-documented small-scale SimCLR recipes exist). STL-10 is the alternative (100k unlabeled, 96×96) but heavier — see Open questions.
- **Optimization:** SGD+momentum or LARS, cosine schedule with linear warmup, batch 256–512, ~200 epochs (more helps; small-scale SimCLR keeps improving to ~800). Fixed seed, config dataclass, JSONL logging + log-spaced checkpoints — reuse `synapses.run`.
- **Same normalization discipline as Phase 1:** an init sanity check on the loss (at random init NT-Xent ≈ `log(2N−1)`, the uniform-over-negatives value) to catch a broken loss before burning GPU hours.

**Done when:** the NT-Xent curve descends and stabilizes; a smoke run (2 epochs) is clean on the Mac before the full run goes to the PC.

---

## Step 4 — Evaluate: linear probe + embedding visualization

The payoff — does a label-free objective produce a representation that *knows the classes*?

- **Linear probe (headline).** Freeze the backbone, encode CIFAR-10 train+test to 512-D features, fit `synapses.metrics.linear_probe`. Report test accuracy. Expect ~80–90% at 200+ epochs with ResNet-18 — and put it next to the Phase 1 numbers and a **raw-pixel baseline** (same as Step 5 there). This is the moment the two phases become comparable.
- **Embedding visualization.** t-SNE and/or UMAP (the `viz` extra) of the 512-D features down to 2-D, colored by label, via `latent_scatter`. Unlike the Phase 1 latent scatter (which came free at L=2), here it's a projection — but the read is the same: do classes form clean clusters *without ever having seen a label*?
- **k-NN accuracy** (cheap, no training): a non-parametric sanity check on the feature space that complements the linear probe.

**Expect:** clear class clusters; the animal/vehicle superstructure of CIFAR-10 visible; the probe far above raw pixels and competitive with a supervised baseline.

---

## Step 5 — Augmentation ablation (the experiment)

The master plan's "strong vs weak augmentation" experiment, made quantitative. Retrain (or fine-tune short runs) under varied augmentation strength and measure the probe:

- **Strong** (full SimCLR recipe) vs **weak** (crop+flip only) vs **minimal** (flip only).
- Toggle individual augmentations (drop color jitter; drop grayscale; drop blur) to isolate each one's contribution — the direct test of the Step 2 invariance hypotheses.

**What to read off:** which augmentations carry the representation (color jitter is famously load-bearing for SimCLR — removing it collapses probe accuracy, because without it the network cheats by matching color histograms instead of learning shape). When does quality *degrade*? This is the concrete "what is the objective actually doing" answer the master plan asks for.

---

## Step 6 — Collapse diagnostics (and an optional second method)

Make the Step 1 collapse story measurable, and connect it to Phase 1.

- **Alignment & uniformity** (Wang & Isola): alignment = mean distance between positive pairs (lower = better); uniformity = log-mean-exp of pairwise distances on the hypersphere (more negative = more spread). Track both over training. A collapsing run shows alignment→0 *and* uniformity→0 (everything coincides).
- **Feature-space rank / per-dimension variance** — the direct analog of Phase 1's active units and per-dimension KL. A healthy SSL representation uses many dimensions; a collapsing one has few effective dimensions (low rank). Reuse the *idea* from `synapses.metrics.active_units`.
- **Optional second method for contrast:** implement **VICReg** on the same backbone. It's the cheapest non-SimCLR method to add (no negatives, no EMA, no queue — just the V/I/C regularizers) and its variance term is *exactly* the anti-collapse mechanism made explicit, so it pairs beautifully with the diagnostics above and with Phase 1's per-dimension analysis. This is where "how does each method avoid collapse" stops being a reading-notes claim and becomes something you measured. (Decision — see Open questions.)

---

## Step 7 — Checkpoints for the longitudinal project + close-out

Mirror Phase 1 Step 7. The stretch project ("From Representation to Features": train a repr model → extract activations → train an SAE → analyze features across checkpoints) can use a SimCLR encoder as its representation model, so the checkpoint discipline matters here too.

- Log-spaced checkpoints (model + optimizer + epoch + RNG + config) via `synapses.run` — already built.
- A frozen held-out batch for comparable feature extraction across checkpoints.
- Record seeds; note the gitignored `outputs/` paths.
- **`experiments/epic1/phase2_simclr/RESULTS.md`** — the numbers, the augmentation-ablation findings, the SimCLR-vs-VAE probe comparison, and the answer to the phase question in your own words.

---

## Deliverables

- `experiments/epic1/phase2_simclr/` — backbone + projection head, NT-Xent, augmentation pipeline, train loop, eval/probe, ablation driver, diagnostics.
- Study notes (Steps 1–2): the collapse taxonomy + the augmentation→invariance mapping.
- Linear-probe accuracy vs a raw-pixel baseline and vs the Phase 1 VAE probe.
- t-SNE/UMAP embedding scatter colored by label.
- Augmentation-ablation table (probe accuracy across recipes).
- Alignment/uniformity (and rank) curves over training.
- Checkpoint series + `RESULTS.md`.

## Success criteria

Not "the code runs". Phase 2 is done when you can answer, from your own results:

1. How good is a label-free representation, on the same linear-probe axis as the Phase 1 VAE? Why is it better (or worse)?
2. Which augmentations carry the representation, and what invariance does each impose?
3. What does representation collapse look like *quantitatively* (alignment/uniformity/rank), and how does SimCLR's negative-sampling avoid it — versus VICReg's explicit regularizers?
4. How does what an SSL objective *chooses to keep* differ from what the VAE kept in Phase 1?

## Exit ramps (out of scope for Phase 2)

- **BYOL / SimSiam / DINO implementations** — study them, but implementing beyond one comparison method (VICReg) is scope creep; DINO in particular wants a ViT and is a project of its own.
- **STL-10 / ImageNet-scale** — CIFAR-10 is enough to see every phenomenon; larger data is a compute sink, not new insight, for this phase.
- **Supervised-contrastive / fine-tuning the whole backbone** — a different question (transfer) than "what did SSL learn".

## Open questions

- **Dataset:** CIFAR-10 (recommended — fast, standard small-scale recipes) vs STL-10 (more unlabeled data, closer to the SSL use case, but heavier).
- **Second method:** implement VICReg as the Step 6 contrast, or keep Phase 2 SimCLR-only and treat other methods as study only?
- **Epoch budget:** ~200 (sees the phenomena, a few hours) vs ~800 (near-SOTA small-scale probe, most of a day on the 4090).
- **Optimizer:** SGD+cosine (simple, robust at this scale) vs LARS (SimCLR's choice, matters more at large batch).
