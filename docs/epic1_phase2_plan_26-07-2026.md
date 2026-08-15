# Phase 2 — Self-Supervised Learning (Detailed Plan)

*Revised 2026-08-14: builds one method per SSL family (SimCLR, SimSiam, VICReg, I-JEPA) on a shared backbone rather than SimCLR alone — the anti-collapse mechanisms only become obvious when you implement and break each one.*

Detail for **Phase 2** of [`epic1_plan_12-07-2026.md`](epic1_plan_12-07-2026.md). Same structure as the [Phase 1 detailed plan](epic1_phase1_plan_23-07-2026.md): the numbered **Steps** are the units of work.

**Phase question:** what can a model learn about images from *no labels at all* — only the constraint that two views of an image should agree? And, the revised emphasis: **how do the different anti-collapse mechanisms actually differ** when everything else (backbone, data, augmentation, evaluation) is held fixed? SimCLR alone answers the first question but not the second; you only *feel* the difference between preventing collapse with negatives, with optimisation asymmetry, with explicit latent statistics, and with latent-space prediction by building — and breaking — one of each.

This is the discriminative mirror of Phase 1. Phase 1 asked what a *generative* objective (reconstruct) learns; Phase 2 asks what an *invariance* objective learns. The **linear probe** is the shared yardstick — the exact protocol from Phase 1 Step 5 (`synapses.metrics.linear_probe`) — so a VAE latent and each SSL embedding land on one comparable axis.

## Which methods, and why all four

The study is already done and written up — two blog posts in the `lifeandtimes` repo cover the landscape ("Self-Supervised Learning: from SimCLR to SimSiam" and "Barlow Twins and VICReg: Regularising the Latent Space"). The build set takes **one method per family**, so every distinct anti-collapse mechanism gets implemented and, crucially, *broken*:

- **SimCLR** — *contrastive*. Collapse avoided by negatives (other images in the batch pushed apart). Its distinguishing cost is the appetite for large batches.
- **SimSiam** — *non-contrastive / optimisation-asymmetry*. Predictor + stop-gradient, no negatives, no EMA, no queue. The simplest member, and the one whose collapse is most dramatic to demonstrate.
- **VICReg** — *regularisation*. Explicit variance/covariance terms on the embeddings. Connects directly to Phase 1's per-dimension analysis.
- **I-JEPA** — *predictive / latent*. Predict representations of masked regions in latent space. The heaviest (needs a ViT) and the natural bridge to Phase 4.

The payoff is the **comparison**: same backbone, same CIFAR-10, same probe → four representations whose only difference is the mechanism, measured side by side. The three view-based CNN methods (SimCLR/SimSiam/VICReg) are directly comparable; I-JEPA (different architecture, no augmentation views) is compared on the shared probe axis.

## Compute — GPU-first, and now four methods

Phase 1 ran on the Mac. **Phase 2 does not.** A real backbone (ResNet-18), batches of 256–512, and ~200 epochs make each run a genuine **4090 handoff** — a few hours apiece. With four methods plus two break-it ablations that is ~6 training runs, so the epoch budget matters: **~200 epochs per method** is the default here. That is enough to see every phenomenon (clusters form, the probe rises, the ablations collapse); it is deliberately *not* a leaderboard chase (near-SOTA CIFAR probes want ~800 epochs, most of a day each). Take a single best method to 800 later if you want the headline number.

What the Mac (and the agent) *can* validate: the augmentation pipeline, every loss (unit tests + a tiny `--smoke` run), and **all post-hoc analysis on saved embeddings** (linear probe, k-NN, t-SNE/UMAP, collapse diagnostics). Pattern: **train on the PC, analyze anywhere.** No new dependencies for the CNN methods — `torchvision` (ResNet, CIFAR-10) is a base dep, `umap-learn`/`seaborn` are in the provisioned `viz` extra. I-JEPA needs a small ViT (hand-rolled or `torchvision`).

## Reuse from Phase 1

- `synapses.metrics.linear_probe` — the frozen-features probe, unchanged. Headline metric + cross-phase axis.
- `synapses.metrics.active_units` — the per-dimension variance idea, reused as a collapse diagnostic on SSL embeddings.
- `synapses.run` — `set_seed`, `RunLogger`, `save/load_checkpoint`, `log_spaced_epochs`, `list_checkpoints`. Phase-agnostic plumbing, reuse wholesale.
- `synapses.viz` — `latent_scatter` (for 2-D t-SNE/UMAP scatters), `line_plot`, `image_grid` (augmentation pairs, collapse curves).
- `synapses.device.get_device`, and the `RESULTS.md` close-out pattern.

New reusable code lands in `src/synapses/`, and this is the moment to introduce **domain grouping**: `src/synapses/models/` for `vae.py` + the new backbones, and `src/synapses/ssl/` (heads + losses per method). Group by *kind*, never by phase.

---

## Step 1 — Study recap + I-JEPA

Study is largely complete (the two blog posts are the deliverable). This step only fills gaps:

- **Re-read your SimCLR/SimSiam and VICReg notes** with an eye to implementation detail: SimCLR's NT-Xent and temperature; SimSiam's predictor/stop-gradient asymmetry and symmetrised cosine loss; VICReg's three terms, standard weights (λ=25, μ=25, ν=1), and the variance hinge target (γ=1).
- **New reading — I-JEPA** (Assran et al. 2023) and the JEPA framing (LeCun): context encoder, EMA target encoder, a predictor that reconstructs *target-block representations* (not pixels) from context + positional tokens, with masking replacing augmentation.
- **Diagnostics — Wang & Isola (2020)**, alignment & uniformity, the quantitative collapse metrics used in Step 6.

**Deliverable:** short notes on I-JEPA's mechanism and where it sits relative to the view-based families (it doesn't avoid collapse with a "trick" — masking + latent targets sidestep the trivial solution differently).

---

## Step 2 — Build the shared SSL substrate

The foundation the view-based methods plug into — build once, reuse for SimCLR, SimSiam, VICReg.

- **Backbone:** CIFAR-adapted ResNet-18 (3×3 stride-1 stem, no initial maxpool — the ImageNet stem destroys 32×32 images). Its 512-D output is *the representation* kept for evaluation.
- **Two-view augmentation pipeline:** random-resized-crop, horizontal flip, color jitter, random grayscale, Gaussian blur. Each image yields two independently augmented views.
- **CIFAR-10 data** (50k train), plus a labelled split for the probe.
- **Training skeleton:** two-view forward, generic loss hook, cosine LR schedule with warmup, JSONL logging + log-spaced checkpoints (reuse `synapses.run`), a `--smoke` path, and a per-method loss init-sanity check.
- **Evaluation harness (the shared yardstick):** freeze backbone → encode train/test → `linear_probe` + k-NN accuracy; t-SNE/UMAP scatter via `latent_scatter`.

Each view-based method (Steps 3–5) supplies only a **projector/predictor head + a loss**; the backbone, augmentation, data, loop, and eval are this shared substrate. That is what makes four methods a modest increment rather than four projects.

**Done when:** a `--smoke` run trains end-to-end on the Mac (tiny, CPU/MPS) and the eval harness produces a probe number on a random-init backbone (the floor to beat).

---

## Step 3 — Build SimCLR (contrastive)

Add a projector (no predictor); loss is NT-Xent over the 2N views in a batch, positives = the two views of an image, negatives = all others:

$$
\ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k)/\tau)}, \qquad \text{sim}(u,v) = \frac{u^\top v}{\lVert u\rVert\,\lVert v\rVert}
$$

Temperature τ≈0.5, batch 256–512 (negatives come from the batch, so this is the method that cares about batch size). Init sanity: at random init NT-Xent ≈ `log(2N−1)`. Train; probe. This is the *sample-side* mechanism — collapse is impossible because equal representations make the positive indistinguishable from negatives.

---

## Step 4 — Build SimSiam (non-contrastive)

Add a 3-layer projector and a 2-layer **predictor**; loss is the symmetrised negative cosine with **stop-gradient** on the target branch:

$$
\mathcal{L} = \tfrac{1}{2}\big[D(p_1, \operatorname{sg}(z_2)) + D(p_2, \operatorname{sg}(z_1))\big], \qquad D(p, z) = -\frac{p}{\lVert p\rVert}\cdot\frac{z}{\lVert z\rVert}
$$

No negatives, no EMA, no queue. Train; probe.

**The signature experiment — break it.** Remove the stop-gradient and retrain: the loss dives to −1 almost immediately and the representation collapses (per-dimension std → 0, probe → chance). This is the most instructive result in non-contrastive SSL — it *shows* the anti-collapse mechanism is the optimisation asymmetry, not the loss. Ablate the predictor too (remove it → collapse) to confirm both pieces are load-bearing.

---

## Step 5 — Build VICReg (regularisation)

Same substrate; swap the head for a wider **expander** and the loss for variance-invariance-covariance, no predictor:

$$
\mathcal{L} = \lambda\,\mathcal{L}_{\text{inv}} + \mu\,\mathcal{L}_{\text{var}} + \nu\,\mathcal{L}_{\text{cov}}
$$

invariance = MSE between views; variance = mean hinge `max(0, γ − std(z_j))` per dimension (γ=1); covariance = off-diagonal squared covariance. Standard weights λ=μ=25, ν=1. Train; probe.

**The signature experiment — break it.** Set μ=0 (drop the variance term) and retrain: the representation collapses, directly demonstrating that the variance hinge is *the* anti-collapse mechanism. This is the exact mirror of SimSiam's stop-grad ablation — a different mechanism, broken the same way, same collapse symptom — which is the contrast the whole phase is for.

---

## Step 6 — The comparison (the payoff)

Same backbone, same data, same probe — line up SimCLR, SimSiam, VICReg. This step is why the phase builds more than one.

- **Representation quality:** linear-probe and k-NN accuracy for the three methods, against a raw-pixel baseline and the **Phase 1 VAE probe**. One table, one axis.
- **Collapse diagnostics, side by side:** alignment & uniformity (Wang-Isola), per-dimension std, and **effective rank** of the representation covariance (the SSL analog of Phase 1's active units — how many dimensions are actually used). The healthy methods should score well; the *broken* variants from Steps 4–5 should light these up differently, exposing how each mechanism fails.
- **Augmentation sensitivity:** a short strong-vs-weak sweep on one method (color jitter is famously load-bearing) — which invariances the representation depends on.
- **Embedding scatters:** t-SNE/UMAP of each method's features, coloured by label.

**The question to answer:** do contrastive, non-contrastive, and regularisation reach similar representation quality by different routes — and do their *failure modes* differ even when their successes converge?

---

## Step 7 — Build I-JEPA (predictive)

The predictive/latent family, and the heaviest build — a different architecture from the CNN methods. Small ViT context + EMA target encoders, block masking, a predictor over target-block representations, L2 loss in representation space — no augmentation-invariance, no negatives. CIFAR-10 at 32×32 is *not* I-JEPA's natural habitat (few patches), so treat this as a learning exercise, not a benchmark: a tiny ViT (patch 4 → 64 tokens) is enough to feel the mechanism, and it is the bridge into Phase 4's transformer work. Evaluate on the same frozen linear probe and fold its number into the Step 6 table (alignment/uniformity don't apply — no augmentation views — so compare it on probe/k-NN/effective-rank only).

---

## Step 8 — Checkpoints + close-out

Mirror Phase 1 Step 7. Log-spaced checkpoints (reuse `synapses.run`) for each method so SSL feature emergence is comparable to the VAE's, a frozen held-out batch, recorded seeds, gitignored `outputs/` paths noted. Then **`experiments/epic1/phase2_ssl/RESULTS.md`**: the four-method comparison table, the two break-it ablations, the SSL-vs-VAE probe comparison, and the answer to the phase question in your own words.

---

## Deliverables

- `experiments/epic1/phase2_ssl/` — shared substrate (backbone, augmentation, data, loop, eval harness) + SimCLR / SimSiam / VICReg heads and losses + the I-JEPA ViT build.
- Linear-probe + k-NN table: SimCLR vs SimSiam vs VICReg vs I-JEPA vs raw-pixel vs Phase 1 VAE.
- The two **break-it** ablations (SimSiam no-stop-grad; VICReg no-variance) with collapse curves.
- Collapse diagnostics (alignment/uniformity, per-dim std, effective rank) across the view-based methods.
- t-SNE/UMAP embedding scatters; an augmentation-sensitivity mini-sweep.
- Checkpoint series + `RESULTS.md`.

## Success criteria

Phase 2 is done when you can answer, from your own results:

1. Do contrastive, non-contrastive, and regularisation methods reach comparable representation quality (linear probe) despite entirely different anti-collapse mechanisms?
2. What does each mechanism's *failure* look like — and why do removing negatives, removing stop-gradient, and removing the variance term all cause collapse?
3. How does a label-free representation compare to the Phase 1 VAE on the shared probe axis, and how does the predictive (I-JEPA) route differ from the view-based ones?
4. Which augmentations carry the representation, and what invariance does each impose?

## Exit ramps (out of scope)

- **BYOL / DINO / Barlow Twins / MoCo implementations** — study only. BYOL is close to SimSiam+EMA (a small add-on if curious); DINO needs a ViT and is its own project; MoCo is SimCLR + a queue.
- **STL-10 / ImageNet-scale, LARS, 800-epoch runs** — CIFAR-10 at ~200 epochs shows every phenomenon; take one method longer only if you want a headline probe number.

## Resolved decisions (from the earlier open questions)

- **Build all four families** — SimCLR, SimSiam, VICReg, I-JEPA (one per mechanism).
- **I-JEPA is a full build** (Step 7), not a stretch.
- **~200 epochs per method** as the default budget.
- **Folder renamed** `phase2_simclr/` → `phase2_ssl/`.
