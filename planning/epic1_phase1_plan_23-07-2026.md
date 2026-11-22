# Phase 1 — Learn a Representation (Detailed Plan)

Detail for **Phase 1** of [`plan_12-07-2026.md`](epic1_plan_12-07-2026.md). Phases 1–4 are the top-level arc; the numbered **Steps** below are the units of work inside Phase 1.

**Phase question:** what information does a learned representation preserve, what does it discard, and what makes the result useful?

Steps 1–4 make that question *visible*. Steps 5–6 make it *measurable*. Don't stop at Step 4 — the pictures are seductive and the numbers are where the intuition actually consolidates.

## Compute

**This entire phase runs on the Mac.** A 784–400–L MLP VAE on MNIST is ~1M parameters and trains to convergence in a couple of minutes on MPS. No 4090 handoff is needed until the optional CIFAR-10 / CelebA extensions in "Exit ramps". Iterate locally at full speed.

---

## Step 1 — Baseline VAE

Architecture (MLP, per the sketch):

```
Encoder                  Decoder
784 → 400 → ReLU         L → 400 → ReLU → 784 → sigmoid
      ├── μ  (L)
      └── logvar (L)
```

Predict `logvar`, never `σ` directly — it keeps the variance positive without a constraint and is numerically stable. Sample via the reparameterization trick: `z = μ + σ ⊙ ε`, `ε ~ N(0, I)`.

Loss, per sample, **summed over the 784 pixels and only then averaged over the batch**:

```
L = BCE(x̂, x).sum(dim=pixels)  +  β · KL(q(z|x) ‖ N(0, I))
KL = -0.5 · Σ_j (1 + logvar_j - μ_j² - exp(logvar_j))
```

> **The trap that ruins most MNIST VAEs.** If you use `reduction='mean'` on the reconstruction term, you silently divide it by 784 and the KL term dominates by ~3 orders of magnitude. The model posterior-collapses, you get blurry digit-averages, and — worse for this phase — your β sweep in Step 6 measures nothing, because β is already effectively ~784. Sum over pixels, mean over batch. Sanity check: at init, reconstruction should be ~540 nats/image, KL ~0.

Binarize MNIST (or use the continuous values with BCE, which is technically a mismatch but standard). Pick one and stay consistent — the nats aren't comparable across the two.

Baseline config: `L=2`, `β=1`, Adam `1e-3`, batch 128, ~30 epochs. Fixed seed, config in a dataclass or YAML so Steps 5–6 can sweep it.

**Done when:** test ELBO curve is flat, reconstructions are recognizable digits.

---

## Step 2 — See the latent space

Encode the test set, plot `μ` (not sampled `z` — less noise) as a 2D scatter, colored by label.

Add one thing the source plan omits: **overlay the prior**. Draw the N(0, I) 1σ/2σ contours on the same axes. The gap between where the aggregate posterior `q(z) = E_x[q(z|x)]` actually sits and where the prior says it should is the *prior hole* problem, and it's the direct explanation for why some samples in Step 3 look like nothing at all.

**Expect:** clean clusters, with 4/9 and 3/5/8 overlapping. That overlap is signal, not failure.

---

## Step 3 — Decode the manifold

Grid-decode the 2D latent into a single large image tile.

**Don't use `linspace(-3, 3)`.** Uniform spacing in `z` oversamples the tails, where the model has seen almost no training mass, so the corners come out as garbage and you'll misread it as a broken model. Space the grid by the **inverse CDF of the prior** — `scipy.stats.norm.ppf(np.linspace(0.005, 0.995, n))` — so each cell covers equal prior probability.

**Expect:** smooth digit-to-digit morphs with no hard boundaries. This is the "manifold".

---

## Step 4 — Sample near a class

Compute `μ̄_c = mean of μ over test images of digit c`, then sample `z ~ N(μ̄_c, 0.1·I)` and decode.

Two corrections to the framing in the source plan:

- "You'll get mostly 3s" holds for well-separated classes and degrades badly for 3/5/8. Report the actual hit rate: classify the generated samples with a small MNIST CNN (or a logistic probe) and record the fraction matching the target class. A number beats an impression.
- **This is not conditional generation.** It's ancestral sampling from a crude Gaussian fit to an empirical class-conditional aggregate posterior. Real conditional generation conditions the encoder and decoder on the label (CVAE). Worth knowing the difference before Phase 3, where the same "is this direction a feature?" question returns in a much sharper form.

---

## Step 5 — Latent dimension sweep (the measurement run)

The master plan calls for `L ∈ {2, 8, 32, 128}` and the source plan drops it. Keep it — 2D is for *seeing*, the sweep is for *knowing*. 2 dimensions is a brutal bottleneck and is not a good representation; you cannot answer the phase question from it alone.

Train one model per `L`, at `β=1`, everything else fixed. Record per model:

| Metric | Meaning |
|---|---|
| Test ELBO, and its recon / KL split (nats/image) | total, plus the rate–distortion split |
| **Active units** | `# dims with Var_x(μ_j) > 0.01` (Burda et al.) — how many dims the model *actually uses* |
| **Per-dimension KL** `E_x[KL_j]`, sorted, as a bar chart | which dims carry information, and how much |
| **Linear probe accuracy** | logistic regression on `μ`, train split → test split |

The per-dimension KL chart is the single most informative plot in this phase. It shows the model allocating its rate budget across dimensions and then *refusing to use the rest* — you will see `L=128` light up ~10–20 dims and leave the remainder at exactly zero KL. That is the information bottleneck made visible, and it's the concrete answer to "what gets discarded".

Linear probe accuracy is deliberately the same protocol you'll use to evaluate SimCLR in Phase 2 — it makes a VAE latent and an SSL embedding directly comparable on one axis.

---

## Step 6 — β sweep and the rate–distortion curve

`β ∈ {0.1, 1, 5, 20}` (add `β=0` for the plain-autoencoder limit if you want the endpoint), at a fixed `L=32` so β has room to act. `L=2` is already so tight that the β effect is muddled.

Qualitatively you'll see the expected trade: small β → sharp reconstructions, unstructured latent; large β → clean isotropic latent, blurry outputs. Then make it quantitative:

**Plot rate vs distortion.** One point per β: `R = E_x[KL(q(z|x)‖p(z))]`, `D = -E[log p(x|z)]`, both in nats/image. The points trace a convex curve; `ELBO = D + R`, and β selects where on the curve you sit. This is the central figure of Alemi et al., *Fixing a Broken ELBO*, and it's where your Shannon refresh cashes out directly — `R` is literally the channel rate through the latent.

Overlay active-unit count per β. Watch it fall to zero as β grows: that's posterior collapse, and it's the same failure mode that shows up as dead features in Phase 3 SAEs.

Two knobs worth trying once collapse appears, since both recur later:
- **KL warmup** — ramp β linearly 0 → target over the first ~10 epochs.
- **Free bits** — `Σ_j max(KL_j, λ)` with `λ ≈ 0.5` nats, which stops the optimizer from zeroing a dimension for a trivial KL saving.

---

## Step 7 — Checkpoint for the longitudinal project

The stretch goal in the master plan is *"How do interpretable features emerge during training?"* — which means comparing features across **training checkpoints**. Phase 1 is where those checkpoints get created, and if you don't save them now you will have to retrain everything in Phase 3.

Cheap insurance, do it from the first run:

- Save at a log-spaced epoch schedule (`1, 2, 4, 8, 16, 32, …`) — feature formation is fast early and slow late, so uniform spacing wastes disk on the boring end.
- Save model state, optimizer state, epoch, RNG state, and the config that produced it.
- Fix and record seeds. A checkpoint you can't reproduce is a checkpoint you can't trust.
- Keep a frozen held-out batch of test images with the run, so activations extracted in Phase 3 are comparable across checkpoints.
- `outputs/` is gitignored — these live on disk only, so note the path in the run log.

---

## Deliverables

- `experiments/epic1/phase1_vae/` — model, train loop, sweep driver, plotting
- Latent scatter (2D, prior contours overlaid)
- Manifold grid tile (inverse-CDF spacing)
- Class-conditional sample grids + hit-rate table
- Per-dimension KL bar charts across the `L` sweep
- Rate–distortion curve across the β sweep
- Checkpoint series for the longitudinal project
- A short `RESULTS.md` — the numbers, and the answer to the phase question in your own words

## Success criteria

Not "the code runs". Phase 1 is done when you can answer, from your own plots:

1. How many nats does the model push through the latent, and how are they distributed across dimensions?
2. Why does `L=128` not beat `L=32` on the probe, given it has 4× the capacity?
3. What does β actually control, in rate–distortion terms?
4. What did the representation throw away, and was that the right call?

## Exit ramps (out of scope for Phase 1)

The source plan's progression — Fashion-MNIST → CIFAR-10 → CelebA → CVAE → VQ-VAE → Latent Diffusion — is a good *generative modeling* path, and it is not the path in the master plan. Your stated goal is representation learning → mechanistic interpretability, and Phases 2–4 go SimCLR → SAE → TransformerLens. Don't let the LDM branch displace them.

Worth taking cheaply:
- **Fashion-MNIST** — same code, one flag, and the harder classes make the bottleneck story sharper.

Worth deferring until after Phase 4, if at all:
- **VQ-VAE** — genuinely relevant, since discrete codes are a direct cousin of SAE dictionaries, and it explains why the field moved off pure Gaussian latents. Best read *after* Phase 3, when you have the sparse-dictionary frame to compare it against.
- **CIFAR-10 / CelebA / Latent Diffusion** — needs conv architectures and the 4090; a different project, not a continuation of this one.

## Open questions

- Binarized vs continuous MNIST for the headline numbers — pick one before Step 5, since the nats aren't comparable across them.
- Whether to add a small conv encoder/decoder variant, or keep MLP-only for phase-wide comparability.
