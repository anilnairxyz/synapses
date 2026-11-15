"""E4 Part B: steering along the correctness direction (4090, HF transformers).

Does pushing the model's internal "this answer is right" signal change the confidence it
reports, and does that depend on whose answer it is?

For one instruct config and a balanced sample of the E2 answers (own, correct, wrong; each
kind), the E2 conversations are rerun with a vector added to the residual stream after the
decoder layer whose output the Phase 1B probe reads (hidden_states[layer]), at every
position, during the prompt and the reply:

  direction  "probe"   the correctness probe's unit direction for the question's kind
                       (directions.py, results/e4_probe_dirs.npz)
             "random"  a random unit direction (fixed seed), the control
  strength   alpha x s, alpha in ALPHAS, s = the standard deviation of the E2 answer states'
             projection on the probe direction at that layer (so alpha = 1 moves a state by
             one typical spread along the correctness axis); the random direction gets the
             same absolute size
  attribution  "self" (the answer in the assistant turn, "your answer") or "user" ("my
             answer"), exactly as in swap_task.py

Greedy decoding, thinking off, up to 24 new tokens; the confidence is parsed with
swap_task's parser. alpha = 0 reproduces the unsteered E2 confidences (checked: the share
of rows equal to the swap logs' values is printed; small differences come from vLLM vs HF
kernels).

Writes results/e4_steer_<config>.parquet: qid, category, source, correct, attribution,
direction, alpha, confidence.

Prediction (plan, before the run): steering along the probe direction moves "my answer"
confidence in the direction of the push, more than the random control, while "your answer"
barely moves (< 0.05 where "my answer" moves by 0.1 or more).

Usage (repo root, WSL; needs states/<config>/swaps.npy from extract_answers.py):
  uv run --extra evals --extra interp python experiments/epic2/phase2_confidence_gap/steer.py qwen3-8b --limit 4
  uv run --extra evals --extra interp python experiments/epic2/phase2_confidence_gap/steer.py qwen3-8b qwen3-1.7b
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import analyze_swaps as an
import phase1
import swap_task as st

ALPHAS = (-4.0, -2.0, 0.0, 2.0, 4.0)
DIRECTIONS = ("probe", "random")
PER_CELL = 50  # answers per kind x source: 3 x 3 x 50 = 450 rows, about 300 to 450 answers
KINDS = ("recall", "computation", "perception")
STATES = phase1.HERE / "states"
SEED = 0
MAX_NEW = 24


def sample_rows(config: str, per_cell: int) -> pd.DataFrame:
    meta = pd.read_parquet(STATES / config / "swaps.parquet")  # row order of swaps.npy
    sw = pd.read_json(phase1.DATA / "swaps.jsonl", lines=True)
    sw = sw[(sw.config == config) & (sw.split == "test")][["qid", "source", "text", "correct"]]
    d = meta.reset_index().merge(sw, on=["qid", "source"], how="left", validate="one_to_one")
    return (d.groupby(["category", "source"], group_keys=False)
             .apply(lambda g: g.sample(min(per_cell, len(g)), random_state=SEED))
             .reset_index(drop=True))


def messages(q: str, answer: str, attribution: str) -> list[dict]:
    """The same conversations as swap_task.py, as plain dicts for the HF chat template."""
    return [{"role": "assistant" if type(m).__name__ == "ChatMessageAssistant" else "user", "content": m.content}
            for m in st.messages(q, answer, attribution)]


def steer_vectors(config: str, d: pd.DataFrame) -> dict[str, dict]:
    """Per kind: layer, unit probe direction, random unit direction, and the step size s."""
    dirs = np.load(phase1.RESULTS / "e4_probe_dirs.npz")
    states = np.load(STATES / config / "swaps.npy", mmap_mode="r")
    pi = list(phase1.probe.POSITIONS).index("ans_last")
    meta = pd.read_parquet(STATES / config / "swaps.parquet")
    rng = np.random.default_rng(SEED)
    out = {}
    for kind in KINDS:
        layer = int(dirs[f"{config}|{kind}|layer"])
        w = dirs[f"{config}|{kind}|dir"].astype(np.float32)
        X = np.asarray(states[np.flatnonzero((meta.category == kind).to_numpy()), layer, pi, :], dtype=np.float32)
        s = float((X @ w).std())
        r = rng.normal(size=w.shape).astype(np.float32)
        out[kind] = {"layer": layer, "probe": w, "random": r / np.linalg.norm(r), "s": s}
        print(f"  {kind}: layer {layer}, step s = {s:.2f} (median state norm {np.median(np.linalg.norm(X, axis=1)):.1f})")
    return out


def generate(model, tok, texts: list[str], vec, layer: int, batch_size: int, device: str) -> list[str]:
    """Greedy replies with ``vec`` (or None) added to hidden_states[layer] at every position."""
    import torch

    # hidden_states[layer] is the output of decoder layer `layer`; the last entry is taken after
    # the final norm (HF Qwen3), so a probe on the last layer is steered after model.model.norm
    n = len(model.model.layers)
    block = model.model.norm if layer == n else model.model.layers[layer - 1]
    add = None if vec is None else torch.tensor(vec, device=device, dtype=model.dtype)

    def hook(_m, _inp, out):
        if add is None:
            return out
        if isinstance(out, tuple):
            return (out[0] + add,) + tuple(out[1:])
        return out + add

    h = block.register_forward_hook(hook)
    tok.padding_side = "left"
    replies = []
    try:
        for b in range(0, len(texts), batch_size):
            enc = tok(texts[b:b + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
            replies += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    finally:
        h.remove()
    return replies


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+", choices=["qwen3-8b", "qwen3-1.7b"])
    ap.add_argument("--limit", type=int, help="smoke test: N answers per kind x source; nothing saved")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    qs = {q["id"]: q for q in map(json.loads, phase1.PHASE1_QUESTIONS.read_text().splitlines())}
    for config in args.configs:
        name, _ = phase1.extract.CONFIGS[config]
        print(f"=== {config} ({name}) {time.strftime('%H:%M')}")
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map=device)
        model.eval()
        d = sample_rows(config, args.limit or PER_CELL)
        vecs = steer_vectors(config, d)
        out = []
        for att in ("self", "user"):
            for kind in KINDS:
                k = d[d.category == kind]
                texts = [tok.apply_chat_template(messages(qs[r.qid]["question"], r.text, att), tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)
                         for r in k.itertuples()]
                v = vecs[kind]
                for direction in DIRECTIONS:
                    for alpha in ALPHAS:
                        if alpha == 0 and direction == "random":
                            continue  # the same as probe at 0
                        vec = None if alpha == 0 else alpha * v["s"] * v[direction]
                        t0 = time.time()
                        replies = generate(model, tok, texts, vec, v["layer"], args.batch_size, device)
                        conf = [st.parse_confidence(x) for x in replies]
                        for r, c, x in zip(k.itertuples(), conf, replies, strict=True):
                            out.append({"qid": r.qid, "category": kind, "source": r.source, "attribution": att,
                                        "direction": direction, "alpha": alpha, "confidence": c, "reply": x[:80]})
                        if args.limit:
                            print(f"  {att} {kind} {direction} {alpha:+.0f}: {time.time() - t0:.0f}s, "
                                  f"parsed {np.mean([c is not None for c in conf]):.2f}, "
                                  f"mean {np.nanmean([np.nan if c is None else c for c in conf]):.2f}; "
                                  f"e.g. {replies[0][:40]!r}")
        res = pd.DataFrame(out).merge(d[["qid", "source", "correct"]], on=["qid", "source"], how="left")
        # alpha = 0 against the unsteered E2 (vLLM) confidences
        e2 = an.swaps_verbal(config).melt(id_vars=["qid", "source"], var_name="att", value_name="e2")
        e2["attribution"] = e2.att.str.replace("verbal_", "")
        z = res[res.alpha == 0].merge(e2[["qid", "source", "attribution", "e2"]], on=["qid", "source", "attribution"])
        print(f"  alpha 0 equals the E2 confidence on {(np.isclose(z.confidence, z.e2)).mean():.0%} of {len(z)}")
        print(res.groupby(["attribution", "direction", "alpha"]).confidence.mean().unstack().round(3).to_string())
        if not args.limit:
            res.to_parquet(phase1.RESULTS / f"e4_steer_{config}.parquet", index=False)
            print(f"  wrote results/e4_steer_{config}.parquet")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
