"""E5: can training make the own-answer report read what the model knows? (4090, HF + peft)

Trains small LoRAs on the E2 "your answer" conversation (Phase 1's prompt, the model's own
answer in the assistant turn, then "What is the probability, from 0 to 1, that your answer
is correct?") over the training questions, and evaluates the reported confidence on the
test questions. Variants (all on the same class-balanced inputs; only the targets differ):

  per_answer   target {"confidence": 1} if the answer was right (Phase 1's judged label),
               {"confidence": 0} if wrong
  lin          (control) target = the model's training accuracy in the answer's stratum
               (kind x level x family), rounded to 0.1, as {"confidence": 0.3}: the coarse,
               which-level-is-hard signal Lin et al. trained on
  shuffled     (control) per-answer targets shuffled
  loko_<kind>  per-answer, trained without <kind>; tested on all (transfer to the held-out kind)

LoRA: rank 16, alpha 32, dropout 0.05, on q/k/v/o and the MLP projections; loss on the reply
tokens only (the output layer is applied at those positions alone); 2 epochs, learning rate
1e-4, batches of up to 16 answers grouped by length within 8,192 padded tokens, bf16. Inputs: the training own answers
from swaps.jsonl; the minority class (right answers) is oversampled to match the majority.

Readout, for the untrained model ("base") and every LoRA, on every test answer of E2 (own,
correct, wrong) under both attributions ("self" = your answer, "user" = my answer):
  ev      the expected value of the reported number: each candidate reply {"confidence": v}
          (v in 0, 1, 0.0 ... 1.0, 0.95, 0.99) scored by teacher forcing (reply tokenised on
          its own, as the model writes it), probabilities renormalised over the candidates,
          EV = sum p(v) v; cand_mass = the candidates' total probability (a check that the
          model answers in the format)
  greedy  the greedy reply (up to 12 tokens), parsed with swap_task's parser

Writes results/e5_scores_<config>.parquet: variant, qid, category, level, family, source,
correct, attribution, ev, cand_mass, greedy, reply. Adapters go to lora_runs/ (gitignored).

Usage (repo root, WSL):
  uv run --extra evals --extra interp python experiments/epic2/phase2_confidence_gap/lora_confidence.py qwen3-1.7b --limit 64
  uv run --extra evals --extra interp python experiments/epic2/phase2_confidence_gap/lora_confidence.py qwen3-1.7b
  ... qwen3-1.7b --reuse   after an interrupted run: variants whose adapter is already in lora_runs/
                           are loaded and scored, not retrained
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import phase1
import steer
import swap_task as st

KINDS = ("recall", "computation", "perception")
VARIANTS = ("per_answer", "lin", "shuffled", *(f"loko_{k}" for k in KINDS))
CANDIDATES = ("0", "1", *(f"{i / 10:.1f}" for i in range(11)), "0.95", "0.99")
RUNS = phase1.HERE / "lora_runs"
SEED = 0
EPOCHS, LR, BATCH, RANK = 2, 1e-4, 16, 16
TOKEN_BUDGET = 8192  # tokens per batch (padded): a few own answers run to ~1,000 tokens


def reply(v: str) -> str:
    return '{"confidence": ' + v + "}"


# ----------------------------------------------------------------------------- data


def questions() -> dict[str, str]:
    out = {}
    for f in ("questions.jsonl", "train_questions.jsonl"):
        for line in (phase1.PHASE1_QUESTIONS.parent / f).read_text().splitlines():
            q = json.loads(line)
            out[q["id"]] = q["question"]
    return out


def training_set(config: str, limit: int | None) -> pd.DataFrame:
    sw = pd.read_json(phase1.DATA / "swaps.jsonl", lines=True)
    tr = sw[(sw.config == config) & (sw.split == "train") & (sw.source == "own")].copy()
    tr["stratum"] = tr.category + ":" + tr.level.astype(str) + ":" + tr.family.fillna("")
    acc = tr.groupby("stratum").correct.mean()
    tr["lin"] = tr.stratum.map(acc).round(1)
    right, wrong = tr[tr.correct == 1], tr[tr.correct == 0]
    big, small = (wrong, right) if len(wrong) >= len(right) else (right, wrong)
    bal = pd.concat([big, small.sample(len(big), replace=True, random_state=SEED)]).sample(frac=1, random_state=SEED)
    bal["shuffled"] = np.random.default_rng(SEED).permutation(bal.correct.to_numpy())
    if limit:
        bal = bal.groupby(["category", "correct"]).head(limit // 6).sample(frac=1, random_state=SEED)
    return bal.reset_index(drop=True)


def targets(bal: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, list[str]]:
    if variant.startswith("loko_"):
        keep = bal[bal.category != variant[5:]]
        return keep, [reply(str(int(c))) for c in keep.correct]
    if variant == "per_answer":
        return bal, [reply(str(int(c))) for c in bal.correct]
    if variant == "shuffled":
        return bal, [reply(str(int(c))) for c in bal.shuffled]
    if variant == "lin":
        return bal, [reply(f"{v:.1f}") for v in bal["lin"]]
    raise ValueError(variant)


def test_set(config: str, limit: int | None) -> pd.DataFrame:
    sw = pd.read_json(phase1.DATA / "swaps.jsonl", lines=True)
    t = sw[(sw.config == config) & (sw.split == "test")].copy()
    if limit:
        t = t.groupby(["category", "source"]).head(max(limit // 9, 2))
    return t.reset_index(drop=True)


def prompt_text(tok, q: str, answer: str, attribution: str) -> str:
    return tok.apply_chat_template(steer.messages(q, answer, attribution), tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


# ----------------------------------------------------------------------------- training


def encode_train(tok, texts: list[str], replies: list[str]) -> list[dict]:
    end = tok("<|im_end|>", add_special_tokens=False)["input_ids"]
    out = []
    for t, r in zip(texts, replies, strict=True):
        p = tok(t, add_special_tokens=False)["input_ids"]
        a = tok(r, add_special_tokens=False)["input_ids"] + end
        out.append({"ids": p + a, "labels": [-100] * len(p) + a})
    return out


def length_batches(lengths: list[int], max_count: int, budget: int) -> list[list[int]]:
    """Indices grouped by length, each batch at most ``max_count`` rows and ``budget`` padded
    tokens (a single longer row still gets its own batch)."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, cur = [], []
    for i in order:
        if cur and (len(cur) + 1 > max_count or (len(cur) + 1) * lengths[i] > budget):
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


def reply_logits(model, ids, mask, positions):
    """Logits only at the given (row, position) pairs: the decoder's hidden states, then the
    output layer on those positions alone (full-vocabulary logits everywhere do not fit)."""
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = inner.model(input_ids=ids, attention_mask=mask).last_hidden_state
    return inner.lm_head(hidden[positions[0], positions[1]])


def train(model, tok, items: list[dict], device: str, max_steps: int | None) -> None:
    import torch

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    batches = length_batches([len(x["ids"]) for x in items], BATCH, TOKEN_BUDGET)
    n_steps = max_steps or EPOCHS * len(batches)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / n_steps))
    rng = np.random.default_rng(SEED)
    model.train()
    step, t0 = 0, time.time()
    while step < n_steps:
        for bi in rng.permutation(len(batches)):
            if step >= n_steps:
                break
            batch = [items[i] for i in batches[bi]]
            width = max(len(x["ids"]) for x in batch)
            ids = torch.full((len(batch), width), tok.pad_token_id, dtype=torch.long)
            lab = torch.full((len(batch), width), -100, dtype=torch.long)
            mask = torch.zeros((len(batch), width), dtype=torch.long)
            for r, x in enumerate(batch):
                ids[r, :len(x["ids"])] = torch.tensor(x["ids"])
                lab[r, :len(x["labels"])] = torch.tensor(x["labels"])
                mask[r, :len(x["ids"])] = 1
            # next-token loss on the reply tokens only: position t predicts token t + 1
            tgt = lab[:, 1:]
            rows, cols = torch.nonzero(tgt != -100, as_tuple=True)
            logits = reply_logits(model, ids.to(device), mask.to(device), (rows.to(device), cols.to(device)))
            loss = torch.nn.functional.cross_entropy(logits.float(), tgt[rows, cols].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 50 == 0 or step == n_steps:
                print(f"    step {step}/{n_steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.eval()


# ----------------------------------------------------------------------------- readout


def candidate_ev(model, tok, texts: list[str], device: str, batch_size: int) -> np.ndarray:
    """[n, 2]: expected value over CANDIDATES and the candidates' total probability."""
    import torch

    cand_ids = [tok(reply(c), add_special_tokens=False)["input_ids"] for c in CANDIDATES]
    vals = np.array([float(c) for c in CANDIDATES])
    prompts = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    seqs = [(i, j, p + c, len(p), len(c)) for i, p in enumerate(prompts) for j, c in enumerate(cand_ids)]
    lp = np.zeros((len(texts), len(CANDIDATES)))
    for b in length_batches([len(s[2]) for s in seqs], batch_size * len(CANDIDATES), TOKEN_BUDGET * 2):
        chunk = [seqs[k] for k in b]
        width = max(len(s[2]) for s in chunk)
        ids = torch.full((len(chunk), width), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        rows, cols, tgt = [], [], []
        for r, (_, _, s_, p, n) in enumerate(chunk):
            ids[r, :len(s_)] = torch.tensor(s_)
            mask[r, :len(s_)] = 1
            rows += [r] * n
            cols += list(range(p - 1, p + n - 1))
            tgt += s_[p:p + n]
        with torch.no_grad():
            logits = reply_logits(model, ids.to(device), mask.to(device),
                                  (torch.tensor(rows, device=device), torch.tensor(cols, device=device)))
            tok_lp = torch.log_softmax(logits.float(), -1).gather(1, torch.tensor(tgt, device=device)[:, None])
        tok_lp = tok_lp.squeeze(1).cpu().numpy()
        pos = 0
        for (i, j, _, _, n) in chunk:
            lp[i, j] = tok_lp[pos:pos + n].sum()
            pos += n
    out = np.zeros((len(texts), 2))
    for i in range(len(texts)):
        w = np.exp(lp[i] - lp[i].max())
        out[i] = (float((w * vals).sum() / w.sum()), float(np.exp(lp[i]).sum()))
    return out


def greedy(model, tok, texts: list[str], device: str, batch_size: int) -> list[str]:
    import torch

    tok.padding_side = "left"
    res = [""] * len(texts)
    lengths = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
    for b in length_batches(lengths, batch_size, TOKEN_BUDGET):
        enc = tok([texts[i] for i in b], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id)
        for i, x in zip(b, tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True), strict=True):
            res[i] = x
    tok.padding_side = "right"
    return res


def evaluate(model, tok, t: pd.DataFrame, qs: dict, variant: str, device: str, batch_size: int) -> pd.DataFrame:
    rows = []
    for att in ("self", "user"):
        texts = [prompt_text(tok, qs[r.qid], r.text, att) for r in t.itertuples()]
        ev = candidate_ev(model, tok, texts, device, batch_size)
        gr = greedy(model, tok, texts, device, batch_size * 2)
        rows.append(t[["qid", "category", "level", "family", "source", "correct"]].assign(
            variant=variant, attribution=att, ev=ev[:, 0], cand_mass=ev[:, 1],
            greedy=[st.parse_confidence(x) for x in gr], reply=[x[:60] for x in gr]))
    return pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", choices=["qwen3-1.7b", "qwen3-8b"])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--limit", type=int, help="smoke test: small training and test sets, 30 steps; nothing saved")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--reuse", action="store_true", help="load adapters already saved in lora_runs/ instead of retraining")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    name, _ = phase1.extract.CONFIGS[args.config]
    tok = AutoTokenizer.from_pretrained(name)
    qs = questions()
    bal = training_set(args.config, args.limit)
    test = test_set(args.config, args.limit)
    print(f"=== {args.config}: {len(bal)} balanced training rows "
          f"({int(bal.correct.sum())} right), {len(test)} test rows {time.strftime('%H:%M')}")
    results = []

    def fresh():
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map=device)
        m.config.use_cache = False
        return m

    base = fresh()
    base.eval()
    results.append(evaluate(base, tok, test, qs, "base", device, args.batch_size))
    print("  base evaluated; candidate mass median "
          f"{results[-1].cand_mass.median():.3f}; e.g. {results[-1].reply.iloc[0]!r}")
    del base
    torch.cuda.empty_cache()

    for variant in args.variants:
        rows, reps = targets(bal, variant)
        texts = [prompt_text(tok, qs[r.qid], r.text, "self") for r in rows.itertuples()]
        items = encode_train(tok, texts, reps)
        print(f"  --- {variant}: {len(items)} training examples, e.g. target {reps[0]!r}")
        saved = RUNS / args.config / variant
        reused = args.reuse and not args.limit and (saved / "adapter_config.json").exists()
        if reused:
            from peft import PeftModel
            print(f"    reusing the adapter in {saved}")
            model = PeftModel.from_pretrained(fresh(), saved)
            model.eval()
        else:
            model = get_peft_model(fresh(), LoraConfig(
                r=RANK, lora_alpha=2 * RANK, lora_dropout=0.05, task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
            train(model, tok, items, device, 30 if args.limit else None)
        model.config.use_cache = True
        res = evaluate(model, tok, test, qs, variant, device, args.batch_size)
        results.append(res)
        own = res[(res.source == "own") & (res.attribution == "self")]
        print(f"    {variant}: own answers, mean EV right {own[own.correct == 1].ev.mean():.3f} vs wrong "
              f"{own[own.correct == 0].ev.mean():.3f}; candidate mass median {res.cand_mass.median():.3f}; "
              f"e.g. {res.reply.iloc[0]!r}")
        if not args.limit and not reused:
            model.save_pretrained(saved)
        del model
        torch.cuda.empty_cache()

    out = pd.concat(results, ignore_index=True)
    if args.limit:
        print(out.groupby(["variant", "attribution", "source"]).ev.mean().unstack().round(3).to_string())
    else:
        out.to_parquet(phase1.RESULTS / f"e5_scores_{args.config}.parquet", index=False)
        print(f"wrote results/e5_scores_{args.config}.parquet")


if __name__ == "__main__":
    main()
