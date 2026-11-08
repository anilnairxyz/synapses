"""Phase 1B step 4: hidden states before and after each logged plain answer (4090 only).

For every question a config answered with method P, rebuild the exact P prompt (the
model's chat template with thinking off, or the raw few-shot prompt for the base model),
append the logged answer's tokens (teacher forcing, so the states belong to exactly the
answer that was graded), run one forward pass and keep, at every layer (embeddings = 0):

  q_last    last token of the question text (Ni et al.'s Pre-State)
  gen       last prompt token, from which the first answer token is predicted
  ans_last  last answer token (Ni et al.'s Last State)
  ans_mean  mean over the answer tokens (Ni et al.'s Avg State)

Output, per config and split, under states/<config>/ (gitignored, large):
  <split>.npy        float16 [n_questions, n_layers + 1, 4, hidden]
  <split>.parquet    qid, category, level, answer, n_prompt, n_answer (row order of the .npy)

Splits: "test" = the Phase 1 full run (answerable + unknowable + ill-posed; base model:
answerable only), "train" = the Phase 1B training run (runs/4090/train/). Answers that are
empty (failed calls) get zero states and n_answer = 0; the probe drops them.

--limit N is the smoke test: only the first N questions per split, nothing saved unless
--save, and a greedy regeneration check (the HF answer should match the logged vLLM
answer on nearly every item; small differences come from bf16 kernels, not the prompt).

Usage (from the repo root, in WSL):
  uv run --extra evals --extra interp python experiments/epic2/phase1_confidence/extract_states.py \
      qwen3-8b --limit 20
  uv run --extra evals --extra interp python experiments/epic2/phase1_confidence/extract_states.py \
      qwen3-8b qwen3-1.7b qwen3-8b-base
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import confidence_task as ct  # noqa: E402

RUNS = HERE / "runs" / "4090"
STATES = HERE / "states"
POSITIONS = ("q_last", "gen", "ans_last", "ans_mean")
SPLITS = {"test": "full", "train": "train"}
CONFIGS = {  # config -> (HF model, base?)
    "qwen3-8b": ("Qwen/Qwen3-8B", False),
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", False),
    "qwen3-8b-base": ("Qwen/Qwen3-8B-Base", True),
}
KEEP = ("recall", "computation", "perception", "ill_posed", "unknowable")


def logged_answers(config: str, split: str) -> list[dict]:
    """(qid, question record, logged P answer) for one config's run, in log order."""
    from inspect_ai.log import read_eval_log

    files = sorted(glob.glob(str(RUNS / SPLITS[split] / config / "*.eval")))
    if len(files) != 1:
        raise SystemExit(f"expected one log in runs/4090/{SPLITS[split]}/{config}, found {len(files)}")
    log = read_eval_log(files[0])
    out = []
    for s in log.samples:
        q = s.metadata["q"]
        if q["category"] in KEEP:
            out.append({"qid": s.id, "q": q, "answer": s.metadata["calls"]["P"].get("answer") or ""})
    return out


def build_prompt(tok, q: dict, base: bool) -> tuple[str, int]:
    """The P prompt text as vLLM saw it, and the character offset where the question ends."""
    if base:
        text = ct.FEW_SHOT.format(q=q["question"])
    else:
        user = ct.PROMPT_P.format(q=q["question"])
        text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    end = text.rindex(q["question"]) + len(q["question"])
    return text, end


def encode(tok, q: dict, answer: str, base: bool) -> dict:
    text, q_end = build_prompt(tok, q, base)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    q_last = max(i for i, (s, _) in enumerate(offsets) if s < q_end)
    # The base model's answers follow "A:" with a space; the instruct answers start fresh
    # after the assistant header. Tokenised separately, as generation produced them.
    ans_ids = tok(((" " + answer) if base else answer), add_special_tokens=False)["input_ids"] if answer else []
    return {"ids": ids + ans_ids, "n_prompt": len(ids), "n_answer": len(ans_ids), "q_last": q_last,
            "text": text}


def extract(model, items: list[dict], batch_size: int, device: str, path: Path | None = None):
    """[n, layers + 1, 4, hidden] float16 states for encoded items (written to ``path`` as a
    memory-mapped .npy if given, so the full array never has to fit in RAM)."""
    import torch

    shape = (len(items), model.config.num_hidden_layers + 1, len(POSITIONS),
             model.config.hidden_size)
    if path is None:
        out = np.zeros(shape, dtype=np.float16)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))  # less padding
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        width = max(len(items[i]["ids"]) for i in idx)
        ids = torch.zeros((len(idx), width), dtype=torch.long)
        mask = torch.zeros((len(idx), width), dtype=torch.long)
        for r, i in enumerate(idx):  # right padding: positions stay as in the unpadded input
            seq = items[i]["ids"]
            ids[r, :len(seq)] = torch.tensor(seq)
            mask[r, :len(seq)] = 1
        with torch.no_grad():
            hs = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                       output_hidden_states=True).hidden_states
        hs = torch.stack(hs, dim=1).float()  # [batch, layers + 1, seq, hidden]
        for r, i in enumerate(idx):
            it = items[i]
            p, a = it["n_prompt"], it["n_answer"]
            vecs = [hs[r, :, it["q_last"]], hs[r, :, p - 1]]
            if a:
                vecs += [hs[r, :, p + a - 1], hs[r, :, p:p + a].mean(dim=1)]
            else:
                vecs += [torch.zeros_like(vecs[0])] * 2
            out[i] = torch.stack(vecs, dim=1).cpu().numpy().astype(np.float16)
    return out


def regenerate_check(model, tok, items: list[dict], answers: list[str], base: bool, device: str) -> None:
    """Greedy HF answers vs the logged vLLM answers (smoke test only)."""
    import torch

    same = 0
    for it, logged in zip(items, answers, strict=True):
        ids = torch.tensor([it["ids"][:it["n_prompt"]]], device=device)
        with torch.no_grad():
            gen = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=48,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        text = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        text = (text.split("\n")[0] if base else text).strip()
        same += text == logged.strip()
        if text != logged.strip():
            print(f"    differs: logged {logged.strip()[:60]!r} | HF {text[:60]!r}")
    print(f"  greedy regeneration matches the logged answer on {same}/{len(items)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+", choices=sorted(CONFIGS))
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    ap.add_argument("--limit", type=int, help="smoke test: first N questions per split")
    ap.add_argument("--save", action="store_true", help="save even with --limit")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    for config in args.configs:
        name, base = CONFIGS[config]
        print(f"=== {config} ({name}) {time.strftime('%H:%M')}")
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map=device)
        model.eval()
        for split in args.splits:
            rows = logged_answers(config, split)
            if args.limit:  # a spread over categories, not just the first category
                step = max(len(rows) // args.limit, 1)
                rows = rows[::step][:args.limit]
            items = [encode(tok, r["q"], r["answer"], base) for r in rows]
            print(f"  {split}: {len(items)} questions, prompt tokens "
                  f"{min(i['n_prompt'] for i in items)}-{max(i['n_prompt'] for i in items)}, "
                  f"{sum(i['n_answer'] == 0 for i in items)} empty answers")
            if args.limit:
                ex = items[0]
                print("  first prompt (repr):", repr(ex["text"][-160:]))
                print("  q_last token:", repr(tok.decode(ex["ids"][ex["q_last"]])),
                      "| gen token:", repr(tok.decode(ex["ids"][ex["n_prompt"] - 1])),
                      "| answer tokens:", repr(tok.decode(ex["ids"][ex["n_prompt"]:])))
                regenerate_check(model, tok, items, [r["answer"] for r in rows], base, device)
            save = not args.limit or args.save
            out = STATES / config
            t0 = time.time()
            states = extract(model, items, args.batch_size, device,
                             out / f"{split}.npy" if save else None)
            print(f"  states {states.shape} in {time.time() - t0:.0f}s; "
                  f"finite: {bool(np.isfinite(states).all())}")
            if not save:
                continue
            states.flush()
            pd.DataFrame({
                "qid": [r["qid"] for r in rows], "category": [r["q"]["category"] for r in rows],
                "level": [r["q"]["level"] for r in rows], "answer": [r["answer"] for r in rows],
                "n_prompt": [i["n_prompt"] for i in items], "n_answer": [i["n_answer"] for i in items],
            }).to_parquet(out / f"{split}.parquet", index=False)
            print(f"  saved {out / split}.npy")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
