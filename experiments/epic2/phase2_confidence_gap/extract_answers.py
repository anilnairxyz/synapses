"""E2 / E3 on the 4090: states, log-probability and P(True) for given answers; probe scores.

For one config, every row of swaps.jsonl (build_swaps.py):

1. States (test rows): Phase 1's plain-answer prompt plus the given answer, teacher-forced,
   one forward pass; the four Phase 1B positions at every layer (Phase 1's encode() and
   extract(), unchanged). Saved under states/<config>/ (gitignored).
2. Answer log-probability (test rows): sum and mean over the answer's tokens of
   log P(token | prompt, earlier answer tokens): how likely the model finds the given
   answer. The familiarity control for E2 (Wataoka et al.).
3. P(True) (all rows, test and train), Kadavath et al.'s prompt:

       Question: <q>
       Proposed Answer: <answer>
       Is the proposed answer:
       (A) True
       (B) False
       The proposed answer is:

   For the instruct models it is the user's message and the assistant's reply is started
   with "(", so the next token is the letter; for the base model the raw text continues
   " (". P(True) = P("A") / (P("A") + P("B")) at that position, one forward pass; the share
   of all probability on "A" or "B" is kept as a check. Variants:
     placement  user       the block alone (the neutral, observer framing)
                assistant  (instruct only) Phase 1's prompt, the answer as the assistant's
                           reply, then the block: the answer is the model's own
     shots      0, or K    K worked examples before the block (Kadavath's few-shot), drawn
                           from this config's own TRAINING answers with Phase 1's labels,
                           half right and half wrong, spread over the kinds. Those rows are
                           flagged is_shot and dropped from analysis.
4. Probe scores (test rows): the Phase 1B probe for the row's kind, refitted on the Phase 1
   training states at the layer and C Phase 1B chose (positions gen and ans_last), plus
   the same probe at layer 0 (ans_last; it sees only the answer's token embeddings). Check:
   on own answers the states are the Phase 1 test states, so the scores must reproduce
   Phase 1B's test scores (printed as a correlation; should be 1.000).

Writes results/extract_<config>.parquet (committed; small): one row per swaps row with
logprob_sum, logprob_mean, n_answer_tokens, ptrue_<placement>_<shots>, ab_mass_<...>,
is_shot, probe_gen, probe_ans_last, probe_l0_ans_last.

Usage (repo root, WSL):
  uv run --extra evals --extra interp python experiments/epic2/phase2_confidence_gap/extract_answers.py qwen3-8b --limit 6
  uv run --extra evals --extra interp python experiments/epic2/phase2_confidence_gap/extract_answers.py qwen3-8b qwen3-1.7b qwen3-8b-base
"""

from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import pandas as pd

import phase1

KADAVATH = ("Question: {q}\nProposed Answer: {a}\nIs the proposed answer:\n(A) True\n(B) False\n"
            "The proposed answer is:")
K_SHOTS = 8
PLACEMENTS = ("user", "assistant")
STATES = phase1.HERE / "states"
SEED = 0


def block(q: str, a: str) -> str:
    return KADAVATH.replace("{q}", q).replace("{a}", a)


# ----------------------------------------------------------------------------- data


def load_rows(config: str, limit: int | None) -> pd.DataFrame:
    d = pd.read_json(phase1.DATA / "swaps.jsonl", lines=True)
    d = d[d.config == config].reset_index(drop=True)
    if limit:
        d = d.groupby(["split", "source", "category"], group_keys=False).head(limit).reset_index(drop=True)
    return d


def questions() -> dict[str, dict]:
    out = {}
    for f in ("questions.jsonl", "train_questions.jsonl"):
        for line in (phase1.PHASE1_QUESTIONS.parent / f).read_text().splitlines():
            q = json.loads(line)
            out[q["id"]] = q
    return out


def choose_shots(rows: pd.DataFrame, qs: dict[str, dict]) -> list[dict]:
    """K training own answers: half right, half wrong, alternating kinds; fixed seed."""
    tr = rows[(rows.split == "train") & (rows.source == "own")]
    rng = random.Random(SEED)
    picked = []
    for ok in (1, 0):
        for i in range(K_SHOTS // 2):
            kind = ("recall", "computation", "perception")[i % 3]
            pool = tr[(tr.correct == ok) & (tr.category == kind)]
            pool = pool[~pool.qid.isin([p["qid"] for p in picked])]
            r = pool.iloc[rng.randrange(len(pool))]
            picked.append({"qid": r.qid, "q": qs[r.qid]["question"], "a": r.text, "ok": ok})
    rng.shuffle(picked)
    return picked


def shots_text(shots: list[dict]) -> str:
    return "".join(block(s["q"], s["a"]) + (" (A)" if s["ok"] else " (B)") + "\n\n" for s in shots)


# ----------------------------------------------------------------------------- P(True)


def ptrue_text(tok, q: str, a: str, placement: str, shots: str, base: bool) -> str:
    content = shots + block(q, a)
    if base:
        return content + " ("
    msgs = [{"role": "user", "content": content}]
    if placement == "assistant":
        msgs = [{"role": "user", "content": phase1.ct.PROMPT_P.format(q=q)},
                {"role": "assistant", "content": a}] + msgs
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False) + "("


def letter_ids(tok) -> tuple[int, int]:
    a, b = tok.encode("A", add_special_tokens=False), tok.encode("B", add_special_tokens=False)
    if len(a) != 1 or len(b) != 1:
        raise SystemExit(f"'A'/'B' are not single tokens: {a} {b}")
    return a[0], b[0]


def next_token_probs(model, tok, texts: list[str], batch_size: int, device: str) -> np.ndarray:
    """[n, 2]: P(true) renormalised over A/B, and the A+B share of all probability."""
    import torch

    ida, idb = letter_ids(tok)
    enc = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    out = np.zeros((len(texts), 2))
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        width = max(len(enc[i]) for i in idx)
        ids = torch.full((len(idx), width), tok.pad_token_id or 0, dtype=torch.long)
        mask = torch.zeros((len(idx), width), dtype=torch.long)
        for r, i in enumerate(idx):  # right padding; read each row at its own last position
            ids[r, :len(enc[i])] = torch.tensor(enc[i])
            mask[r, :len(enc[i])] = 1
        with torch.no_grad():  # the output layer only at each row's last token (memory)
            hidden = model.model(input_ids=ids.to(device), attention_mask=mask.to(device)).last_hidden_state
            last = torch.tensor([len(enc[i]) - 1 for i in idx], device=hidden.device)
            logits = model.lm_head(hidden[torch.arange(len(idx), device=hidden.device), last])
        for r, i in enumerate(idx):
            p = torch.softmax(logits[r].float(), dim=-1)
            pa, pb = float(p[ida]), float(p[idb])
            out[i] = (pa / (pa + pb) if pa + pb > 0 else np.nan, pa + pb)
    return out


# ----------------------------------------------------------------------------- log-probability


def answer_logprobs(model, items: list[dict], batch_size: int, device: str) -> np.ndarray:
    """[n, 2]: sum and mean log-probability of the answer tokens given the prompt."""
    import torch

    out = np.full((len(items), 2), np.nan)
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        width = max(len(items[i]["ids"]) for i in idx)
        ids = torch.zeros((len(idx), width), dtype=torch.long)
        mask = torch.zeros((len(idx), width), dtype=torch.long)
        for r, i in enumerate(idx):
            ids[r, :len(items[i]["ids"])] = torch.tensor(items[i]["ids"])
            mask[r, :len(items[i]["ids"])] = 1
        with torch.no_grad():
            hidden = model.model(input_ids=ids.to(device), attention_mask=mask.to(device)).last_hidden_state
        for r, i in enumerate(idx):
            p, a = items[i]["n_prompt"], items[i]["n_answer"]
            if not a:
                continue
            with torch.no_grad():  # the output layer only over this answer's positions (memory)
                logp = torch.log_softmax(model.lm_head(hidden[r, p - 1:p + a - 1]).float(), -1)
            tgt = torch.tensor(items[i]["ids"][p:p + a], device=logp.device)
            lp = logp.gather(1, tgt[:, None]).squeeze(1)
            out[i] = (float(lp.sum()), float(lp.mean()))
    return out


# ----------------------------------------------------------------------------- probes


def probe_scores(config: str, rows: pd.DataFrame, states: np.ndarray) -> pd.DataFrame:
    """Phase 1B probes refitted at their chosen layer and C; scored on the new states."""
    pr = phase1.probe
    chosen = pd.read_parquet(phase1.PHASE1_RESULTS / "probe_scores.parquet")
    chosen = (chosen[(chosen.config == config) & (chosen.probe == "logreg") & (chosen.train_kind == chosen.category)]
              .groupby(["train_kind", "position"])[["layer", "C"]].first())
    tr_states, tr_rows = pr.load(config, "train")
    tr_rows = tr_rows[tr_rows.n_answer > 0]
    out = pd.DataFrame(index=rows.index)
    for kind in ("recall", "computation", "perception"):
        a = (tr_rows.category == kind).to_numpy()
        y = tr_rows.y.to_numpy()[a]
        b = (rows.category == kind).to_numpy()
        for pos in ("gen", "ans_last"):
            layer, C = chosen.loc[(kind, pos)]
            pi = list(pr.POSITIONS).index(pos)
            for name, L in ((f"probe_{pos}", int(layer)), *((("probe_l0_ans_last", 0),) if pos == "ans_last" else ())):
                X = np.asarray(tr_states[tr_rows.index.to_numpy()[a], L, pi, :], dtype=np.float32)
                clf = pr.logreg(float(C)).fit(X, y)
                out.loc[b, name] = clf.decision_function(np.asarray(states[b][:, L, pi, :], dtype=np.float32))
    return out


def check_reproduces_phase1b(config: str, rows: pd.DataFrame, scores: pd.DataFrame) -> None:
    ref = pd.read_parquet(phase1.PHASE1_RESULTS / "probe_scores.parquet")
    ref = ref[(ref.config == config) & (ref.probe == "logreg") & (ref.train_kind == ref.category)]
    for pos in ("gen", "ans_last"):
        r = ref[ref.position == pos].set_index("qid").score
        own = rows.source == "own"
        mine = scores.loc[own, f"probe_{pos}"]
        theirs = rows.loc[own, "qid"].map(r)
        ok = theirs.notna()
        print(f"  probe {pos}: own answers vs Phase 1B test scores, r = "
              f"{np.corrcoef(mine[ok], theirs[ok])[0, 1]:.4f} on {int(ok.sum())}")


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+", choices=sorted(phase1.extract.CONFIGS))
    ap.add_argument("--limit", type=int, help="smoke test: N rows per split x source x kind; nothing saved")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--skip-probes", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    qs = questions()
    for config in args.configs:
        name, base = phase1.extract.CONFIGS[config]
        print(f"=== {config} ({name}) {time.strftime('%H:%M')}")
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map=device)
        model.eval()
        rows = load_rows(config, None)
        shots = choose_shots(rows, qs)
        if args.limit:
            rows = load_rows(config, args.limit)
        res = rows[["config", "split", "qid", "category", "level", "family", "source", "correct", "fmt"]].copy()
        res["is_shot"] = rows.qid.isin({s["qid"] for s in shots}) & (rows.split == "train")

        # P(True), every row
        stext = shots_text(shots)
        for placement in (("user",) if base else PLACEMENTS):
            for k, sh in ((0, ""), (K_SHOTS, stext)):
                texts = [ptrue_text(tok, qs[r.qid]["question"], r.text, placement, sh, base)
                         for r in rows.itertuples()]
                if args.limit and placement == "user" and k == K_SHOTS:
                    print("  P(True) prompt end (repr):", repr(texts[0][-220:]))
                t0 = time.time()
                p = next_token_probs(model, tok, texts, args.batch_size, device)
                res[f"ptrue_{placement}_{k}"], res[f"ab_mass_{placement}_{k}"] = p[:, 0], p[:, 1]
                print(f"  P(True) {placement} {k}-shot: {len(texts)} in {time.time() - t0:.0f}s; "
                      f"A+B mass median {np.nanmedian(p[:, 1]):.3f}, min {np.nanmin(p[:, 1]):.3f}")

        # states, log-probability, probes: test rows
        test = rows.split == "test"
        trows = rows[test].reset_index(drop=True)
        items = [phase1.extract.encode(tok, qs[r.qid], r.text, base) for r in trows.itertuples()]
        if args.limit:
            ex = items[0]
            print("  state prompt end (repr):", repr(ex["text"][-160:]), "| answer:",
                  repr(tok.decode(ex["ids"][ex["n_prompt"]:])))
        save = not args.limit
        t0 = time.time()
        states = phase1.extract.extract(model, items, args.batch_size, device,
                                        STATES / config / "swaps.npy" if save else None)
        lp = answer_logprobs(model, items, args.batch_size, device)
        print(f"  states {states.shape} and log-probabilities in {time.time() - t0:.0f}s")
        res.loc[test, "logprob_sum"], res.loc[test, "logprob_mean"] = lp[:, 0], lp[:, 1]
        res.loc[test, "n_answer_tokens"] = [i["n_answer"] for i in items]
        print(res[test].groupby(["category", "source"]).logprob_mean.median().unstack().round(2).to_string())
        del model
        torch.cuda.empty_cache()
        if not args.skip_probes:
            sc = probe_scores(config, trows, states)
            check_reproduces_phase1b(config, trows, sc)
            for c in sc:
                res.loc[test, c] = sc[c].to_numpy()
        if save:
            states.flush()
            trows.drop(columns=["text"]).to_parquet(STATES / config / "swaps.parquet", index=False)
            phase1.RESULTS.mkdir(parents=True, exist_ok=True)
            res.to_parquet(phase1.RESULTS / f"extract_{config}.parquet", index=False)
            print(f"  wrote results/extract_{config}.parquet")
        else:
            print(res.groupby(["split", "source"])[[c for c in res if c.startswith("ptrue_")]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
