"""Sanity-check model connections before any eval run.

For each model, four cheap calls (a few hundred tokens in all):

  connect    one plain question ("What is 17 × 23?"): reachable, authenticated, and right?
  json       the answer + confidence JSON that methods a/d rely on: does it parse?
  sampling   3 calls at temperature 1.0 (method b, sample agreement): accepted? do they vary?
  logprobs   token log-probabilities (method e): returned by this provider/model?

Every check is reported separately, so a model that rejects temperature or logprobs (common
for reasoning models) still shows what it *can* do. Nothing is written except to stdout.

Portable: Inspect + stdlib only (python-dotenv comes with Inspect), so the file can be
copied as-is to the office repo or the 4090 PC.

Usage (keys in .env or the environment; never paste them in chat):
  uv run --extra evals python experiments/epic2/phase1_confidence/check_models.py [--no-think] MODEL [MODEL ...]

  --no-think   turn off thinking on hybrid open-weights models served by vLLM (Qwen3 and
               similar; applied to openai-api/local/* models only), via the chat template's enable_thinking=False. Without it, those
               models write <think>...</think> before every answer; Inspect strips that out
               of the completion, provided the thinking finishes within max_tokens.

Model names, by where they run:
  anthropic/claude-haiku-5-5            personal key: ANTHROPIC_API_KEY
  openai/gpt-4o-mini                    personal key: OPENAI_API_KEY
  openai-api/foundry/<deployment>       office: FOUNDRY_BASE_URL={endpoint}/openai/v1, FOUNDRY_API_KEY
                                        (or the office repo's AZURE_OPENAI_ENDPOINT / _API_KEY)
  openai-api/local/<served-model-name>  4090 vLLM: LOCAL_BASE_URL=http://localhost:8000/v1, LOCAL_API_KEY
  mockllm/model                         offline dry run of this script
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time

from inspect_ai.model import GenerateConfig, get_model

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # keys can come from the environment instead
    pass

# The office ai-model-evals repo keeps its Foundry settings as AZURE_OPENAI_*; reuse them for
# openai-api/foundry/* models unless FOUNDRY_* is set explicitly.
if not os.environ.get("FOUNDRY_BASE_URL") and os.environ.get("AZURE_OPENAI_ENDPOINT"):
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    os.environ["FOUNDRY_BASE_URL"] = endpoint if endpoint.endswith("/openai/v1") else endpoint + "/openai/v1"
if not os.environ.get("FOUNDRY_API_KEY") and os.environ.get("AZURE_OPENAI_API_KEY"):
    os.environ["FOUNDRY_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]

ARITH = "What is 17 × 23? Reply with the number only."
JSON_PROMPT = (
    "What is the capital of Australia? Reply with JSON only, in the form "
    '{"answer": "<your answer>", "confidence": <probability from 0 to 1 that your answer is right>}'
)
SAMPLE_PROMPT = "Name one random animal. Reply with one word."
# Fail fast: without this, Inspect retries a dead endpoint with growing back-off and the
# script looks hung.
BASE = GenerateConfig(timeout=60, max_retries=2)


def _step(name: str, what: str) -> None:
    print(f"[{name}] {what}...", flush=True)


def _short(e: Exception) -> str:
    if not str(e) and e.__cause__ is not None:  # e.g. ModelGenerateError wrapping the API error
        e = e.__cause__
    last = getattr(e, "last_attempt", None)  # tenacity RetryError: report the real cause
    if last is not None and last.exception() is not None:
        e = last.exception()
    if "Connection" in type(e).__name__:
        return f"{type(e).__name__}: server not reachable (is it running? check the *_BASE_URL)"
    return f"{type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''}"


def _answer(text: str) -> str:
    """The completion without any <think>...</think> block Inspect didn't strip."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    return "[still thinking at max_tokens] " + text[7:60] if text.startswith("<think>") else text


def _parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


async def check(name: str, no_think: bool = False) -> dict[str, str]:
    out: dict[str, str] = {}
    base = BASE
    if no_think and name.startswith("openai-api/local/"):  # vLLM only; other APIs reject it
        base = base.merge(GenerateConfig(extra_body={"chat_template_kwargs": {"enable_thinking": False}}))
    try:
        model = get_model(name, config=base)
    except Exception as e:  # unknown provider, missing key or package
        hint = " (API key / base URL not set?)" if "Unable to initialise" in str(e) else ""
        return {"connect": "FAIL " + _short(e) + hint}

    _step(name, "connect")
    try:
        t = time.perf_counter()
        r = await model.generate(ARITH, config=GenerateConfig(max_tokens=2000))
        dt = time.perf_counter() - t
        text = _answer(r.completion)
        ok = "391" in text.replace(",", "")
        out["connect"] = f"{'ok' if ok else 'WRONG'} ({dt:.1f}s) {text[:40]!r}"
    except Exception as e:
        return {"connect": "FAIL " + _short(e)}

    _step(name, "json")
    try:
        r = await model.generate(JSON_PROMPT, config=GenerateConfig(max_tokens=2000))
        d = _parse_json(r.completion)
        if d and "answer" in d and isinstance(d.get("confidence"), (int, float)):
            out["json"] = f"ok answer={d['answer']!r} confidence={d['confidence']}"
        else:
            out["json"] = f"UNPARSED {r.completion.strip()[:60]!r}"
    except Exception as e:
        out["json"] = "FAIL " + _short(e)

    _step(name, "sampling")
    try:
        cfg = GenerateConfig(temperature=1.0, max_tokens=2000)
        rs = await asyncio.gather(*[model.generate(SAMPLE_PROMPT, config=cfg) for _ in range(3)])
        words = [_answer(r.completion)[:20] for r in rs]
        out["sampling"] = f"ok {len(set(words))}/3 distinct {words}"
    except Exception as e:
        out["sampling"] = "FAIL " + _short(e)

    _step(name, "logprobs")
    try:
        cfg = GenerateConfig(logprobs=True, top_logprobs=5, max_tokens=2000)
        r = await model.generate(ARITH, config=cfg)
        lp = r.choices[0].logprobs
        if lp and lp.content:
            # First token after any thinking, i.e. the token the answer starts with.
            toks = [t.token for t in lp.content]
            start = toks.index("</think>") + 1 if "</think>" in toks else 0
            while start < len(toks) - 1 and not toks[start].strip():
                start += 1
            first = lp.content[start]
            where = " (after thinking)" if start else ""
            out["logprobs"] = f"ok answer token {first.token!r} logprob={first.logprob:.3f}{where}"
        else:
            out["logprobs"] = "none returned"
    except Exception as e:
        out["logprobs"] = "FAIL " + _short(e)
    return out


async def main(names: list[str], no_think: bool = False) -> int:
    results = await asyncio.gather(*[check(n, no_think) for n in names])
    failed = 0
    for name, res in zip(names, results, strict=True):
        print(f"\n{name}")
        for k in ("connect", "json", "sampling", "logprobs"):
            if k in res:
                print(f"  {k:9s} {res[k]}")
        failed += not res["connect"].startswith("ok")
    return 1 if failed else 0


if __name__ == "__main__":
    args = sys.argv[1:]
    no_think = "--no-think" in args
    names = [a for a in args if a != "--no-think"]
    if not names:
        sys.exit(__doc__)
    sys.exit(asyncio.run(main(names, no_think)))
