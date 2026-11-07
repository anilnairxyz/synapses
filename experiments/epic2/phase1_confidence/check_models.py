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
  uv run --extra evals python experiments/epic2/phase1_confidence/check_models.py MODEL [MODEL ...]

Model names, by where they run:
  anthropic/claude-haiku-5-5            personal key: ANTHROPIC_API_KEY
  openai/gpt-4o-mini                    personal key: OPENAI_API_KEY
  openai-api/foundry/<deployment>       office: FOUNDRY_BASE_URL={endpoint}/openai/v1, FOUNDRY_API_KEY
  openai-api/local/<served-model-name>  4090 vLLM: LOCAL_BASE_URL=http://localhost:8000/v1, LOCAL_API_KEY
  mockllm/model                         offline dry run of this script
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time

from inspect_ai.model import GenerateConfig, get_model

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # keys can come from the environment instead
    pass

ARITH = "What is 17 × 23? Reply with the number only."
JSON_PROMPT = (
    "What is the capital of Australia? Reply with JSON only, in the form "
    '{"answer": "<your answer>", "confidence": <probability from 0 to 1 that your answer is right>}'
)
SAMPLE_PROMPT = "Name one random animal. Reply with one word."


def _short(e: Exception) -> str:
    return f"{type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''}"


def _parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


async def check(name: str) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        model = get_model(name)
    except Exception as e:  # unknown provider, missing key or package
        hint = " (API key / base URL not set?)" if "Unable to initialise" in str(e) else ""
        return {"connect": "FAIL " + _short(e) + hint}

    try:
        t = time.perf_counter()
        r = await model.generate(ARITH, config=GenerateConfig(max_tokens=200))
        dt = time.perf_counter() - t
        text = r.completion.strip()
        ok = "391" in text.replace(",", "")
        out["connect"] = f"{'ok' if ok else 'WRONG'} ({dt:.1f}s) {text[:40]!r}"
    except Exception as e:
        return {"connect": "FAIL " + _short(e)}

    try:
        r = await model.generate(JSON_PROMPT, config=GenerateConfig(max_tokens=300))
        d = _parse_json(r.completion)
        if d and "answer" in d and isinstance(d.get("confidence"), (int, float)):
            out["json"] = f"ok answer={d['answer']!r} confidence={d['confidence']}"
        else:
            out["json"] = f"UNPARSED {r.completion.strip()[:60]!r}"
    except Exception as e:
        out["json"] = "FAIL " + _short(e)

    try:
        cfg = GenerateConfig(temperature=1.0, max_tokens=50)
        rs = await asyncio.gather(*[model.generate(SAMPLE_PROMPT, config=cfg) for _ in range(3)])
        words = [r.completion.strip()[:20] for r in rs]
        out["sampling"] = f"ok {len(set(words))}/3 distinct {words}"
    except Exception as e:
        out["sampling"] = "FAIL " + _short(e)

    try:
        r = await model.generate(ARITH, config=GenerateConfig(logprobs=True, top_logprobs=5, max_tokens=20))
        lp = r.choices[0].logprobs
        if lp and lp.content:
            first = lp.content[0]
            out["logprobs"] = f"ok first token {first.token!r} logprob={first.logprob:.3f}"
        else:
            out["logprobs"] = "none returned"
    except Exception as e:
        out["logprobs"] = "FAIL " + _short(e)
    return out


async def main(names: list[str]) -> int:
    results = await asyncio.gather(*[check(n) for n in names])
    failed = 0
    for name, res in zip(names, results, strict=True):
        print(f"\n{name}")
        for k in ("connect", "json", "sampling", "logprobs"):
            if k in res:
                print(f"  {k:9s} {res[k]}")
        failed += not res["connect"].startswith("ok")
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(asyncio.run(main(sys.argv[1:])))
