#!/usr/bin/env bash
# Judge on the 4090: serve the grader (Qwen3-14B-FP8) and either check it against the gold set
# or re-score run logs in place. Re-scoring is safe to repeat: each pass overwrites the last.
#
# Usage (from the repo root, in WSL):
#   experiments/epic2/phase1_confidence/judge_4090.sh check              # gold set, thinking off and on
#   experiments/epic2/phase1_confidence/judge_4090.sh                    # re-score every log under runs/
#   experiments/epic2/phase1_confidence/judge_4090.sh foundry            # only runs/foundry/ (path under runs/)
#   experiments/epic2/phase1_confidence/judge_4090.sh 4090/full/qwen3-8b # one config
#   THINK=1 experiments/epic2/phase1_confidence/judge_4090.sh 4090/full  # re-score with thinking on
#
# Check results go to checks/judge_check_<think|nothink>.txt (and .jsonl).
set -u
cd "$(dirname "$0")/../../.." || exit 1

VLLM=${VLLM:-$HOME/vllm-env/bin/vllm}
HERE=experiments/epic2/phase1_confidence
JUDGE=Qwen/Qwen3-14B-FP8
if [ "${THINK:-0}" = 1 ]; then
  GRADER="{model: openai-api/local/$JUDGE}"
else
  GRADER="{model: openai-api/local/$JUDGE, extra_body: {chat_template_kwargs: {enable_thinking: false}}}"
fi

# inspect score builds a client for each log's original model even though only the grader is
# called, so models not served here (Foundry, Claude) need some credentials set. Placeholders do:
# a call with them would fail rather than spend anything.
export FOUNDRY_API_KEY=${FOUNDRY_API_KEY:-unused-for-judging}
export FOUNDRY_BASE_URL=${FOUNDRY_BASE_URL:-http://localhost:9/v1}
export ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-unused-for-judging}

if [ "${1:-}" != check ]; then
  mapfile -t LOGS < <(find "$HERE/runs/${1:-}" -name "*.eval" | sort)
  [ ${#LOGS[@]} -gt 0 ] || { echo "no .eval logs under $HERE/runs/${1:-}"; exit 1; }
fi

mkdir -p logs
pkill -f "vllm serve" 2>/dev/null; sleep 5
PATH="$(dirname "$VLLM"):$PATH" "$VLLM" serve "$JUDGE" --max-model-len 8192 --gpu-memory-utilization 0.90 \
  > logs/vllm_judge.log 2>&1 &
pid=$!
until curl -s localhost:8000/v1/models > /dev/null; do
  if ! kill -0 $pid 2> /dev/null; then echo "vLLM failed to start: see logs/vllm_judge.log"; exit 1; fi
  sleep 10
done

if [ "${1:-}" = check ]; then
  for mode in nothink think; do
    flag=""; [ $mode = think ] && flag="--think"
    echo "=== gold-set check, $mode $(date +%H:%M)"
    # shellcheck disable=SC2086
    uv run --extra evals python "$HERE/check_judge.py" --model "openai-api/local/$JUDGE" $flag \
      --out "$HERE/checks/judge_check_$mode.jsonl" | tee "$HERE/checks/judge_check_$mode.txt"
  done
else
  for f in "${LOGS[@]}"; do
    echo "=== $f $(date +%H:%M)"
    uv run --extra evals inspect score "$f" \
      --scorer "$HERE/confidence_task.py@confidence_scorer" -S judge=true \
      --model-role "grader=$GRADER" --action overwrite --overwrite --display plain
  done
fi

kill $pid; wait $pid 2> /dev/null
echo "=== done $(date +%H:%M)"
