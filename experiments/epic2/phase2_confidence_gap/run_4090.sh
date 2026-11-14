#!/usr/bin/env bash
# Phase 2 (E2, E3) runs on the 4090.
#
# Usage (repo root, WSL):
#   experiments/epic2/phase2_confidence_gap/run_4090.sh smoke     # small: check prompts and outputs
#   experiments/epic2/phase2_confidence_gap/run_4090.sh full      # everything (a few hours)
#   experiments/epic2/phase2_confidence_gap/run_4090.sh full extract   # only one part:
#       extract  states, log-probabilities, P(True), probe scores (extract_answers.py, HF)
#       swap     verbal confidence, self vs user (swap_task.py, vLLM + Inspect)
#
# Needs: Phase 1's training states on this machine (experiments/epic2/phase1_confidence/states/),
# data/epic2/confidence/phase2/swaps.jsonl (pulled), LOCAL_BASE_URL / LOCAL_API_KEY in .env.
# Outputs: results/extract_<config>.parquet; Inspect logs in runs/4090/<mode>/<config>/.
set -u
cd "$(dirname "$0")/../../.." || exit 1

MODE=${1:-smoke}
PART=${2:-all}
VLLM=${VLLM:-$HOME/vllm-env/bin/vllm}
HERE=experiments/epic2/phase2_confidence_gap
case $MODE in
  smoke) XLIM="--limit 4"; SLIM="-T limit_per_cell=3" ;;
  full) XLIM=""; SLIM="" ;;
  *) echo "mode must be smoke or full"; exit 1 ;;
esac

if [ "$PART" = all ] || [ "$PART" = extract ]; then
  echo "=== extract $(date +%H:%M)"
  # shellcheck disable=SC2086
  uv run --extra evals --extra interp python $HERE/extract_answers.py qwen3-8b qwen3-1.7b qwen3-8b-base $XLIM
fi

if [ "$PART" = all ] || [ "$PART" = swap ]; then
  mkdir -p $HERE/logs
  for entry in "qwen3-8b|Qwen/Qwen3-8B" "qwen3-1.7b|Qwen/Qwen3-1.7B"; do
    IFS="|" read -r name model <<< "$entry"
    echo "=== swap $name $(date +%H:%M)"
    pkill -f "vllm serve" 2>/dev/null; sleep 5
    PATH="$(dirname "$VLLM"):$PATH" "$VLLM" serve "$model" --max-model-len 8192 \
      --gpu-memory-utilization 0.90 > "$HERE/logs/vllm_$name.log" 2>&1 &
    pid=$!
    until curl -s localhost:8000/v1/models > /dev/null; do
      if ! kill -0 $pid 2> /dev/null; then echo "vLLM failed: see $HERE/logs/vllm_$name.log"; continue 2; fi
      sleep 10
    done
    # shellcheck disable=SC2086
    uv run --extra evals inspect eval $HERE/swap_task.py --model "openai-api/local/$model" \
      -T config=$name $SLIM --max-connections 64 --tags "$name" \
      --log-dir "$HERE/runs/4090/$MODE/$name" --display plain
    kill $pid; wait $pid 2> /dev/null
  done
fi
echo "=== done $(date +%H:%M)"
