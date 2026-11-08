#!/usr/bin/env bash
# Phase 1 runs on the 4090: for each configuration, start vLLM, run the task, stop vLLM.
#
# Usage (from the repo root, in WSL):
#   experiments/epic2/phase1_confidence/run_4090.sh pilot            # 10 questions per category
#   experiments/epic2/phase1_confidence/run_4090.sh full             # whole question set
#   experiments/epic2/phase1_confidence/run_4090.sh pilot qwen3-8b   # only the named configs
#   experiments/epic2/phase1_confidence/run_4090.sh train            # Phase 1B probe training set:
#       plain answer (P) only, on train_questions.jsonl, for the Phase 1B configs
#       (qwen3-8b, qwen3-1.7b, qwen3-8b-base); judge afterwards with judge_4090.sh 4090/train
#
# Logs: experiments/epic2/phase1_confidence/runs/4090/<mode>/<config>/ (Inspect .eval files; committed so the Mac can analyse them);
#       vLLM server output in logs/vllm_<config>.log.
# Needs LOCAL_BASE_URL / LOCAL_API_KEY in .env (as for check_models.py).
set -u
cd "$(dirname "$0")/../../.." || exit 1

MODE=${1:-pilot}
shift || true
VLLM=${VLLM:-$HOME/vllm-env/bin/vllm}
HERE=experiments/epic2/phase1_confidence
TASK=$HERE/confidence_task.py
case $MODE in
  pilot) LIMIT="-T per_category=10" ;;
  full) LIMIT="" ;;
  train) LIMIT="-T methods=p -T questions=data/epic2/confidence/train_questions.jsonl"
         [ $# -gt 0 ] || set -- qwen3-8b qwen3-1.7b qwen3-8b-base ;;
  *) echo "mode must be pilot, full or train"; exit 1 ;;
esac

# config | served model | vLLM args | task args
# Thinking runs use Qwen's recommended temperature 0.6 for the main calls: greedy decoding
# can make thinking loop.
CONFIGS=(
  "qwen3-8b|Qwen/Qwen3-8B|--max-model-len 8192|-T no_think=true"
  "qwen3-1.7b|Qwen/Qwen3-1.7B|--max-model-len 8192|-T no_think=true"
  "qwen3-14b|Qwen/Qwen3-14B-FP8|--max-model-len 8192|-T no_think=true"
  "qwen3-8b-base|Qwen/Qwen3-8B-Base|--max-model-len 4096 --chat-template $HERE/chat_template_raw.jinja|-T base=true -T max_tokens=32"
  "qwen3-8b-think|Qwen/Qwen3-8B|--max-model-len 8192|-T main_temperature=0.6 -T max_tokens=6000"
)

mkdir -p logs
for entry in "${CONFIGS[@]}"; do
  IFS="|" read -r name model vargs targs <<< "$entry"
  if [ $# -gt 0 ] && [[ ! " $* " =~ " $name " ]]; then continue; fi
  echo "=== $name ($model) $(date +%H:%M)"

  pkill -f "vllm serve" 2>/dev/null; sleep 5
  # The vLLM venv's bin goes on PATH (as if activated): flashinfer compiles kernels at
  # start-up and looks for `ninja` there.
  # shellcheck disable=SC2086
  PATH="$(dirname "$VLLM"):$PATH" "$VLLM" serve "$model" $vargs --gpu-memory-utilization 0.90 > "logs/vllm_$name.log" 2>&1 &
  pid=$!
  until curl -s localhost:8000/v1/models > /dev/null; do
    if ! kill -0 $pid 2> /dev/null; then echo "vLLM failed to start: see logs/vllm_$name.log"; continue 2; fi
    sleep 10
  done

  # shellcheck disable=SC2086
  uv run --extra evals inspect eval "$TASK" --model "openai-api/local/$model" $LIMIT $targs \
    --max-connections 64 --tags "$name" --log-dir "$HERE/runs/4090/$MODE/$name" \
    --display plain

  kill $pid; wait $pid 2> /dev/null
  echo "=== $name done $(date +%H:%M)"
done
