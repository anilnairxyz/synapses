#!/usr/bin/env bash
# Phase 1 runs on Azure AI Foundry (office). Generation only: the judge runs later on the 4090.
# This file is copied, with confidence_task.py and questions.jsonl, into the office repo
# (ai-model-evals/confidence/); the synapses copy is the source.
#
# Usage (office repo root):
#   confidence/run_foundry.sh pilot                 # 10 questions per category, every model
#   confidence/run_foundry.sh full                  # whole question set
#   confidence/run_foundry.sh pilot gpt-5-nano      # only the named configs
#
# Needs FOUNDRY_BASE_URL ({endpoint}/openai/v1) and FOUNDRY_API_KEY, or the AZURE_OPENAI_ENDPOINT /
# AZURE_OPENAI_API_KEY pair that check_models.py also accepts, in the environment or .env.
# Logs: outputs/confidence/<mode>/<config>/*.eval (commit them; they go to synapses for judging).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/.." || exit 1

MODE=${1:-pilot}
shift || true
case $MODE in
  pilot) LIMIT="-T per_category=10" ;;
  full) LIMIT="" ;;
  *) echo "mode must be pilot or full"; exit 1 ;;
esac

if [ -f .env ]; then set -a; . ./.env; set +a; fi
if [ -z "${FOUNDRY_BASE_URL:-}" ] && [ -n "${AZURE_OPENAI_ENDPOINT:-}" ]; then
  ep=${AZURE_OPENAI_ENDPOINT%/}
  case $ep in */openai/v1) FOUNDRY_BASE_URL=$ep ;; *) FOUNDRY_BASE_URL=$ep/openai/v1 ;; esac
  export FOUNDRY_BASE_URL
fi
if [ -z "${FOUNDRY_API_KEY:-}" ] && [ -n "${AZURE_OPENAI_API_KEY:-}" ]; then
  export FOUNDRY_API_KEY=$AZURE_OPENAI_API_KEY
fi
[ -n "${FOUNDRY_BASE_URL:-}" ] && [ -n "${FOUNDRY_API_KEY:-}" ] || { echo "FOUNDRY_BASE_URL / FOUNDRY_API_KEY not set"; exit 1; }

# config | deployment | extra args
# gpt-5-nano is a reasoning model: it accepts only its default temperature (main_temperature=null)
# and its hidden reasoning counts against max_tokens, so it gets more room and low effort.
CONFIGS=(
  "grok-4-1-fast|grok-4-1-fast-non-reasoning|"
  "gpt-4o|gpt-4o|"
  "phi-4-mini|phi-4-mini-instruct|"
  "gpt-5-nano|gpt-5-nano|-T main_temperature=null -T max_tokens=4000 --reasoning-effort low"
)

for entry in "${CONFIGS[@]}"; do
  IFS="|" read -r name deployment extra <<< "$entry"
  if [ $# -gt 0 ] && [[ ! " $* " =~ " $name " ]]; then continue; fi
  echo "=== $name ($deployment) $(date +%H:%M)"
  # shellcheck disable=SC2086
  uv run --extra azure inspect eval "$HERE/confidence_task.py" --model "openai-api/foundry/$deployment" \
    -T questions="$HERE/questions.jsonl" $LIMIT $extra \
    --max-connections 20 --tags "$name" --log-dir "outputs/confidence/$MODE/$name" --display plain
  echo "=== $name done $(date +%H:%M)"
done
