#!/usr/bin/env bash
# E2b on Azure AI Foundry (office). Copy this folder into the office repo (for example as
# ai-model-evals/swap/) and run from the office repo root:
#   swap/run_foundry.sh smoke                 # 3 questions per kind x source x attribution
#   swap/run_foundry.sh full                  # everything (about 2,600 calls per model)
#   swap/run_foundry.sh full gpt-4o           # only the named configs
# Needs FOUNDRY_BASE_URL ({endpoint}/openai/v1) and FOUNDRY_API_KEY (or the AZURE_OPENAI_ENDPOINT /
# AZURE_OPENAI_API_KEY pair), in the environment or .env, as for Phase 1's run_foundry.sh.
# Logs: outputs/swap/<mode>/<config>/*.eval -> bring the full ones back to synapses as
# experiments/epic2/phase2_confidence_gap/runs/foundry/full/<config>/.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/.." || exit 1
MODE=${1:-smoke}
shift || true
case $MODE in
  smoke) LIMIT="-T limit_per_cell=3" ;;
  full) LIMIT="" ;;
  *) echo "mode must be smoke or full"; exit 1 ;;
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

# config | deployment | extra args (Phase 1's per-model settings; max_tokens 512 because under
# "my answer" a model may work the problem in its reply before the JSON)
CONFIGS=(
  "grok-4-1-fast|grok-4-1-fast-non-reasoning|-T max_tokens=512"
  "gpt-4o|gpt-4o|-T max_tokens=512"
  "gpt-5-nano|gpt-5-nano|-T temperature=null -T max_tokens=4000 --reasoning-effort low"
)
NAME=$(basename "$HERE")
status=0
for entry in "${CONFIGS[@]}"; do
  IFS="|" read -r name deployment extra <<< "$entry"
  if [ $# -gt 0 ] && [[ ! " $* " =~ " $name " ]]; then continue; fi
  echo "=== $name ($deployment) $(date +%H:%M)"
  # shellcheck disable=SC2086
  if uv run --extra azure inspect eval "$NAME/swap_task_foundry.py" --model "openai-api/foundry/$deployment" \
    -T config=$name $LIMIT $extra --max-connections 20 --tags "$name" \
    --log-dir "outputs/swap/$MODE/$name" --display plain; then
    echo "=== $name done $(date +%H:%M)"
  else
    echo "=== $name FAILED (exit $?) $(date +%H:%M)"; status=1
  fi
done
exit $status
