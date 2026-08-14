#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_URL="${NEGOTIATION_QWEN_BASE_URL:-http://127.0.0.1:8002/v1}"
MODEL="${NEGOTIATION_QWEN_MODEL:-Qwen3-30B-A3B-Instruct-2507-base}"
RUNS="${RUNS:-5}"
EPISODES_PER_RUN="${EPISODES_PER_RUN:-20}"
MAX_TURNS="${MAX_TURNS:-10}"
TEMPERATURE="${TEMPERATURE:-0.7}"
SEED="${SEED:-20260820}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-arena_runs/formal_belief_matrix_qwen30b_5x20_20260812}"
METHODS="${METHODS:-direct opponent_simulation_paper framework_v3_3 framework_v4 framework_v4_frozen framework_v4_wrong_confident framework_v4_shuffled framework_v4_oracle framework_v5}"
SETTINGS="${SETTINGS:-buyer seller resource_first resource_second}"
SWITCH_EPISODE="${SWITCH_EPISODE:-0}"
SWITCH_POLICY="${SWITCH_POLICY:-direct}"

"$PYTHON_BIN" - "$BASE_URL" <<'PY'
import json
import sys
import urllib.request

base = sys.argv[1].rstrip("/")
with urllib.request.urlopen(base + "/models", timeout=10) as response:
    payload = json.load(response)
if not payload.get("data"):
    raise SystemExit(f"No model available at {base}")
print(json.dumps({"service": base, "models": [row.get("id") for row in payload["data"]]}))
PY

mkdir -p "$OUTPUT_ROOT/logs"

run_cell() {
  local method="$1"
  local setting="$2"
  local output="$OUTPUT_ROOT/$setting/$method"
  local log="$OUTPUT_ROOT/logs/${setting}__${method}.log"
  "$PYTHON_BIN" -u -m arena_integration.run_opponent_simulation_setting \
    --method "$method" \
    --setting "$setting" \
    --runs "$RUNS" \
    --episodes-per-run "$EPISODES_PER_RUN" \
    --max-turns "$MAX_TURNS" \
    --candidate-count 5 \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --api-key-env NEGOTIATION_QWEN_API_KEY \
    --opponent-model "$MODEL" \
    --opponent-base-url "$BASE_URL" \
    --opponent-api-key-env NEGOTIATION_QWEN_API_KEY \
    --opponent-policy strategic_brainstorming \
    --opponent-switch-episode "$SWITCH_EPISODE" \
    --opponent-switch-policy "$SWITCH_POLICY" \
    --protocol-mode normalize_retry \
    --protocol-repair-attempts 1 \
    --temperature "$TEMPERATURE" \
    --max-tokens 1600 \
    --context-char-limit 30000 \
    --seed "$SEED" \
    --output-dir "$output" \
    --resume \
    --restart-error-runs \
    --fail-fast-transport >"$log" 2>&1
}

active=0
failures=0
for setting in $SETTINGS; do
  for method in $METHODS; do
    run_cell "$method" "$setting" &
    active=$((active + 1))
    if (( active >= MAX_PARALLEL )); then
      if ! wait -n; then
        failures=$((failures + 1))
      fi
      active=$((active - 1))
    fi
  done
done

while (( active > 0 )); do
  if ! wait -n; then
    failures=$((failures + 1))
  fi
  active=$((active - 1))
done

"$PYTHON_BIN" scripts/summarize_formal_belief_matrix.py "$OUTPUT_ROOT"
if (( failures > 0 )); then
  echo "$failures matrix cells failed; inspect $OUTPUT_ROOT/logs" >&2
  exit 1
fi
