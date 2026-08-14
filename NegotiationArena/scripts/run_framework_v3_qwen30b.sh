#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_URL="${NEGOTIATION_QWEN_BASE_URL:-http://127.0.0.1:8002/v1}"
MODEL="${NEGOTIATION_QWEN_MODEL:-Qwen3-30B-A3B-Instruct-2507-base}"
OPPONENT_BASE_URL="${NEGOTIATION_OPPONENT_BASE_URL:-$BASE_URL}"
OPPONENT_MODEL="${NEGOTIATION_OPPONENT_MODEL:-$MODEL}"

RUNS="${RUNS:-1}"
EPISODES_PER_RUN="${EPISODES_PER_RUN:-1}"
MAX_TURNS="${MAX_TURNS:-10}"
TEMPERATURE="${TEMPERATURE:-0.7}"
SEED="${SEED:-20260812}"
SETTINGS="${SETTINGS:-buyer seller resource_first resource_second}"
METHOD="${METHOD:-framework_v3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-arena_runs/dc_bap_v3_canary_20260812}"

"$PYTHON_BIN" - "$BASE_URL" <<'PY'
import json
import sys
import urllib.request

base = sys.argv[1].rstrip("/")
try:
    with urllib.request.urlopen(base + "/models", timeout=5) as response:
        payload = json.load(response)
except Exception as exc:
    raise SystemExit(f"Model service preflight failed for {base}: {exc}")
if not payload.get("data"):
    raise SystemExit(f"Model service returned no models: {base}")
print(json.dumps({"service": base, "models": [row.get("id") for row in payload["data"]]}))
PY

for SETTING in $SETTINGS; do
  "$PYTHON_BIN" -u -m arena_integration.run_opponent_simulation_setting \
    --method "$METHOD" \
    --setting "$SETTING" \
    --runs "$RUNS" \
    --episodes-per-run "$EPISODES_PER_RUN" \
    --max-turns "$MAX_TURNS" \
    --candidate-count 5 \
    --temperature "$TEMPERATURE" \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --api-key-env NEGOTIATION_QWEN_API_KEY \
    --opponent-model "$OPPONENT_MODEL" \
    --opponent-base-url "$OPPONENT_BASE_URL" \
    --opponent-api-key-env NEGOTIATION_QWEN_API_KEY \
    --opponent-policy strategic_brainstorming \
    --protocol-mode normalize_retry \
    --protocol-repair-attempts 1 \
    --max-tokens 1600 \
    --context-char-limit 30000 \
    --seed "$SEED" \
    --output-dir "$OUTPUT_ROOT/$SETTING/$METHOD" \
    --resume
done
