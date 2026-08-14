#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
ACTOR_BASE_URL="${NEGOTIATION_QWEN_BASE_URL:-http://127.0.0.1:8002/v1}"
ACTOR_MODEL="${NEGOTIATION_QWEN_MODEL:-Qwen3-30B-A3B-Instruct-2507-base}"

# By default this is a reproducible same-model local comparison, not a numeric
# reproduction of the paper (whose actual opponent is Gemini-2.5-Flash).
OPPONENT_BASE_URL="${NEGOTIATION_OPPONENT_BASE_URL:-$ACTOR_BASE_URL}"
OPPONENT_MODEL="${NEGOTIATION_OPPONENT_MODEL:-$ACTOR_MODEL}"
OPPONENT_API_KEY_ENV="${NEGOTIATION_OPPONENT_API_KEY_ENV:-NEGOTIATION_QWEN_API_KEY}"

RUNS="${RUNS:-1}"
EPISODES_PER_RUN="${EPISODES_PER_RUN:-3}"
MAX_TURNS="${MAX_TURNS:-10}"
SETTINGS="${SETTINGS:-buyer seller resource_first resource_second}"
METHODS="${METHODS:-direct opponent_simulation_paper framework_v2}"
PROTOCOL_MODE="${PROTOCOL_MODE:-normalize_retry}"
OUTPUT_ROOT="${OUTPUT_ROOT:-arena_runs/paper_baselines_framework_v2_qwen30b_smoke}"

"$PYTHON_BIN" - "$ACTOR_BASE_URL" <<'PY'
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
  for METHOD in $METHODS; do
    "$PYTHON_BIN" -u -m arena_integration.run_opponent_simulation_setting \
      --method "$METHOD" \
      --setting "$SETTING" \
      --runs "$RUNS" \
      --episodes-per-run "$EPISODES_PER_RUN" \
      --max-turns "$MAX_TURNS" \
      --candidate-count 5 \
      --model "$ACTOR_MODEL" \
      --base-url "$ACTOR_BASE_URL" \
      --api-key-env NEGOTIATION_QWEN_API_KEY \
      --opponent-model "$OPPONENT_MODEL" \
      --opponent-base-url "$OPPONENT_BASE_URL" \
      --opponent-api-key-env "$OPPONENT_API_KEY_ENV" \
      --protocol-mode "$PROTOCOL_MODE" \
      --protocol-repair-attempts 1 \
      --max-tokens 1600 \
      --context-char-limit 30000 \
      --output-dir "$OUTPUT_ROOT/$SETTING/$METHOD" \
      --resume
  done

  "$PYTHON_BIN" -m arena_integration.summarize_repeated_comparison \
    --direct "$OUTPUT_ROOT/$SETTING/direct" \
    --opponent-simulation "$OUTPUT_ROOT/$SETTING/opponent_simulation_paper" \
    --framework "$OUTPUT_ROOT/$SETTING/framework_v2" \
    --output "$OUTPUT_ROOT/$SETTING/comparison.json"
done
