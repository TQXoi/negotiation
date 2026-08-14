#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BASE_URL="${NEGOTIATION_QWEN_BASE_URL:-http://127.0.0.1:8002/v1}"
MODEL="${NEGOTIATION_QWEN_MODEL:-Qwen3-30B-A3B-Instruct-2507-base}"
RUNS="${RUNS:-1}"
EPISODES_PER_RUN="${EPISODES_PER_RUN:-20}"
MAX_TURNS="${MAX_TURNS:-10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-arena_runs/oppsim_parser_safe_v2_pilot}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SETTINGS="${SETTINGS:-buyer}"
METHODS="${METHODS:-direct opponent_simulation framework}"
PROTOCOL_MODE="${PROTOCOL_MODE:-normalize_retry}"

for SETTING in $SETTINGS; do
  for METHOD in $METHODS; do
    "$PYTHON_BIN" -u -m arena_integration.run_opponent_simulation_setting \
      --method "$METHOD" \
      --setting "$SETTING" \
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
      --protocol-mode "$PROTOCOL_MODE" \
      --protocol-repair-attempts 1 \
      --output-dir "$OUTPUT_ROOT/$SETTING/$METHOD" \
      --resume
  done

  if [[ "$METHODS" == *"direct"* && "$METHODS" == *"opponent_simulation"* && "$METHODS" == *"framework"* ]]; then
    "$PYTHON_BIN" -m arena_integration.summarize_repeated_comparison \
      --direct "$OUTPUT_ROOT/$SETTING/direct" \
      --opponent-simulation "$OUTPUT_ROOT/$SETTING/opponent_simulation" \
      --framework "$OUTPUT_ROOT/$SETTING/framework" \
      --output "$OUTPUT_ROOT/$SETTING/comparison.json"
  fi
done
