#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BASE_URL="${NEGOTIATION_QWEN_BASE_URL:-http://127.0.0.1:8002/v1}"
MODEL="${NEGOTIATION_QWEN_MODEL:-Qwen3-30B-A3B-Instruct-2507-base}"
RUNS="${RUNS:-1}"
EPISODES_PER_RUN="${EPISODES_PER_RUN:-3}"
MAX_TURNS="${MAX_TURNS:-10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-arena_runs/opponent_simulation_qwen30b}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SETTINGS="${SETTINGS:-buyer seller resource_first resource_second}"

for SETTING in $SETTINGS; do
  for METHOD in direct opponent_simulation framework; do
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
      --output-dir "$OUTPUT_ROOT/$SETTING/$METHOD" \
      --resume
  done

  "$PYTHON_BIN" -m arena_integration.summarize_repeated_comparison \
    --direct "$OUTPUT_ROOT/$SETTING/direct" \
    --opponent-simulation "$OUTPUT_ROOT/$SETTING/opponent_simulation" \
    --framework "$OUTPUT_ROOT/$SETTING/framework" \
    --output "$OUTPUT_ROOT/$SETTING/comparison.json"
done
