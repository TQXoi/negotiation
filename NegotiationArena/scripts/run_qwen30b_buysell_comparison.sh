#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BASE_URL="${NEGOTIATION_QWEN_BASE_URL:-http://127.0.0.1:8002/v1}"
MODEL="${NEGOTIATION_QWEN_MODEL:-Qwen3-30B-A3B-Instruct-2507-base}"
EPISODES="${EPISODES:-20}"
OUTPUT_ROOT="${OUTPUT_ROOT:-arena_runs/qwen30b_buysell_8002}"
PYTHON_BIN="${PYTHON_BIN:-python}"

for MODE in direct framework_seller framework_buyer; do
  "$PYTHON_BIN" -m arena_integration.run_buysell \
    --mode "$MODE" \
    --episodes "$EPISODES" \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --api-key-env NEGOTIATION_QWEN_API_KEY \
    --output-dir "$OUTPUT_ROOT/$MODE" \
    --resume
done
