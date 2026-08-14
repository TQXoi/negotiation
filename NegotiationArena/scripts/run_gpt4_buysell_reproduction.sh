#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running GPT-4 experiments}"
MODEL="${NEGOTIATION_GPT4_MODEL:-gpt-4o}"
EPISODES="${EPISODES:-60}"
OUTPUT_ROOT="${OUTPUT_ROOT:-arena_runs/gpt4_buysell}"
PYTHON_BIN="${PYTHON_BIN:-python}"

for MODE in direct framework_seller framework_buyer; do
  "$PYTHON_BIN" -m arena_integration.run_buysell \
    --mode "$MODE" \
    --episodes "$EPISODES" \
    --model "$MODEL" \
    --api-key-env OPENAI_API_KEY \
    --output-dir "$OUTPUT_ROOT/$MODE" \
    --resume
done
