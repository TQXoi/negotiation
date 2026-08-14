#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY for GPT-4o}"

PYTHON_BIN="${PYTHON_BIN:-python}"
QWEN_MODEL="${NEGOTIATION_QWEN_MODEL:-Qwen3-30B-A3B-Instruct-2507-base}"
QWEN_URL="${NEGOTIATION_QWEN_BASE_URL:-http://127.0.0.1:8002/v1}"
GPT_MODEL="${NEGOTIATION_GPT4_MODEL:-gpt-4o}"
EPISODES="${EPISODES:-60}"
OUTPUT_ROOT="${OUTPUT_ROOT:-arena_runs/paper_protocol_qwen_gpt4}"

# The ICML protocol excludes self-pairs. With the two currently available
# models this produces both ordered cross-model cells.
"$PYTHON_BIN" -m arena_integration.run_buysell \
  --mode direct --episodes "$EPISODES" \
  --seller-model "$QWEN_MODEL" --seller-base-url "$QWEN_URL" --seller-api-key-env NEGOTIATION_QWEN_API_KEY \
  --buyer-model "$GPT_MODEL" --buyer-api-key-env OPENAI_API_KEY \
  --output-dir "$OUTPUT_ROOT/qwen_seller__gpt4o_buyer" --resume

"$PYTHON_BIN" -m arena_integration.run_buysell \
  --mode direct --episodes "$EPISODES" \
  --seller-model "$GPT_MODEL" --seller-api-key-env OPENAI_API_KEY \
  --buyer-model "$QWEN_MODEL" --buyer-base-url "$QWEN_URL" --buyer-api-key-env NEGOTIATION_QWEN_API_KEY \
  --output-dir "$OUTPUT_ROOT/gpt4o_seller__qwen_buyer" --resume
