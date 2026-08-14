#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export METHODS="${METHODS:-opponent_simulation_paper framework_v4 framework_v4_frozen framework_v4_wrong_confident framework_v4_shuffled}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-arena_runs/opponent_switch_matrix_qwen30b_5x20_20260812}"
export SWITCH_EPISODE="${SWITCH_EPISODE:-11}"
export SWITCH_POLICY="${SWITCH_POLICY:-direct}"
export RUNS="${RUNS:-5}"
export EPISODES_PER_RUN="${EPISODES_PER_RUN:-20}"

bash scripts/run_formal_belief_matrix_qwen30b.sh
