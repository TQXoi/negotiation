#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGENTICPAY_DIR="${REPO_ROOT}/benchmarks/AgenticPay"
CASINO_DIR="${REPO_ROOT}/external_upstreams/CaSiNo"

mkdir -p "${REPO_ROOT}/benchmarks" "${REPO_ROOT}/external_upstreams"

if [[ ! -d "${AGENTICPAY_DIR}/.git" ]]; then
  git clone https://github.com/SafeRL-Lab/AgenticPay.git "${AGENTICPAY_DIR}"
fi

if [[ ! -d "${CASINO_DIR}/.git" ]]; then
  git clone https://github.com/kushalchawla/CaSiNo.git "${CASINO_DIR}"
fi

python -m CaSiNo_Env.tools.prepare_casino_data \
  --source-dir "${CASINO_DIR}/data" \
  --output-dir "${REPO_ROOT}/CaSiNo_Env/data/casino"

printf 'Upstreams ready under %s\n' "${REPO_ROOT}"
