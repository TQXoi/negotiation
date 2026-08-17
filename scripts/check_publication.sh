#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SECRET_PATTERN='sk-(proj-)?[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{20,}|AIza[0-9A-Za-z_-]{20,}|AKIA[0-9A-Z]{16}'

if rg -n --hidden \
  -g '!**/.git/**' -g '!**/__pycache__/**' -g '!**/*.pyc' \
  -e "${SECRET_PATTERN}" .; then
  printf 'Potential secret detected. Refusing publication.\n' >&2
  exit 1
fi

large_files="$(find . -type f -size +25M -not -path './.git/*' -print)"
if [[ -n "${large_files}" ]]; then
  printf 'Files larger than 25 MiB require manual review:\n%s\n' "${large_files}" >&2
  exit 1
fi

git diff --check
git diff --cached --check
printf 'Publication checks passed.\n'
