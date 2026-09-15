#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SECRET_PATTERN='sk-(proj-)?[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{20,}|AIza[0-9A-Za-z_-]{20,}|AKIA[0-9A-Z]{16}'

# Scan only publishable files: already tracked files plus untracked files that
# are not ignored. Local runs/checkpoints may contain private model traces and
# must remain outside the scan and outside Git. Never print a matching secret.
mapfile -d '' PUBLISH_FILES < <(git ls-files -z --cached --others --exclude-standard)
for candidate in "${PUBLISH_FILES[@]}"; do
  [[ -f "${candidate}" ]] || continue
  if [[ $(stat -c '%s' -- "${candidate}") -gt 26214400 ]]; then
    printf 'Publishable file exceeds 25 MiB: %s\n' "${candidate}" >&2
    exit 1
  fi
  if [[ "${candidate}" == *.pptx ]]; then
    if ! command -v unzip >/dev/null 2>&1; then
      printf 'Cannot inspect PPTX without unzip: %s\n' "${candidate}" >&2
      exit 1
    fi
    if (set +o pipefail; unzip -p -- "${candidate}" | rg -q -e "${SECRET_PATTERN}"); then
      printf 'Potential secret in PPTX: %s\n' "${candidate}" >&2
      exit 1
    fi
  elif rg -q -e "${SECRET_PATTERN}" -- "${candidate}"; then
    printf 'Potential secret in publishable file: %s\n' "${candidate}" >&2
    exit 1
  fi
done

git diff --check
git diff --cached --check
printf 'Publication checks passed.\n'
