#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"${ROOT}/scripts/verify_checkpoint.sh"

bad="$(mktemp)"
printf 'bad checkpoint' > "${bad}"
if "${ROOT}/scripts/verify_checkpoint.sh" --checkpoint "${bad}"; then
  echo "Corrupt checkpoint unexpectedly passed verification." >&2
  exit 1
fi
rm -f "${bad}"
