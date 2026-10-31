#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${ROOT}/checkpoints/LightStereo-S-KITTI.ckpt"
RECOVER=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --recover) RECOVER=1; shift ;;
    *) echo "Usage: $0 [--checkpoint PATH] [--recover]" >&2; exit 2 ;;
  esac
done

EXPECTED_SHA256="3d768e0344c2b8bfacb8f7f27cc647cd338e5ba93ec66d944a9a73fd63ec9b2a"
actual_sha256() {
  local path="${1:-${CHECKPOINT}}"
  [[ -f "${path}" ]] || return 1
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${EXPECTED_SHA256}" ]]
}

if ! actual_sha256; then
  if [[ ${RECOVER} -ne 1 ]]; then
    echo "Checkpoint checksum failed. Re-run with --recover to download the official file." >&2
    exit 1
  fi
  mkdir -p "$(dirname "${CHECKPOINT}")"
  curl --fail --location --continue-at - --output "${CHECKPOINT}.part" \
    "https://huggingface.co/XiandaGuo/OpenStereo/resolve/main/checkpoint/LightStereo/LightStereo-S-KITTI.ckpt"
  actual_sha256 "${CHECKPOINT}.part" || { echo "Recovery download checksum failed." >&2; exit 1; }
  mv "${CHECKPOINT}.part" "${CHECKPOINT}"
fi
actual_sha256 || { echo "Checkpoint checksum failed." >&2; exit 1; }
echo "${EXPECTED_SHA256}  ${CHECKPOINT}"
