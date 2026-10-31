#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
test -f checkpoints/LightStereo-S-KITTI.ckpt
test -f checkpoints/SHA256SUMS
! find . -path './.git' -prune -o -path './outputs' -print | grep -q .
! find . -path './.git' -prune -o -path './datasets' -print | grep -q .
! find . -path './.git' -prune -o -name '*.before_*' -print | grep -q .
! find . -path './.git' -prune -o -name 'id_ed25519' -print | grep -q .
