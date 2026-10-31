#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="${ROOT}/vendor/OpenStereo"
VENV="${ROOT}/.venv"
CHECKPOINT="${ROOT}/checkpoints/LightStereo-S-KITTI.ckpt"
GPU=0
LEFT=""; RIGHT=""; OUTPUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --left) LEFT="$2"; shift 2 ;;
    --right) RIGHT="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    *) echo "Usage: $0 --left PATH --right PATH --output PATH [--gpu ID] [--checkpoint PATH]" >&2; exit 2 ;;
  esac
done
[[ -f "${LEFT}" && -f "${RIGHT}" && -n "${OUTPUT}" ]] || { echo "Left, right and output paths are required." >&2; exit 2; }
"${ROOT}/scripts/verify_checkpoint.sh" --checkpoint "${CHECKPOINT}"
"${VENV}/bin/python" - "${LEFT}" "${RIGHT}" <<'PY'
import sys
from PIL import Image
left, right = (Image.open(path) for path in sys.argv[1:])
assert left.size == right.size, f"Image sizes differ: {left.size} vs {right.size}"
PY
mkdir -p "$(dirname "${OUTPUT}")"
cd "${UPSTREAM_DIR}"
CUDA_VISIBLE_DEVICES="${GPU}" "${VENV}/bin/python" tools/infer.py \
  --cfg_file cfgs/lightstereo/lightstereo_s_kitti.yaml \
  --left_img_path "${LEFT}" --right_img_path "${RIGHT}" \
  --pretrained_model "${CHECKPOINT}" --savename "${OUTPUT}"
