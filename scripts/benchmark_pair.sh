#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="${ROOT}/vendor/OpenStereo"
VENV="${ROOT}/.venv"
CHECKPOINT="${ROOT}/checkpoints/LightStereo-S-KITTI.ckpt"
GPU=0
LEFT=""; RIGHT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --left) LEFT="$2"; shift 2 ;;
    --right) RIGHT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    *) echo "Usage: $0 --left PATH --right PATH [--gpu ID] [--checkpoint PATH]" >&2; exit 2 ;;
  esac
done
[[ -f "${LEFT}" && -f "${RIGHT}" ]] || { echo "Left and right paths are required." >&2; exit 2; }
"${ROOT}/scripts/verify_checkpoint.sh" --checkpoint "${CHECKPOINT}"
cd "${UPSTREAM_DIR}"
PYTHONPATH="${UPSTREAM_DIR}" CUDA_VISIBLE_DEVICES="${GPU}" "${VENV}/bin/python" - "${LEFT}" "${RIGHT}" "${CHECKPOINT}" <<'PY'
import sys, time
import numpy as np
import torch, yaml
from easydict import EasyDict
from PIL import Image
from stereo.datasets.dataset_template import build_transform_by_cfg
from stereo.modeling.models.lightstereo.lightstereo import LightStereo

left_path, right_path, checkpoint = sys.argv[1:]
cfg = EasyDict(yaml.safe_load(open("cfgs/lightstereo/lightstereo_s_kitti.yaml")))
transform = build_transform_by_cfg(cfg.DATA_CONFIG.DATA_TRANSFORM.EVALUATING)
sample = transform({"left": np.array(Image.open(left_path).convert("RGB"), dtype=np.float32), "right": np.array(Image.open(right_path).convert("RGB"), dtype=np.float32)})
sample = {key: value.unsqueeze(0).cuda() for key, value in sample.items()}
model = LightStereo(cfg.MODEL)
model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model_state"], strict=True)
model.cuda().eval()
with torch.no_grad():
    model(sample); torch.cuda.synchronize()
    timings = []
    for _ in range(5):
        torch.cuda.synchronize(); start = time.perf_counter(); model(sample); torch.cuda.synchronize(); timings.append(time.perf_counter() - start)
mean = sum(timings) / len(timings)
print("forward_seconds=" + ",".join(f"{value:.3f}" for value in timings))
print(f"mean_forward_seconds={mean:.3f}")
print(f"mean_fps={1 / mean:.2f}")
PY
