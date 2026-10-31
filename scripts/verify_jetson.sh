#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="${ROOT}/vendor/OpenStereo"
VENV="${ROOT}/.venv"
"${ROOT}/scripts/verify_checkpoint.sh"
[[ -x "${VENV}/bin/python" ]] || { echo "Run scripts/install_jetson.sh first." >&2; exit 1; }
[[ -d "${UPSTREAM_DIR}" ]] || { echo "Run scripts/bootstrap_upstream.sh first." >&2; exit 1; }
cd "${UPSTREAM_DIR}"
"${VENV}/bin/python" -m unittest tests.test_lightstereo_backbone -v
"${VENV}/bin/python" - "${ROOT}" <<'PY'
import sys
import torch
import yaml
from easydict import EasyDict
from stereo.modeling.models.lightstereo.lightstereo import LightStereo

root = sys.argv[1]
cfg = EasyDict(yaml.safe_load(open("cfgs/lightstereo/lightstereo_s_kitti.yaml")))
model = LightStereo(cfg.MODEL)
state = torch.load(f"{root}/checkpoints/LightStereo-S-KITTI.ckpt", map_location="cpu")["model_state"]
model.load_state_dict(state, strict=True)
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.get_device_capability() == (8, 7), torch.cuda.get_device_capability()
assert int((torch.tensor([1], device="cuda") + 1).item()) == 2
print("LightStereo-S offline checkpoint and CUDA verification: OK")
PY
