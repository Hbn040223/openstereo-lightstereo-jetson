#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ROOT}/.venv"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv) VENV="$2"; shift 2 ;;
    *) echo "Usage: $0 [--venv PATH]" >&2; exit 2 ;;
  esac
done

[[ "$(uname -m)" == "aarch64" ]] || { echo "Jetson Orin NX requires aarch64." >&2; exit 1; }
grep -q "R36 (release), REVISION: 4" /etc/nv_tegra_release || {
  echo "Supported baseline is JetPack 6.2.1 / L4T R36.4." >&2; exit 1;
}
python3 -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version' || {
  echo "Python 3.10 is required." >&2; exit 1;
}

python3 -m venv "${VENV}"
"${VENV}/bin/python" -m pip install --upgrade pip
"${VENV}/bin/python" -m pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 torch==2.8.0 torchvision==0.23.0
"${VENV}/bin/python" -m pip install -r "${ROOT}/requirements-jetson.txt"
"${VENV}/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.get_device_capability() == (8, 7), torch.cuda.get_device_capability()
assert int((torch.tensor([1], device="cuda") + 1).item()) == 2
print("Jetson CUDA preflight: OK")
PY
"${VENV}/bin/python" -m pip freeze > "${ROOT}/.jetson-install-report.txt"
