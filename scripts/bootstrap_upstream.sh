#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="${ROOT}/vendor/OpenStereo"
UPSTREAM_URL="https://github.com/XiandaGuo/OpenStereo.git"
UPSTREAM_COMMIT="23d71c9"
PATCH_FILE="${ROOT}/patches/openstereo-v2-lightstereo-jetson.patch"

if [[ "${1:-}" == "--upstream-dir" ]]; then
  UPSTREAM_DIR="$2"
fi

if [[ ! -d "${UPSTREAM_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${UPSTREAM_DIR}")"
  git clone "${UPSTREAM_URL}" "${UPSTREAM_DIR}"
fi

if ! git -C "${UPSTREAM_DIR}" rev-parse --verify --quiet "${UPSTREAM_COMMIT}^{commit}" >/dev/null; then
  git -C "${UPSTREAM_DIR}" fetch --quiet origin v2
fi
git -C "${UPSTREAM_DIR}" checkout --quiet --detach --force "${UPSTREAM_COMMIT}"

if git -C "${UPSTREAM_DIR}" apply --check "${PATCH_FILE}"; then
  git -C "${UPSTREAM_DIR}" apply "${PATCH_FILE}"
elif ! grep -q "mobilenetv2_100', pretrained=False" "${UPSTREAM_DIR}/stereo/modeling/models/lightstereo/backbone.py"; then
  echo "Patch cannot be applied and the expected LightStereo change is absent." >&2
  exit 1
fi

mkdir -p "${UPSTREAM_DIR}/tests"
cp "${ROOT}/tests/test_lightstereo_backbone.py" "${UPSTREAM_DIR}/tests/test_lightstereo_backbone.py"
echo "Bootstrapped patched OpenStereo at ${UPSTREAM_DIR}"
