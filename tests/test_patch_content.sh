#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_FILE="${ROOT}/patches/openstereo-v2-lightstereo-jetson.patch"
grep -q "mobilenetv2_100', pretrained=False" "${PATCH_FILE}"
grep -q "FoundationStereo': FoundationStereoTrainer" "${PATCH_FILE}"
grep -q "FastFoundationStereo': FastFoundationStereoTrainer" "${PATCH_FILE}"
