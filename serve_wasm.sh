#!/usr/bin/env bash
set -euo pipefail

# Serve the Emscripten build directory over HTTP.
# Usage (from repo root):
#   ./serve_wasm.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_WASM_DIR="${ROOT_DIR}/build-wasm"

if [ ! -d "${BUILD_WASM_DIR}" ]; then
  echo "[ERROR] build-wasm directory does not exist. Run ./build_all.sh first." >&2
  exit 1
fi

cd "${BUILD_WASM_DIR}"
python3 -m http.server 8000
