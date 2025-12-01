#!/usr/bin/env bash
set -euo pipefail

# Build WebAssembly (Emscripten) target without ImGui debug/control windows.
# This script is intended for exporting the GPU-only visualization to the web.

if ! command -v emcmake >/dev/null 2>&1; then
  echo "[ERROR] emcmake not found in PATH. Activate your Emscripten SDK (emsdk_env) first." >&2
  exit 1
fi

BUILD_DIR="${1:-build-wasm}"

emcmake cmake -S . -B "${BUILD_DIR}" -DNNDEMO_ENABLE_IMGUI=OFF
cmake --build "${BUILD_DIR}"
