#!/usr/bin/env bash
set -euo pipefail

# Build WebAssembly (Emscripten) target with ImGui visualization windows
# (network diagram, loss, accuracy) but no on-canvas control knobs. All
# controls are driven via the C API from JavaScript.

if ! command -v emcmake >/dev/null 2>&1; then
  echo "[ERROR] emcmake not found in PATH. Activate your Emscripten SDK (emsdk_env) first." >&2
  exit 1
fi

BUILD_DIR="${1:-build-wasm}"

emcmake cmake -S . -B "${BUILD_DIR}" -DNNDEMO_ENABLE_IMGUI=ON
cmake --build "${BUILD_DIR}"
