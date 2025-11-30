#!/usr/bin/env bash
set -euo pipefail

# Build native (desktop) target
cmake -S . -B build
cmake --build build

# Build WebAssembly (Emscripten) target
if command -v emcmake >/dev/null 2>&1; then
  emcmake cmake -S . -B build-wasm
  cmake --build build-wasm
else
  echo "[WARN] emcmake not found in PATH; skipping wasm build" >&2
  echo "       Activate your Emscripten environment, then re-run this script." >&2
fi
