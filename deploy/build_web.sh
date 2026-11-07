#!/usr/bin/env bash
# Assemble the static in-browser site in site/public (then: cd site && npm run build && npx wrangler deploy).
# Fetches build artifacts instead of committing them: ONNX Runtime Web (jsDelivr), the ONNX
# models (Hugging Face) and the resenh wheel (PyPI). Usage: bash deploy/build_web.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ORT_VER=1.20.1
RESENH_VER=$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)
HF=https://huggingface.co/SharadhNaiduTrains/resenh/resolve/main/web
mkdir -p site/public/ort site/public/models site/public/py
for f in ort.webgpu.min.js ort-wasm-simd-threaded.jsep.mjs ort-wasm-simd-threaded.jsep.wasm; do
  curl -sfL -o "site/public/ort/$f" "https://cdn.jsdelivr.net/npm/onnxruntime-web@$ORT_VER/dist/$f"
done
for m in resenh-web-hybrid resenh-web-dl; do curl -sfL -o "site/public/models/$m.onnx" "$HF/$m.onnx"; done
WHL="resenh-$RESENH_VER-py3-none-any.whl"
curl -sfL -o "site/public/py/$WHL" "https://files.pythonhosted.org/packages/py3/r/resenh/$WHL"
sed -i "s#py/resenh-[^"]*-py3-none-any.whl#py/$WHL#" site/src/lib/engine.ts
ls -la site/public/ort site/public/models site/public/py
