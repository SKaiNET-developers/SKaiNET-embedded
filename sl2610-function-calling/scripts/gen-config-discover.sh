#!/usr/bin/env bash
# Derive the Torq per-op executor map (softmax→host) for the Moonshine encoder, in Docker.
# Runs the vendor `torq.gen_config discover` on the SL2610 simulator (no board needed) and emits
# the executor-map JSON that feeds `iree-compile --torq-executor-map=…`.
#
#   scripts/gen-config-discover.sh <enc_xformer.onnx> <out-dir>
#
# Needs the gen_config image:
#   docker build -f scripts/.docker/Dockerfile.genconfig -t sl2610-genconfig:v2.0.0 scripts/.docker
#
# Notes (from the working run): the HF export has a dynamic pos-id Range/Shape that the compiler
# keeps dynamic → freeze batch=1 + onnxsim to fold it away before discover. ~15–25 min.
set -euo pipefail

ONNX=${1:?usage: gen-config-discover.sh <enc_xformer.onnx> <out-dir>}
OUT=${2:?usage: gen-config-discover.sh <enc_xformer.onnx> <out-dir>}
IMAGE=${GENCONFIG_IMAGE:-sl2610-genconfig:v2.0.0}
FRAMES=${ENC_FRAMES:-165}

ONNX_DIR=$(cd "$(dirname "$ONNX")" && pwd); ONNX_BASE=$(basename "$ONNX")
mkdir -p "$OUT"; OUT_ABS=$(cd "$OUT" && pwd)

docker run --rm \
  -v "$ONNX_DIR:/in:ro" -v "$OUT_ABS:/out" \
  -e ONNX_BASE="$ONNX_BASE" -e FRAMES="$FRAMES" \
  "$IMAGE" bash -lc '
    set -e
    # 1) static-ONNX fix: freeze batch=1 + simplify (folds the dynamic pos-id Range/Shape).
    python - <<PY
import onnx, onnxsim, os
m = onnx.load(f"/in/{os.environ[\"ONNX_BASE\"]}")
inp = m.graph.input[0].name
dims = m.graph.input[0].type.tensor_type.shape.dim
feat = dims[2].dim_value or 288
shape = [1, int(os.environ["FRAMES"]), feat]
ms, ok = onnxsim.simplify(m, overwrite_input_shapes={inp: shape})
assert ok, "onnxsim simplify failed"
onnx.save(ms, "/out/enc_static.onnx")
print(f"[static] {inp}={shape}  {len(m.graph.node)}→{len(ms.graph.node)} nodes")
PY
    # 2) discover the per-op NSS/host executor map (bf16 auto-convert; runs on the simulator).
    python -m torq.gen_config discover --model /out/enc_static.onnx \
      --auto-convert-bf16 --save-bf16-model /out/enc_bf16.onnx --skip-mode \
      --output-dir /out --log-file /out/discover.log
    echo "[executor-map] $(ls /out/*_compiler.json 2>/dev/null || echo NONE)"
  '
