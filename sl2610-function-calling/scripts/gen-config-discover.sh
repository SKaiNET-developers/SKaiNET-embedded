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
DOCKER_DIR=$(cd "$(dirname "$0")/.docker" && pwd)   # static_onnx.py lives here
mkdir -p "$OUT"; OUT_ABS=$(cd "$OUT" && pwd)

docker run --rm \
  -v "$ONNX_DIR:/in:ro" -v "$OUT_ABS:/out" -v "$DOCKER_DIR:/tools:ro" \
  -e ONNX_BASE="$ONNX_BASE" -e FRAMES="$FRAMES" \
  "$IMAGE" bash -lc '
    set -e
    # 1) static-ONNX fix (standalone script, no heredoc escaping): freeze batch=1 + onnxsim.
    python /tools/static_onnx.py "/in/$ONNX_BASE" /out/enc_static.onnx "$FRAMES"
    # 2) discover the per-op NSS/host executor map. gen_config drives pytest on $OT/tests, so it
    #    MUST run from the release dir ($OT) for its pytest plugin (--model-path etc.) to load.
    #    Absolute --output-dir/--model keep the artifacts on the /out mount. Runs on the simulator.
    cd "$OT"
    python -m torq.gen_config discover --model /out/enc_static.onnx \
      --auto-convert-bf16 --save-bf16-model /out/enc_bf16.onnx --skip-mode \
      --output-dir /out --log-file /out/discover.log
    echo "[executor-map] $(ls /out/*_compiler.json 2>/dev/null || echo NONE)"
  '
