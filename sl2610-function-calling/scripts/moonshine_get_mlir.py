"""Get a STATIC-shape Moonshine-tiny encoder StableHLO/torch MLIR for the NPU.

Pipeline (the prebuilt Synaptics torq vmfbs are version-stale on the updated
board, so we recompile from source):
  1. download encoder_model.onnx from onnx-community/moonshine-tiny-ONNX
  2. fix the dynamic input [batch,samples] -> [1, N] (N=64000 = 4s @16kHz ->
     165 frames <= max_position_embeddings 194); onnx shape-infer
  3. iree-import-onnx -> encoder_static.mlir (torch-onnx dialect, input [1,64000]
     f32 raw audio -> [1,165,288] features)

Then compile:
  CPU (works): iree-compile --iree-input-type=onnx --iree-hal-local-target-device-backends=llvm-cpu ...
  NPU (BLOCKED): scripts/iree-compile-torq.sh-style + --iree-input-type=onnx
     --torq-convert-dtypes -> torq compiler CRASHES:
     "getWeightMemoryFormat: Invalid weight memory format conversion" (segfault),
     after lowering ~20 dispatches. The MLIR is valid (CPU runs it, [1,165,288]
     real output) -> this is a Torq-compiler limitation on Moonshine's weights,
     not our graph. Needs: bisect the offending op/weight + model surgery, or
     Synaptics' exact onnx->torq conversion recipe, or a newer torq compiler.
"""
import sys, shutil, os
import onnx
from onnx import shape_inference
from huggingface_hub import hf_hub_download

DST = sys.argv[1] if len(sys.argv) > 1 else "/home/miso/projects/coral/build-mlir/moonshine"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 64000
os.makedirs(DST, exist_ok=True)

enc = hf_hub_download("onnx-community/moonshine-tiny-ONNX", "onnx/encoder_model.onnx")
shutil.copy(enc, f"{DST}/encoder.onnx")
m = onnx.load(f"{DST}/encoder.onnx")
d = m.graph.input[0].type.tensor_type.shape.dim
d[0].Clear(); d[0].dim_value = 1
d[1].Clear(); d[1].dim_value = N
del m.graph.value_info[:]
m = shape_inference.infer_shapes(m)
onnx.save(m, f"{DST}/encoder_static.onnx")
out = [(x.dim_value or x.dim_param) for x in m.graph.output[0].type.tensor_type.shape.dim]
print(f"encoder_static.onnx: input [1,{N}] -> output {out}")
print("now: iree-import-onnx encoder_static.onnx -o encoder_static.mlir")
