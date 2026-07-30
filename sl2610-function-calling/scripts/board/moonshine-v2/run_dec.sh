#!/bin/sh
cd /home/root/moon
RM=/home/root/torq2-stable/torq/_runtime_libs/torq-run-module
export LD_LIBRARY_PATH=/home/root/torq2-stable/iree/_runtime_libs:/home/root/torq2-stable/torq/_runtime_libs

echo "=== PREFILL ==="
PO=""
i=0; while [ $i -lt 25 ]; do PO="$PO --output=@pre_bout$i.bin"; i=$((i+1)); done
$RM --device=local-task --module=pre_masked_arm64.vmfb --function=moonshine_v2_decoder_prefill \
  --input=1x1x320xf32=@pre_in0.bin --input=1x96x320xf32=@pre_in1.bin --input=1x1x1x96xf32=@pre_in2.bin \
  $PO 2>&1 | tail -2

echo "=== WITH_PAST (past=1) ==="
# arg order: e, cq, sq, then per-layer [sK,sV,cK,cV] with mask after L0
WI="--input=1x1x320xf32=@wp_in0.bin --input=1x40xf32=@wp_in1.bin --input=1x40xf32=@wp_in2.bin"
WI="$WI --input=1x8x1x40xf32=@wp_in3.bin --input=1x8x1x40xf32=@wp_in4.bin --input=1x8x96x40xf32=@wp_in5.bin --input=1x8x96x40xf32=@wp_in6.bin --input=1x1x1x96xf32=@wp_in7.bin"
n=8; l=1
while [ $l -lt 6 ]; do
  WI="$WI --input=1x8x1x40xf32=@wp_in$n.bin"; n=$((n+1))
  WI="$WI --input=1x8x1x40xf32=@wp_in$n.bin"; n=$((n+1))
  WI="$WI --input=1x8x96x40xf32=@wp_in$n.bin"; n=$((n+1))
  WI="$WI --input=1x8x96x40xf32=@wp_in$n.bin"; n=$((n+1))
  l=$((l+1))
done
WO=""; i=0; while [ $i -lt 13 ]; do WO="$WO --output=@wp_bout$i.bin"; i=$((i+1)); done
$RM --device=local-task --module=wp_masked_arm64.vmfb --function=moonshine_v2_decoder_with_past \
  $WI $WO 2>&1 | tail -2
echo "=== done; outputs ==="
ls -la pre_bout24.bin wp_bout12.bin
