#!/usr/bin/env python3
"""
Checkpoint prep (HOST build tooling, not the runtime): convert a moonshine-tiny safetensors
checkpoint into per-tensor little-endian f32 `.bin` files for the demo's weight baker.

`voicecc.export.DirBinWeightSource` reads `<dir>/<hf-tensor-name>.bin` as little-endian f32,
so this writes exactly that: one file per tensor, named by its (prefix-stripped) HF name.

Usage
-----
  # 1) LIST tensor names/shapes/dtypes — verify against MoonshineWeights.hfNameFor():
  python scripts/convert_moonshine_weights.py model.safetensors --list

  # 2) WRITE the .bin files:
  python scripts/convert_moonshine_weights.py model.safetensors --out weights/

Then feed the demo's self-compile:
  CHECKPOINT=weights ENC_FRAMES=207 ENC_BOARD_LAYOUT=1 ./gradlew moonshineEncoderMlir

Requires: `pip install safetensors numpy`. (bf16 checkpoints need torch — see --framework.)
"""
import argparse
import os
import sys


def open_checkpoint(path, framework):
    from safetensors import safe_open
    return safe_open(path, framework=framework)


def to_f32_le(tensor, framework):
    import numpy as np
    if framework == "pt":
        tensor = tensor.float().cpu().numpy()   # handles bf16/f16 → f32
    return np.asarray(tensor).astype("<f4")      # little-endian float32


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("safetensors", help="path to the moonshine-tiny .safetensors")
    ap.add_argument("--out", default=None, help="output dir for the .bin files (required unless --list)")
    ap.add_argument("--list", action="store_true", dest="list_only", help="print tensor names/shapes/dtypes and exit")
    ap.add_argument("--strip-prefix", default="model.",
                    help="prefix stripped from tensor names so they match hfNameFor (default 'model.')")
    ap.add_argument("--framework", default="numpy", choices=["numpy", "pt"],
                    help="'numpy' (f32/f16 checkpoints) or 'pt' (needs torch; required for bf16)")
    args = ap.parse_args()

    def name_of(key):
        p = args.strip_prefix
        return key[len(p):] if p and key.startswith(p) else key

    with open_checkpoint(args.safetensors, args.framework) as f:
        keys = list(f.keys())

        if args.list_only:
            for k in keys:
                t = f.get_tensor(k)
                shape = list(getattr(t, "shape", ()))
                dtype = getattr(t, "dtype", "?")
                print(f"{name_of(k)}\t{shape}\t{dtype}")
            print(f"# {len(keys)} tensors in {args.safetensors}", file=sys.stderr)
            return

        if not args.out:
            ap.error("--out DIR is required unless --list")
        os.makedirs(args.out, exist_ok=True)

        n = 0
        for k in keys:
            arr = to_f32_le(f.get_tensor(k), args.framework)
            arr.tofile(os.path.join(args.out, name_of(k) + ".bin"))
            n += 1
        print(f"wrote {n} tensors → {args.out} (little-endian f32 .bin, one per tensor)")


if __name__ == "__main__":
    main()
