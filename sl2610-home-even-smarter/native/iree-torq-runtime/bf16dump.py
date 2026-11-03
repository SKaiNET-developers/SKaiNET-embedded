# Print bf16/f32 binary output files as floats (board has BusyBox od without -A).
import struct, sys
kind, path = sys.argv[1], sys.argv[2]
b = open(path, "rb").read()
if kind == "bf16":
    v = [struct.unpack("<f", struct.pack("<I", h << 16))[0] for (h,) in struct.iter_unpack("<H", b)]
else:
    v = [x for (x,) in struct.iter_unpack("<f", b)]
print(len(v), "values; first", v[:8], "unique", sorted(set(v))[:8])
