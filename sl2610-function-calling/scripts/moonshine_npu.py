#!/usr/bin/env python3
"""Transcribe a wav with Moonshine on the Torq NPU via the prebuilt vmfbs +
the version-matched torq.runtime (what the Python sample uses). Called as a
subprocess by the Kotlin/Native app (option 1 binding). Prints the transcript
to stdout as: TRANSCRIPT\t<text>

Usage (on board, with the sample's venv python):
  python3 moonshine_npu.py <wav> [model_dir] [vendor_dir]
"""
import sys, os, wave
import numpy as np

WAV = sys.argv[1]
MODEL_DIR = sys.argv[2] if len(sys.argv) > 2 else "/home/root/sl2610-examples/models/Synaptics/moonshine-tiny-bf16-torq"
VENDOR = sys.argv[3] if len(sys.argv) > 3 else "/home/root/sl2610-voice-cc/vendor"
sys.path.insert(0, VENDOR)

def load_wav_16k(path):
    w = wave.open(path, "rb")
    sr, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
    raw = w.readframes(n); w.close()
    dt = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    a = np.frombuffer(raw, dtype=dt).astype(np.float32)
    if ch > 1: a = a.reshape(-1, ch).mean(axis=1)
    a /= float(np.iinfo(dt).max)              # -> [-1,1]
    if sr != 16000:                            # linear resample to 16k
        t = np.arange(a.size) / sr
        nt = np.arange(0, t[-1], 1.0 / 16000)
        a = np.interp(nt, t, a).astype(np.float32)
    return a

audio = load_wav_16k(WAV)
from utils.moonshine import load_moonshine
runner = load_moonshine(model_path=MODEL_DIR, model_name="tiny-en")
toks = runner.run(audio.reshape(1, -1))
toks = np.asarray(toks).reshape(-1).tolist()

# detokenize with the bundled tokenizer.json
from tokenizers import Tokenizer
tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
text = tok.decode([int(t) for t in toks], skip_special_tokens=True)
print("TOKENS\t" + ",".join(str(int(t)) for t in toks))
print("TRANSCRIPT\t" + text.strip())
