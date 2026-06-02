#!/usr/bin/env python3
"""Synthesize a command wav for the demo (host-side, Piper TTS — no mic needed).
  python3.12 -m pip install --target <pkg> piper-tts
  voice: rhasspy/piper-voices en/en_US/lessac/medium/en_US-lessac-medium.onnx(+.json)
Usage: PYTHONPATH=<piperpkg> python3.12 make_command_wav.py "Turn the light on." out.wav <voice.onnx>

Verified end-to-end on the SL2610 (native Kotlin):
  cmd.wav "Turn the light on." -> ASR(Moonshine/NPU) "Turn the light on"
  -> FunctionGemma (our vmfb) <tool_0>(state="on") -> set_lights(state=on) [ok]
"""
import sys, wave
from piper import PiperVoice

text = sys.argv[1] if len(sys.argv) > 1 else "Turn the light on."
out = sys.argv[2] if len(sys.argv) > 2 else "cmd.wav"
voice = sys.argv[3] if len(sys.argv) > 3 else "en_US-lessac-medium.onnx"
v = PiperVoice.load(voice)
with wave.open(out, "wb") as w:
    v.synthesize_wav(text, w)
print(f"wrote {out}: {text!r}")
