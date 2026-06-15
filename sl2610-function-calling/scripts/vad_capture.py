#!/usr/bin/env python3
"""Light capture + Silero VAD helper (NO Moonshine — keeps RAM low so the LLM
gen subprocess fits the 1.9GB board). Streams one wav per detected utterance and
prints `SEGMENT\t<wav>` (flushed). The Kotlin `listen` mode runs the full
ASR->LLM->action pipeline on each segment, sequentially.

Reuses the sample's SileroSpeechSegmenter (silero_vad_notorch).
  --source mic [--device <sel>]   live mic via sounddevice
  --source <wav>                  replay a wav through the VAD (for testing)
"""
import sys, os, wave, time
import numpy as np

VENDOR = os.environ.get("VENDOR", "/home/root/sl2610-voice-cc/vendor")
sys.path.insert(0, VENDOR)
from utils.speech import SileroSpeechSegmenter, SAMPLING_RATE, CHUNK_SIZE  # noqa: E402

src = "mic"
device = None
once = False  # exit after the first detected utterance (frees the ~250MB Silero
              # VAD + the mic so the LLM gen subprocess gets the full 1.9GB board)
args = sys.argv[1:]
i = 0
while i < len(args):
    if args[i] == "--source": src = args[i + 1]; i += 2
    elif args[i] == "--device": device = args[i + 1]; i += 2
    elif args[i] == "--once": once = True; i += 1
    else: i += 1

OUT = "/tmp"
seg_i = 0
seg = SileroSpeechSegmenter(sample_rate=SAMPLING_RATE, chunk_size=CHUNK_SIZE)

def emit(segment):
    global seg_i
    a = np.asarray(segment.audio, dtype=np.float32).reshape(-1)
    pcm = np.clip(a, -1, 1)
    pcm = (pcm * 32767).astype("<i2")
    path = f"{OUT}/seg_{seg_i}.wav"; seg_i += 1
    w = wave.open(path, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLING_RATE)
    w.writeframes(pcm.tobytes()); w.close()
    print(f"SEGMENT\t{path}", flush=True)

if src == "mic":
    import sounddevice as sd
    from utils.speech import SoundDeviceAudioSource
    dev = device
    if dev is not None and dev.lstrip("-").isdigit(): dev = int(dev)
    if dev is None:
        # No --device given: auto-pick a real capture device. The board's ALSA
        # "default" routes to the klamath i2s card (no mic attached -> silence),
        # so prefer a USB-audio input (e.g. the C920 webcam mic) when present.
        devs = sd.query_devices()
        cap = [i for i, d in enumerate(devs) if d["max_input_channels"] > 0]
        usb = [i for i in cap if any(k in devs[i]["name"].lower()
                                     for k in ("usb", "webcam", "c920"))]
        if usb or cap:
            dev = (usb or cap)[0]
            print(f"auto-selected input device {dev}: {devs[dev]['name']}", flush=True)
    print("LISTENING (mic). Speak a command...", flush=True)
    sa = SoundDeviceAudioSource(device=dev, sample_rate=SAMPLING_RATE, chunk_size=CHUNK_SIZE)
    sa.start()
    try:
        while True:
            # Don't leak: if the Kotlin parent died (reparented to init), exit so
            # an orphaned capture can't keep holding the mic + RAM.
            if os.getppid() == 1: break
            c = sa.read_chunk(timeout_s=0.2)
            if c is None: continue
            s = seg.feed(c[0])
            if s is not None:
                emit(s)
                if once: break  # release mic + VAD RAM; caller runs the pipeline then re-listens
    finally:
        sa.stop()
else:  # wav replay
    w = wave.open(src, "rb"); sr = w.getframerate(); n = w.getnframes()
    raw = w.readframes(n); w.close()
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    if sr != SAMPLING_RATE:
        t = np.arange(a.size) / sr; nt = np.arange(0, t[-1], 1.0 / SAMPLING_RATE)
        a = np.interp(nt, t, a).astype(np.float32)
    # Append ~0.6s of trailing silence so the VAD detects end-of-speech and
    # closes a segment that runs to EOF (clean TTS clips have no trailing
    # silence; a live mic does). The real mic path needs no such padding.
    a = np.concatenate([a, np.zeros(int(0.6 * SAMPLING_RATE), dtype=np.float32)])
    for off in range(0, a.size, CHUNK_SIZE):
        chunk = a[off:off + CHUNK_SIZE]
        if chunk.size < CHUNK_SIZE: chunk = np.pad(chunk, (0, CHUNK_SIZE - chunk.size))
        s = seg.feed(chunk)
        if s is not None: emit(s)
print("DONE", flush=True)
