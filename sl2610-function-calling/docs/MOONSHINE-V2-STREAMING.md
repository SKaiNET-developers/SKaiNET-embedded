# Moonshine v2 streaming runtime — design (P6)

Low-latency, chunked ASR on the board (done-when #6). The model pieces are authored in the DSL; this is the
board-side **runtime** that drives them incrementally instead of on a fixed full-utterance clip.

## Pipeline (all pieces DSL-authored)
```
audio 20ms frames → [v2 encoder] → [v2 adapter] → [causal KV decoder] → provisional/finalized tokens
   (50 Hz)          sliding-window   +learned pos-embed   prefill-once + with_past-loop
                    causal (16,4/0)    (LayerNorm)         (reuse MoonshineKvDecoder)
```
- **v2 encoder** — `moonshineV2Encoder()` (transformers #244): position-free, sliding-window local attention,
  bounded lookahead (16,4)/(16,0). Streamable because each frame only needs 16 past + ≤4 future frames.
- **v2 adapter** — `MoonshineV2Adapter` (transformers #251): injects a learned positional embedding + norm,
  turning the position-free memory into decoder-ready memory.
- **decoder** — the existing **`MoonshineKvDecoder`** (`MOONSHINE_KV=1`): prefill-once emits self/cross K/V,
  then `with_past` steps one token over a growing self-cache. **No decoder change needed** — the adapter output
  is a drop-in for the encoder memory it already consumes.

## The streaming contract (from the v2 paper)
- **Frames:** 50 Hz (20 ms) feature frames from 16 kHz audio.
- **Bounded lookahead:** a frame's encoder output is **provisional** until its right-context (≤4 frames,
  0 for the causal-tail layers) has arrived; then it **finalizes**. Algorithmic latency ≈ 80–320 ms.
- **Rolling window:** as frames arrive, only the bounded window (16 back + lookahead) is (re)computed — O(Tw),
  not O(T²). Finalized frames never recompute.

## Runtime state machine (`MoonshineV2StreamingRunner`, to add under `src/linuxArm64Main/.../asr/`)
```
loop over incoming 20ms frames:
  1. append frame to a rolling buffer; run the conv frontend on the new frame(s).
  2. run the v2 encoder over the bounded window → provisional frame states; mark frames whose
     lookahead is complete as FINALIZED (append to the finalized encoder memory).
  3. run the adapter over the FINALIZED memory (positions = absolute frame indices).
  4. decode: MoonshineKvDecoder — prefill on the first finalized chunk, then `with_past` steps;
     re-feed cross-K/V as the finalized memory grows (or window it).
  5. emit PROVISIONAL transcript (from provisional states) for live captions; emit FINALIZED
     transcript when a segment ends (VAD boundary or silence).
```
- **Chunk-shaped graphs:** replace today's fixed `INPUT_LEN = 80000` (5 s) clip with **fixed-chunk** encoder +
  adapter vmfbs (e.g. N frames/chunk), compiled from the DSL exports. Decoder graphs are already incremental.
- **Reuse:** `MoonshineKvDecoder` (decode loop), `TorqRunModule` (vmfb driver), `Wav`/`Bin` (I/O). New code is
  the frame buffer + window/finalization bookkeeping + provisional/finalized emission — plug into
  `Pipeline.runListen` in place of the per-utterance `runPipeline`.

## Reuse of existing runtime
| Piece | Reused from | Role |
|---|---|---|
| decode loop | `MoonshineKvDecoder` (`MOONSHINE_KV=1`) | prefill + with_past over the growing self-cache |
| vmfb driver | `TorqRunModule` | runs encoder/adapter/decoder vmfbs (`torq` / `local-task`) |
| audio I/O | `Wav`, `Bin` | 16 kHz frames, raw-bin tensor I/O |

## Open dependencies (before this runs)
1. **v2 checkpoint** — weights for the v2 encoder + adapter + decoder (the architecture is authored; the exact
   configs — window/lookahead per layer, adapter form — are unconfirmed vs a release). Bake to `.bin`/npy.
2. **Compile the v2 graphs** to chunk-shaped vmfbs: `moonshineV2Encoder()` + `MoonshineV2Adapter` → StableHLO
   → `iree-compile` (CPU first; NPU via the Torq tiling of the **bounded window** — its O(Tw) attention should
   fit LRAM where v1's O(T²) couldn't, `docs/synaptics-support/README.md`).
3. **Board verification** — the provisional/finalized latency budget (≤ ~320 ms) + WER parity vs full-utterance
   on recorded wavs; then wire into `runListen` (and it kills the VAD-segmented full-utterance path, P3/4.1).

## Status
- ✅ v2 encoder (#244) + v2 adapter (#251) authored + traced to StableHLO.
- ◻ This runtime (scaffold pending the v2 vmfbs + board — steps above).
- ◻ NPU tiling of the bounded window.

Tracked as plan item **P6**. Board bring-up joins `BOARD-RUNBOOK.md` once the v2 vmfbs exist.
