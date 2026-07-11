# Example: add a custom command to FunctionGemma

A minimal, forkable walk-through of `docs/FINETUNING.md`. It teaches the demo one **new tool** —
`open_door` on the spare functional token `<tool_6>` — end to end: dataset → LoRA → GGUF → re-bake →
wire codec+router → verify.

Files here:
- `train.jsonl` — ~20 rows: the new command (paraphrased) + a few base commands to prevent regression.
- `router-patch.md` — the two code edits (codec mapping + router handler) to apply after training.

## Run it

```bash
# 1. train + convert (offline Python — see docs/FINETUNING.md steps 2-3)
python finetune.py            # uses examples/custom-command/train.jsonl -> functiongemma-mine
python convert_hf_to_gguf.py functiongemma-mine --outfile fg-mine-f16.gguf --outtype f16
./llama-quantize fg-mine-f16.gguf fg-mine-Q5_K_M.gguf Q5_K_M

# 2. wire the new tool into the app (apply router-patch.md), then re-bake (unchanged path)
GEMMA_GGUF=$PWD/fg-mine-Q5_K_M.gguf scripts/compile-gemma.sh board
BOARD=root@<ip> ./gradlew deployBoard

# 3. verify
voicecc gen "open the front door"     # -> <tool_6>(which="front")<end> -> open_door which=front
voicecc gen "turn the light on"       # base command still works (no regression)
```

If `<tool_6>` is **not** a spare token in your FunctionGemma vocab, follow the "Advanced: new tool with no
spare functional token" section of `docs/FINETUNING.md` (tokenizer extend + embedding resize) before step 1.
