# Teaching FunctionGemma your own commands

The whole point of this port: once it clones-and-runs, you retrain the LLM to understand **your**
commands and re-bake into the same demo. **The re-bake is literally "swap the GGUF"** — the entire
self-compile → board path (`compile-gemma.sh` → `deployBoard`) is unchanged. This doc is the loop.

Training itself is offline Python (HF PEFT / llama.cpp) — that's fine, it's *training*, not the on-device
runtime, which stays Python-free.

## Mental model (read this first)

FunctionGemma's trick (Octopus-v2) is **functional tokens**: each tool is a *single special token*, so no
JSON schema is injected into the prompt. Inference is just:

```
prompt:      <start_of_turn>user\n{your instruction}<end_of_turn>\n<start_of_turn>model\n
completion:  <tool_N>(arg="value" ...)<end>
```

The demo's three moving parts you may touch:

| Part | Where | Role |
|---|---|---|
| Functional token → tool name | `CompactCodec.TOKEN_TO_NAME` (upstream `gemma-iree`) | `<tool_0>` → `set_lights`, … `<tool_5>` → `respond` |
| Tool name → handler | `LoggingActions.handlers()` / `defaultRouter()` (`src/commonMain/.../actions/Actions.kt`) | run the action |
| Prompt template | `GemmaDecoder.generate()` (upstream) | the Gemma chat wrapper above |

Current tools & tokens: `0 set_lights · 1 play_buzzer · 2 set_alarm · 3 cancel_alarm · 4 get_system_status · 5 respond`.

## Pick your case

**Case A — new phrasings / new args, or map new commands onto the existing tool set.** No vocab change,
pure LoRA. This covers most "understand my commands" needs (e.g. teach it that *"blackout"* → 
`set_lights(state="off")`, or add a `brightness` arg to `set_lights`). **Start here.**

**Case B — a genuinely new tool** (a new device/action that no existing `<tool_N>` covers). Needs a spare
functional token. FunctionGemma reserves a block of `<tool_N>` special tokens; if a spare exists (e.g.
`<tool_6>`), reuse it — still no vocab surgery, just LoRA + wire it in the codec/router. If none is spare,
you must extend the tokenizer + resize embeddings (advanced — see the bottom).

---

## The loop

### 1. Author the dataset

One JSONL row per example. Match the prompt template **exactly** (the `<start_of_turn>` wrapper is what the
model was trained on). Example `train.jsonl`:

```json
{"instruction": "blackout the room", "completion": "<tool_0>(state=\"off\")<end>"}
{"instruction": "kill the lights", "completion": "<tool_0>(state=\"off\")<end>"}
{"instruction": "set the lamp to 20 percent", "completion": "<tool_0>(state=\"on\" brightness=\"20\")<end>"}
{"instruction": "is the board hot?", "completion": "<tool_4>(metric=\"temperature\")<end>"}
```

Guidance: 10–50 rows per new phrase/tool is usually enough for LoRA on a 270M model; include a few
paraphrases each and a handful of *negative*/existing commands so you don't regress the base 6 tools.
Keep arg names consistent with what your handler reads (`Actions.kt`).

### 2. Train (LoRA, offline)

```python
# finetune.py — LoRA on the base FunctionGemma. Offline; needs a GPU or patience.
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import json, torch

BASE = "path/to/functiongemma-base"           # the HF checkpoint the GGUF was made from
tok  = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"],
                                         lora_dropout=0.05, task_type="CAUSAL_LM"))

def fmt(r):
    text = f'<start_of_turn>user\n{r["instruction"]}<end_of_turn>\n<start_of_turn>model\n{r["completion"]}'
    ids = tok(text, truncation=True, max_length=128)
    ids["labels"] = ids["input_ids"].copy()   # simple causal LM; mask the prompt if you prefer
    return ids

rows = [fmt(json.loads(l)) for l in open("train.jsonl")]
Trainer(model, TrainingArguments("out", per_device_train_batch_size=4, num_train_epochs=3,
        learning_rate=2e-4, bf16=True, logging_steps=5), train_dataset=rows,
        data_collator=lambda b: tok.pad(b, return_tensors="pt")).train()
model = model.merge_and_unload()              # fold LoRA back into the base weights
model.save_pretrained("functiongemma-mine"); tok.save_pretrained("functiongemma-mine")
```

### 3. Convert merged weights → GGUF (the format `compile-gemma.sh` eats)

```bash
# llama.cpp
python convert_hf_to_gguf.py functiongemma-mine --outfile functiongemma-mine-f16.gguf --outtype f16
./llama-quantize functiongemma-mine-f16.gguf functiongemma-mine-Q5_K_M.gguf Q5_K_M
```

### 4. Re-bake into the demo (unchanged self-compile path)

```bash
# point demo.env (or inline) at your new GGUF, then the exact same board build:
GEMMA_GGUF=$PWD/functiongemma-mine-Q5_K_M.gguf scripts/compile-gemma.sh board
BOARD=root@<ip> ./gradlew deployBoard
```

That's the whole finetuning story for Case A. For **Case B**, also do step 5.

### 5. (Case B only) Wire the new tool into codec + router

If you taught the model a spare token, say `<tool_6>` → `open_door`:

- **Codec** — add the mapping. `CompactCodec.TOKEN_TO_NAME` is a private map in the *upstream* `gemma-iree`
  module, so today you fork/patch it (or, cleaner, send an upstream tweak making the map injectable — noted
  in `BOARD-RUNBOOK.md`). Add: `"6" to "open_door"`.
- **Router** — register a handler (`src/commonMain/.../actions/Actions.kt`):
  ```kotlin
  fun defaultRouter() = ActionRouter().apply {
      registerAll(LoggingActions().handlers())
      register("open_door") { i -> ActionResult("open_door", true, "opening ${i.args["which"] ?: "front"}") }
  }
  ```
- Rebuild + redeploy (`./gradlew deployBoard`).

See `examples/custom-command/` for a minimal, forkable worked example.

### 6. Verify

```bash
# on the board (or host jvmRun), drive the flow and check the tool call:
VOICECC_PROFILE=1 voicecc gen "blackout the room"     # -> <tool_0>(state="off")<end> -> set_lights state=off
# regression: the base commands must still work
voicecc gen "turn the light on"                        # -> [262146,236769,3255,718,498,1373,262152,106]
```
`ActionRouterTest.kt` is the place to add a unit assertion for your new intent → action.

---

## Advanced: a new tool with NO spare functional token

If the vocab has no free `<tool_N>`, you must extend the tokenizer before step 2:
```python
tok.add_special_tokens({"additional_special_tokens": ["<tool_6>"]})
model.resize_token_embeddings(len(tok))   # new row is random — the LoRA data teaches it
```
Then train (step 2), convert (step 3, the new token rides in the GGUF vocab), and wire codec+router
(step 5). Caveat: a resized embedding changes the model's vocab dim, so re-confirm the DSL export shapes
and the token oracle after baking — treat the first board run as a bring-up (see `BOARD-RUNBOOK.md`).
