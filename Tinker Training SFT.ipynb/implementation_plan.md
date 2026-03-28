# GPT OSS 120B No-Tool SFT Training Notebook — Implementation Plan

A Kaggle notebook for SFT fine-tuning `openai/gpt-oss-120b` via Tinker using the Nemotron-Cascade no-tool math dataset with high-reasoning `<think>` traces.

---

## Context & Key Decisions

> [!IMPORTANT]  
> **Model**: `openai/gpt-oss-120b` — available on Tinker's model lineup. This is a large MoE model, cost-effective because cost is proportional to active parameters.

> [!IMPORTANT]
> **Renderer**: We use `GptOssRenderer` from `tinker_cookbook.renderers.gpt_oss` (same approach as previous implementation plan). GPT OSS uses the Harmony token format (`<|start|>/<|end|>/<|channel|>/<|message|>/<|return|>`), NOT the `<|im_start|>/<|im_end|>` format of Nemotron. The reference notebook's manual `build_datum_manual` approach does NOT apply here.

> [!IMPORTANT]
> **Dataset**: No-tool math rows from `nemotron-cascade-math-tool-no-tool-10k-rows`. The data has DeepSeek-style `<think>...</think>` reasoning blocks that must be converted to Harmony's `ThinkingPart`/`TextPart` format. The user loads via `pd.read_json(..., lines=True)` from a Kaggle dataset path.

### What the Dataset Looks Like

From the CSV example, each row has columns: `domain`, `source`, `messages`, `generator`. The `messages` column is a list of dicts:
```
[
  {"role": "system", "content": "..."},       # system prompt (tool definition — NOT relevant for no-tool)
  {"role": "user", "content": "Solve ..."},
  {"role": "assistant", "content": "<think>...</think>\n\\boxed{...}"}
]
```

For the **no-tool** variant:
- We **strip the system message** (it contains tool definitions we don't need)
- We keep only user → assistant turns
- The assistant content has `<think>...</think>` followed by the final answer

---

## Proposed Notebook Structure

**File**: `d:\ALL CODES\AIMO Progress Prize 3\Tinker Training SFT.ipynb\No tool.ipynb`

### Cell 1 — Config (already exists, will update)

```python
class Config:
    model_name = "openai/gpt-oss-120b"
    lora_rank = 32
    alpha = 64
    lr = 2e-4
    epoch = 1
    batch_size = 32          # Smaller than ref (64) — 120B model needs more memory per sample
    max_length = 100_000     # User specified — very long ctx for deep reasoning
    warmup_ratio = 0.05      # 5% warmup
    lr_schedule = 'cosine'   # Cosine decay
    eval_split = 5           # Hold out 5 examples for eval
    save_every = 10          # Checkpoint every 10 steps
    adam_beta1 = 0.9
    adam_beta2 = 0.95
    adam_eps = 1e-8
```

### Cell 2 — Installs
```python
!pip install -q tinker tinker-cookbook transformers safetensors requests
```

### Cell 3 — Imports & API Key
```python
import os, json, time, random, logging, re, ast
import pandas as pd
import tinker
from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook import tokenizer_utils

os.environ["TINKER_API_KEY"] = "YOUR_API_KEY_HERE"   # <-- replace this!

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gpt-oss-notool-sft")

print("Tinker SDK version:", tinker.__version__)
print("API key set:", "TINKER_API_KEY" in os.environ and os.environ["TINKER_API_KEY"] != "YOUR_API_KEY_HERE")
```

### Cell 4 — Load Dataset (user's existing cell — refined)

```python
# Load the JSONL dataset from Kaggle input
data = pd.read_json(
    "/kaggle/input/datasets/nahidhossainredom/nemotron-cascade-math-tool-no-tool-10k-rows/math_notool_10k.jsonl",
    lines=True
)

# Filter: keep only rows where user prompt contains \boxed (competition-style)
user_contents = data['messages'].apply(lambda x: x[1]['content'])
mask = user_contents.str.contains(r"\boxed", case=False, regex=False)
training_data = data[mask].copy()

print(f"Total rows: {len(data)}, After \\boxed filter: {len(training_data)}")
print(f"Sample row keys: {list(training_data.iloc[0].keys())}")
```

### Cell 5 — Inspect Raw Data
```python
# Look at a sample to understand format
sample = training_data.iloc[0]
msgs = sample['messages']
# Handle messages stored as string (from CSV/JSONL)
if isinstance(msgs, str):
    msgs = ast.literal_eval(msgs)

print(f"Number of messages: {len(msgs)}")
for i, msg in enumerate(msgs):
    content_preview = str(msg.get("content", ""))[:200]
    print(f"  [{i}] role={msg['role']}: {content_preview}...")
```

### Cell 6 — DeepSeek → Harmony Formatter (No-Tool)

Core conversion function. Strips system messages (tool defs), extracts `<think>` blocks, converts to Harmony `ThinkingPart`/`TextPart`.

```python
def format_notool_to_harmony(messages: list[dict]) -> list[dict]:
    """
    Convert DeepSeek no-tool conversation to GPT OSS Harmony format.
    
    - Strips system messages (contain tool definitions we don't need)
    - Converts <think>...</think> blocks → {"type": "thinking", "thinking": "..."}
    - Text after </think> → {"type": "text", "text": "..."}
    
    The GptOssRenderer then maps:
      thinking → <|start|>assistant<|channel|>analysis<|message|>...<|end|>
      text     → <|start|>assistant<|channel|>final<|message|>...<|return|>
    """
    formatted = []
    for msg in messages:
        # Skip system messages (tool definitions not needed for no-tool)
        if msg["role"] == "system":
            continue
        
        if msg["role"] == "assistant":
            content = msg["content"]
            think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            
            parts = []
            if think_match:
                thinking_text = think_match.group(1).strip()
                if thinking_text:
                    parts.append({"type": "thinking", "thinking": thinking_text})
                after_think = content[think_match.end():].strip()
                if after_think:
                    parts.append({"type": "text", "text": after_think})
            else:
                parts.append({"type": "text", "text": content.strip()})
            
            formatted.append({"role": "assistant", "content": parts})
        else:
            # User messages pass through
            formatted.append(msg)
    
    return formatted
```

### Cell 7 — Initialize Tokenizer & Renderer

```python
tokenizer = tokenizer_utils.get_tokenizer(Config.model_name)
print(f"Tokenizer loaded! Vocab size: {tokenizer.vocab_size}")

renderer = GptOssRenderer(
    tokenizer=tokenizer,
    use_system_prompt=True,
    reasoning_effort="high",       # High reasoning for math
    current_date="2026-03-29",     # Fixed for reproducibility
)

# Quick test
sample_msgs = training_data.iloc[0]['messages']
if isinstance(sample_msgs, str):
    sample_msgs = ast.literal_eval(sample_msgs)
sample_formatted = format_notool_to_harmony(sample_msgs)

model_input, weights = renderer.build_supervised_example(sample_formatted)
print(f"Token count: {model_input.length}")
print(f"Trainable tokens (weight>0): {int(weights.sum().item())}")
```

### Cell 8 — Convert All Rows to Datums

```python
def row_to_datum(row) -> tinker.Datum:
    """Convert one dataset row into a Tinker Datum."""
    msgs = row['messages']
    if isinstance(msgs, str):
        msgs = ast.literal_eval(msgs)
    harmony_msgs = format_notool_to_harmony(msgs)
    model_input, weights = renderer.build_supervised_example(harmony_msgs)
    return datum_from_model_input_weights(model_input, weights, max_length=Config.max_length)

print(f"Converting {len(training_data)} rows to datums...")
all_datums = []
skipped = 0
for i, (_, row) in enumerate(training_data.iterrows()):
    try:
        datum = row_to_datum(row)
        all_datums.append(datum)
    except Exception as e:
        skipped += 1
        if skipped <= 5:
            print(f"  Skipped row {i}: {e}")
    if (i + 1) % 1000 == 0:
        print(f"  Processed {i+1}/{len(training_data)}...")

print(f"\nConverted {len(all_datums)} datums ({skipped} skipped)")

# Token stats
token_counts = [d.model_input.length for d in all_datums]
print(f"Token stats: mean={sum(token_counts)/len(token_counts):.0f}, "
      f"max={max(token_counts)}, min={min(token_counts)}")
```

### Cell 9 — Sanity Checks

```python
print("=== Sanity Check: Decoded tokens for 3 random samples ===\n")
for idx in random.sample(range(len(all_datums)), min(3, len(all_datums))):
    datum = all_datums[idx]
    tokens = []
    for chunk in datum.model_input.chunks:
        tokens.extend(chunk.tokens)
    decoded = tokenizer.decode(tokens)
    
    has_start = "<|start|>" in decoded
    has_channel = "<|channel|>" in decoded
    has_message = "<|message|>" in decoded
    has_end = "<|end|>" in decoded or "<|return|>" in decoded
    
    print(f"Sample {idx}: {len(tokens)} tokens")
    print(f"  Has <|start|>: {has_start}, <|channel|>: {has_channel}, "
          f"<|message|>: {has_message}, <|end|>/<|return|>: {has_end}")
    
    w = datum.loss_fn_inputs["weights"].data
    n_train = sum(1 for x in w if x > 0)
    print(f"  Trainable tokens: {n_train}/{len(w)}")
    print(f"  First 300 chars: {decoded[:300]}...")
    print()

print("✓ Sanity checks complete!")
```

### Cell 10 — Training Loop

Full LoRA SFT training loop (adapted from reference notebook for GPT OSS 120B).

```python
def compute_nll(fwd_bwd_result, batch):
    """Compute mean negative log-likelihood."""
    total_nll = 0.0
    total_weight = 0.0
    for output, datum in zip(fwd_bwd_result.loss_fn_outputs, batch):
        logprobs = output["logprobs"].data
        w = datum.loss_fn_inputs["weights"].data
        for lp, wi in zip(logprobs, w):
            if wi > 0:
                total_nll -= lp
                total_weight += wi
    return total_nll / max(total_weight, 1.0)


import math

def train_lora(all_datums):
    """Full LoRA SFT training loop on Tinker for GPT OSS 120B."""
    cfg = Config
    
    # Train/eval split
    random.seed(42)
    indices = list(range(len(all_datums)))
    random.shuffle(indices)
    eval_size = min(cfg.eval_split, len(all_datums) // 10)
    eval_datums = [all_datums[i] for i in indices[:eval_size]]
    train_datums = [all_datums[i] for i in indices[eval_size:]]
    
    n_batches_per_epoch = len(train_datums) // cfg.batch_size
    total_steps = n_batches_per_epoch * cfg.epoch
    warmup_steps = max(1, int(cfg.warmup_ratio * total_steps))
    
    print(f"Train: {len(train_datums)} | Eval: {len(eval_datums)}")
    print(f"Steps/epoch: {n_batches_per_epoch} | Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    
    # Connect to Tinker
    print("\nConnecting to Tinker API...")
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=cfg.model_name, rank=cfg.lora_rank,
    )
    print("Connected!")
    
    # Training loop
    global_step = 0
    best_eval_nll = float("inf")
    train_losses = []
    t_start = time.time()
    
    for epoch in range(cfg.epoch):
        epoch_indices = list(range(len(train_datums)))
        random.seed(epoch)
        random.shuffle(epoch_indices)
        
        for batch_idx in range(n_batches_per_epoch):
            step_start = time.time()
            
            # LR schedule: cosine with warmup
            if global_step < warmup_steps:
                lr_mult = global_step / warmup_steps
            else:
                progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
                lr_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
            current_lr = cfg.lr * lr_mult
            
            adam_params = tinker.AdamParams(
                learning_rate=current_lr,
                beta1=cfg.adam_beta1, beta2=cfg.adam_beta2, eps=cfg.adam_eps,
            )
            
            # Get batch
            start_idx = batch_idx * cfg.batch_size
            batch = [train_datums[epoch_indices[i]]
                     for i in range(start_idx, start_idx + cfg.batch_size)]
            
            # Forward + backward + optimizer
            fwd_bwd_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
            optim_future = training_client.optim_step(adam_params)
            fwd_bwd_result = fwd_bwd_future.result()
            optim_result = optim_future.result()
            
            train_nll = compute_nll(fwd_bwd_result, batch)
            train_losses.append(train_nll)
            step_time = time.time() - step_start
            
            if global_step % 5 == 0 or global_step == total_steps - 1:
                elapsed = time.time() - t_start
                avg_loss = sum(train_losses[-10:]) / len(train_losses[-10:])
                print(f"[Step {global_step:4d}/{total_steps}] "
                      f"epoch={epoch+1}/{cfg.epoch} lr={current_lr:.2e} "
                      f"train_nll={train_nll:.4f} avg_nll(10)={avg_loss:.4f} "
                      f"step={step_time:.1f}s elapsed={elapsed:.0f}s")
            
            # Evaluate periodically
            if eval_datums and (global_step % cfg.save_every == 0 or global_step == total_steps - 1):
                eval_future = training_client.forward_backward(
                    eval_datums, loss_fn="cross_entropy")
                eval_result = eval_future.result()
                eval_nll = compute_nll(eval_result, eval_datums)
                is_best = eval_nll < best_eval_nll
                if is_best:
                    best_eval_nll = eval_nll
                print(f"  >>> EVAL nll={eval_nll:.4f} "
                      f"{'NEW BEST!' if is_best else ''} (best={best_eval_nll:.4f})")
                # Zero out gradient from eval pass
                training_client.optim_step(
                    tinker.AdamParams(learning_rate=0.0, beta1=0.9, beta2=0.95, eps=1e-8)
                ).result()
            
            # Save checkpoint
            if cfg.save_every > 0 and global_step % cfg.save_every == 0 and global_step > 0:
                name = f"step_{global_step:04d}"
                training_client.save_state(name=name).result()
                print(f"  Saved checkpoint: {name}")
            
            global_step += 1
    
    # Save final
    print("\nSaving final checkpoint...")
    training_client.save_state(name="final").result()
    sampler_result = training_client.save_weights_for_sampler(
        name="gpt_oss_120b_notool_final").result()
    sampler_path = sampler_result.path
    
    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE!")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Final train NLL: {train_losses[-1]:.4f}")
    print(f"  Best eval NLL: {best_eval_nll:.4f}")
    print(f"  Sampler path: {sampler_path}")
    print(f"{'='*60}")
    
    return service_client, training_client, sampler_path, train_losses


# ============================================================
#  TO ACTUALLY TRAIN, UNCOMMENT:
# ============================================================
# service_client, training_client, sampler_path, losses = train_lora(all_datums)
print("Training function defined! Uncomment above to run.")
```

### Cell 11 — Evaluation (Sampling)

```python
import re as _re

def extract_boxed_balanced(text):
    """Extract content from the last \\boxed{...} in text, handling nested braces."""
    key = r"\boxed{"
    idx = text.rfind(key)
    if idx < 0:
        return None
    i = idx + len(key)
    depth = 1
    while i < len(text) and depth:
        if text[i] == "{": depth += 1
        elif text[i] == "}": depth -= 1
        i += 1
    return text[idx + len(key):i - 1].strip() if depth == 0 else None

def evaluate_model(service_client, sampler_path, test_problems, max_tokens=8192):
    from tinker.types import SamplingParams
    
    sampling_client = service_client.create_sampling_client(model_path=sampler_path)
    stop_sequences = renderer.get_stop_sequences()
    params = SamplingParams(max_tokens=max_tokens, temperature=0.6, stop=stop_sequences)
    
    correct = 0
    for i, problem in enumerate(test_problems):
        messages = [{"role": "user", "content": problem["question"]}]
        prompt = renderer.build_generation_prompt(messages)
        output = sampling_client.sample(prompt, sampling_params=params, num_samples=1).result()
        
        response_msg, success = renderer.parse_response(output.sequences[0].tokens)
        
        if isinstance(response_msg["content"], list):
            answer_text = "".join(
                p.get("text", "") for p in response_msg["content"] if p["type"] == "text"
            )
        else:
            answer_text = response_msg["content"]
        
        predicted = extract_boxed_balanced(answer_text)
        gold = str(problem["answer"]).strip()
        is_correct = predicted is not None and (
            predicted.strip() == gold or
            predicted.replace(" ", "") == gold.replace(" ", "")
        )
        if is_correct:
            correct += 1
        if i < 5:
            print(f"Q: {problem['question'][:80]}...")
            print(f"  Predicted: {predicted}, Gold: {gold}, Correct: {is_correct}")
    
    print(f"\nAccuracy: {correct}/{len(test_problems)} = {correct/len(test_problems)*100:.1f}%")

# evaluate_model(service_client, sampler_path, your_test_problems)
```

### Cell 12 — Download Weights

```python
def download_weights(service_client, sampler_path, output_dir="gpt_oss_120b_notool_weights"):
    os.makedirs(output_dir, exist_ok=True)
    rest_client = service_client.create_rest_client()
    url_resp = rest_client.get_checkpoint_archive_url_from_tinker_path(sampler_path).result()
    
    import requests
    print(f"Downloading checkpoint from: {sampler_path}")
    r = requests.get(url_resp.url, stream=True)
    r.raise_for_status()
    total_bytes = int(r.headers.get("content-length", 0))
    print(f"  File size: {total_bytes / 1e9:.2f} GB")
    
    output_file = os.path.join(output_dir, "lora_checkpoint.tar")
    downloaded = 0
    with open(output_file, "wb") as f:
        for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total_bytes:
                pct = 100 * downloaded / total_bytes
                print(f"\r  Progress: {pct:.1f}% ({downloaded/1e9:.2f}/{total_bytes/1e9:.2f} GB)", end="")
    
    print(f"\n  Saved: {output_file} ({downloaded / 1e9:.2f} GB)")

# download_weights(service_client, sampler_path)
```

---

## Hyperparameter Rationale

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `model_name` | `openai/gpt-oss-120b` | User specified — large MoE model, cost-effective via active params |
| `lora_rank` | 32 | Same as reference — good balance of capacity vs. efficiency |
| `alpha` | 64 | 2× rank — standard LoRA alpha scaling |
| `lr` | 2e-4 | Sweet spot for LoRA SFT per reference & Tinker docs |
| `epoch` | 1 | User specified — single pass to avoid overfitting on 10k rows |
| `batch_size` | 32 | Reduced from 64 (reference) — 120B model may need smaller batches |
| `max_length` | 100,000 | User specified — very long context for deep math reasoning |
| `warmup_ratio` | 0.05 | 5% linear warmup — prevents early instability |
| `lr_schedule` | cosine | User specified — smoother convergence than linear decay |
| `adam_betas` | (0.9, 0.95) | Same as reference — slightly less momentum on second moment |

---

## Open Questions

> [!WARNING]
> **batch_size = 32 vs 64**: The reference used batch_size=64 for Nemotron-3-Nano-30B. For the much larger GPT OSS 120B, you may want to keep 32 or even go lower if Tinker reports memory issues. Tinker handles distributed training, but larger batches with very long sequences (100K tokens) could still be an issue. Do you have a preference?

> [!WARNING]
> **max_length = 100,000**: This is 12× longer than the reference notebook's 8192. The dataset traces in the CSV sample appear to be ~2K-10K tokens long. Are you sure you want 100K, or would a more conservative 32,768 or 65,536 be acceptable? Very long sequences slow training significantly.

> [!WARNING]
> **alpha = 64**: The reference notebook didn't specify alpha (used default). Your Config has alpha=64 but the `create_lora_training_client()` API may not support an `alpha` parameter directly — need to verify if Tinker accepts it or if it's handled differently.

---

## Verification Plan

### Automated (Cell 9)
1. Decode tokens → verify Harmony special tokens present (`<|start|>`, `<|channel|>`, etc.)
2. Check weight distribution (prompt=0, completion=1)  
3. Verify 3 random samples have correct format

### During Training
1. Monitor NLL decrease across steps
2. Check eval NLL improves periodically

### Post-Training
1. Run evaluation on held-out problems
2. Verify `\boxed{}` extraction works correctly
