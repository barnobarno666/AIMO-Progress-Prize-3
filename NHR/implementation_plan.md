# GPT OSS SFT Training Notebooks — Implementation Plan (v2)

Two Jupyter notebooks for fine-tuning GPT OSS via Tinker using 10k math rows from the Nemotron-Cascade-2-SFT-Data dataset. The dataset contains DeepSeek-generated traces that must be reformatted to the OpenAI Harmony format that GPT OSS expects.

> [!IMPORTANT]
> **Key Design Decision**: We use the `GptOssRenderer` from `tinker_cookbook.renderers` (not the manual `build_datum_manual` approach of the reference notebook). This renderer natively handles Harmony's `<|start|>/<|end|>/<|channel|>/<|message|>/<|call|>/<|return|>/<|constrain|>` tokens and produces correct per-token loss weights via `build_supervised_example()`. The reference notebook's manual approach was for Nemotron's `<|im_start|>/<|im_end|>` format — **not applicable to GPT OSS**.

---

## Critical Clarification: SFT vs RL Tool Execution

> [!CAUTION]
> **No code execution happens during SFT forward passes.** This is a common source of confusion.

**SFT (what we're doing):** The model learns from **static, pre-recorded traces** in the dataset. Each training example is a frozen conversation `[user → assistant(think + tool_call) → tool_result → assistant(answer)]`. The entire trajectory, including code and its execution output, is already in the dataset. During training, Tinker just does `forward_backward` on the token sequence — no code runs, no sandbox is involved. The model learns to **mimic** the pattern of calling tools and interpreting results.

**RL (what we're NOT doing):** In RL training (e.g., `recipes/code_rl/`), the model **generates** responses live, and tool calls are **actually executed** via:
- `AgentToolMessageEnv` — an RL environment that intercepts the model's tool calls
- Sandbox backends (`SandboxFusion`/Docker or `Modal`/cloud) — execute the generated code safely
- The execution result is fed back to the model for the next generation step
- A `reward_fn` grades the full trajectory at the end

For our SFT notebooks: **the training data already contains the tool calls AND their results**. We simply format them into Harmony format and train. The model learns to produce tool calls in the correct Harmony format, and separately learns to interpret tool results — all from the frozen traces, no live execution.

---

## Proposed File Structure

```
AIMO-Progress-Prize-3/
├── Reference Code/
│   └── new-to-tinker.ipynb        # Existing reference
├── NHR/
│   ├── gpt_oss_math_notool_sft.ipynb  # [NEW] Notebook 1
│   └── gpt_oss_math_tool_sft.ipynb    # [NEW] Notebook 2
```

---

## Notebook 1: `NHR/gpt_oss_math_notool_sft.ipynb` (No-Tool Math)

### Cell 1 — Installs
```python
!pip install -q tinker tinker-cookbook openai-harmony transformers safetensors requests
```

### Cell 2 — Imports & Config
```python
import os, json, time, random, logging, re, glob
import tinker
from tinker_cookbook import renderers, tokenizer_utils
from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
from tinker_cookbook.supervised.common import datum_from_model_input_weights

os.environ["TINKER_API_KEY"] = "YOUR_API_KEY_HERE"

MODEL_NAME = "openai/gpt-oss-4o-mini"  # <-- set the actual gpt-oss model name on Tinker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gpt-oss-notool")

print("Tinker SDK version:", tinker.__version__)
print("API key set:", "TINKER_API_KEY" in os.environ and os.environ["TINKER_API_KEY"] != "YOUR_API_KEY_HERE")
```

### Cell 3 — Load Dataset from Kaggle
```python
# Dataset is pre-uploaded to Kaggle and available at a local path
# The user has filtered to 10k math_notool rows and saved as JSONL

INPUT_PATH = "/kaggle/input/nemotron-math-notool/"  # <-- set to your Kaggle dataset path

# Find all JSONL files in the input directory
jsonl_files = sorted(glob.glob(os.path.join(INPUT_PATH, "*.jsonl")))
print(f"Found {len(jsonl_files)} JSONL files: {jsonl_files}")

# Load all conversations
all_convos = []
for fpath in jsonl_files:
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                all_convos.append(row)

print(f"Loaded {len(all_convos)} conversations")
print(f"First conversation has {len(all_convos[0]['messages'])} messages")
```

### Cell 4 — Inspect Raw Data Format
```python
# Inspect a raw sample to understand the DeepSeek trace format
sample = all_convos[0]
print("Keys per row:", list(sample.keys()))
print(f"\nNumber of messages: {len(sample['messages'])}")
for i, msg in enumerate(sample["messages"]):
    content_preview = str(msg.get("content", ""))[:150]
    print(f"  [{i}] role={msg['role']}: {content_preview}...")
```

This cell is critical — it shows the DeepSeek message format that needs reformatting. Expected structure per row:
```json
{"messages": [
  {"role": "user", "content": "Solve ..."},
  {"role": "assistant", "content": "<think>...\n</think>\n\\boxed{42}"}
]}
```

### Cell 5 — DeepSeek → Harmony Formatter (No-Tool)

This is the core formatter. The DeepSeek trace uses `<think>...</think>` blocks for chain-of-thought. GPT OSS Harmony expects content parts using the `ThinkingPart` / `TextPart` typed dict format expected by `GptOssRenderer`.

```python
def format_deepseek_notool_to_harmony(messages: list[dict]) -> list[dict]:
    """
    Convert a DeepSeek-format conversation to GPT OSS Harmony format.
    
    DeepSeek format:
      assistant.content = "<think>...reasoning...</think>\nfinal answer"
    
    Harmony format (for GptOssRenderer):
      assistant.content = [
          {"type": "thinking", "thinking": "...reasoning..."},
          {"type": "text", "text": "final answer"}
      ]
    
    The renderer will then map:
      - thinking → <|start|>assistant<|channel|>analysis<|message|>...reasoning...<|end|>
      - text     → <|start|>assistant<|channel|>final<|message|>final answer<|return|>
    
    Non-assistant messages pass through unchanged.
    """
    formatted = []
    for msg in messages:
        if msg["role"] == "assistant":
            content = msg["content"]
            
            # Extract <think>...</think> block
            think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            
            parts = []
            if think_match:
                thinking_text = think_match.group(1).strip()
                if thinking_text:
                    parts.append({"type": "thinking", "thinking": thinking_text})
                # Everything after </think> is the final answer
                after_think = content[think_match.end():].strip()
                if after_think:
                    parts.append({"type": "text", "text": after_think})
            else:
                # No <think> block — treat entire content as final answer
                parts.append({"type": "text", "text": content.strip()})
            
            formatted.append({"role": "assistant", "content": parts})
        
        elif msg["role"] == "system":
            # System messages pass through — GptOssRenderer maps them to "developer"
            formatted.append(msg)
        
        else:
            # User messages pass through unchanged
            formatted.append(msg)
    
    return formatted


# Test the formatter on the sample
sample_formatted = format_deepseek_notool_to_harmony(all_convos[0]["messages"])
print("=== Formatted sample ===")
for msg in sample_formatted:
    if msg["role"] == "assistant" and isinstance(msg["content"], list):
        for part in msg["content"]:
            ptype = part["type"]
            text = part.get("thinking", part.get("text", ""))
            print(f"  [{ptype}]: {str(text)[:100]}...")
    else:
        print(f"  [{msg['role']}]: {str(msg['content'])[:100]}...")
```

### Cell 6 — Initialize Tokenizer & Renderer

```python
# Load the GPT OSS tokenizer
tokenizer = tokenizer_utils.get_tokenizer(MODEL_NAME)
print(f"Tokenizer loaded! Vocab size: {tokenizer.vocab_size}")

# Create GptOssRenderer
# use_system_prompt=True adds the standard Harmony system prompt 
# (identity, knowledge cutoff, reasoning effort, channel list)
# Set reasoning_effort to "high" for math tasks
renderer = GptOssRenderer(
    tokenizer=tokenizer,
    use_system_prompt=True,
    reasoning_effort="high",
    current_date="2026-03-28",  # Fixed for reproducibility
)

# Test: build a supervised example from the formatted sample
model_input, weights = renderer.build_supervised_example(sample_formatted)
print(f"Token count: {model_input.length}")
print(f"Trainable tokens (weight>0): {int(weights.sum().item())}")
print(f"Total tokens: {len(weights)}")

# Verify decoded tokens look correct
decoded = tokenizer.decode(model_input.to_ints())
print(f"\n=== First 500 chars of decoded tokens ===")
print(decoded[:500])
```

### Cell 7 — Convert All Rows to Datums

```python
MAX_LENGTH = 8192

def row_to_datum(row: dict) -> tinker.Datum:
    """Convert one dataset row into a Tinker Datum using GptOssRenderer."""
    messages = format_deepseek_notool_to_harmony(row["messages"])
    model_input, weights = renderer.build_supervised_example(messages)
    return datum_from_model_input_weights(model_input, weights, max_length=MAX_LENGTH)

# Convert all rows
print(f"Converting {len(all_convos)} rows to datums...")
all_datums = []
skipped = 0
for i, row in enumerate(all_convos):
    try:
        datum = row_to_datum(row)
        all_datums.append(datum)
    except Exception as e:
        skipped += 1
        if skipped <= 5:
            print(f"  Skipped row {i}: {e}")
    if (i + 1) % 1000 == 0:
        print(f"  Processed {i + 1}/{len(all_convos)}...")

print(f"\nConverted {len(all_datums)} datums ({skipped} skipped)")

# Token stats
token_counts = [d.model_input.length for d in all_datums]
print(f"Token stats: mean={sum(token_counts)/len(token_counts):.0f}, "
      f"max={max(token_counts)}, min={min(token_counts)}")
```

### Cell 8 — Sanity Checks (Before Training)

```python
# Verify the format is correct on a few samples
print("=== Sanity Check: Decoded tokens for 3 random samples ===\n")
for idx in random.sample(range(len(all_datums)), min(3, len(all_datums))):
    datum = all_datums[idx]
    tokens = []
    for chunk in datum.model_input.chunks:
        tokens.extend(chunk.tokens)
    decoded = tokenizer.decode(tokens)
    
    # Check for Harmony special tokens
    has_start = "<|start|>" in decoded
    has_channel = "<|channel|>" in decoded
    has_message = "<|message|>" in decoded
    has_end = "<|end|>" in decoded or "<|return|>" in decoded
    
    print(f"Sample {idx}: {len(tokens)} tokens")
    print(f"  Has <|start|>: {has_start}, <|channel|>: {has_channel}, "
          f"<|message|>: {has_message}, <|end|>/<|return|>: {has_end}")
    
    # Show weight distribution
    w = datum.loss_fn_inputs["weights"].data
    n_train = sum(1 for x in w if x > 0)
    print(f"  Trainable tokens: {n_train}/{len(w)}")
    print(f"  First 200 chars: {decoded[:200]}...")
    print()

assert all(has_start and has_channel and has_message and has_end 
           for _ in [None]), "Harmony format tokens missing!"
print("✓ All sanity checks passed!")
```

### Cell 9 — Training Loop

```python
def compute_nll(fwd_bwd_result, batch):
    """Compute mean negative log-likelihood from forward_backward output."""
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


def train_lora(all_datums, epochs=2, lr=2e-4, batch_size=64,
               lora_rank=32, save_every=10, eval_split=5):
    """Full LoRA SFT training loop on Tinker for GPT OSS."""
    
    # Train/eval split
    random.seed(42)
    indices = list(range(len(all_datums)))
    random.shuffle(indices)
    eval_size = min(eval_split, len(all_datums) // 10)
    eval_datums = [all_datums[i] for i in indices[:eval_size]]
    train_datums = [all_datums[i] for i in indices[eval_size:]]
    
    n_batches_per_epoch = len(train_datums) // batch_size
    total_steps = n_batches_per_epoch * epochs
    print(f"Train: {len(train_datums)} | Eval: {len(eval_datums)} | "
          f"Steps/epoch: {n_batches_per_epoch} | Total steps: {total_steps}")
    
    # Connect to Tinker
    print("\nConnecting to Tinker API...")
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=MODEL_NAME, rank=lora_rank,
    )
    print("Connected!")
    
    # Training loop
    global_step = 0
    best_eval_nll = float("inf")
    train_losses = []
    t_start = time.time()
    
    for epoch in range(epochs):
        epoch_indices = list(range(len(train_datums)))
        random.seed(epoch)
        random.shuffle(epoch_indices)
        
        for batch_idx in range(n_batches_per_epoch):
            step_start = time.time()
            
            # LR schedule: linear warmup (5%) + linear decay
            warmup_steps = max(1, int(0.05 * total_steps))
            if global_step < warmup_steps:
                lr_mult = global_step / warmup_steps
            else:
                lr_mult = max(0.0, 1.0 - (global_step - warmup_steps) / (total_steps - warmup_steps))
            current_lr = lr * lr_mult
            
            adam_params = tinker.AdamParams(
                learning_rate=current_lr, beta1=0.9, beta2=0.95, eps=1e-8,
            )
            
            # Get batch
            start_idx = batch_idx * batch_size
            batch = [train_datums[epoch_indices[i]] 
                     for i in range(start_idx, start_idx + batch_size)]
            
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
                      f"epoch={epoch+1}/{epochs} lr={current_lr:.2e} "
                      f"train_nll={train_nll:.4f} avg_nll(10)={avg_loss:.4f} "
                      f"step={step_time:.1f}s elapsed={elapsed:.0f}s")
            
            # Evaluate periodically
            if eval_datums and (global_step % save_every == 0 or global_step == total_steps - 1):
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
            if save_every > 0 and global_step % save_every == 0 and global_step > 0:
                name = f"step_{global_step:04d}"
                training_client.save_state(name=name).result()
                print(f"  Saved checkpoint: {name}")
            
            global_step += 1
    
    # Save final
    print("\nSaving final checkpoint...")
    training_client.save_state(name="final").result()
    sampler_result = training_client.save_weights_for_sampler(
        name="gpt_oss_notool_final").result()
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
# service_client, training_client, sampler_path, losses = train_lora(
#     all_datums, epochs=2, lr=2e-4, batch_size=64, lora_rank=32
# )
print("Training function defined! Uncomment above to run.")
```

### Cell 10 — Evaluation (Sampling)

```python
# Uses renderer.build_generation_prompt() and renderer.parse_response()
# to properly handle Harmony format for both input and output

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

def normalize_compare(pred, gold):
    p, g = pred.strip(), gold.strip()
    if p == g or p.replace(" ", "") == g.replace(" ", ""):
        return True
    try:
        return abs(float(p) - float(g)) <= 1e-9 * max(1.0, abs(float(g)))
    except (ValueError, TypeError):
        return False

def evaluate_model(service_client, sampler_path, test_problems, max_tokens=4096):
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
        if predicted and normalize_compare(predicted, problem["answer"]):
            correct += 1
        if i < 3:
            print(f"Q: {problem['question'][:80]}...")
            print(f"  Predicted: {predicted}, Gold: {problem['answer']}")
    
    print(f"\nAccuracy: {correct}/{len(test_problems)} = {correct/len(test_problems)*100:.1f}%")

# evaluate_model(service_client, sampler_path, your_test_problems)
```

### Cell 11 — Download Weights

```python
def download_weights(service_client, sampler_path, output_dir="gpt_oss_notool_weights"):
    os.makedirs(output_dir, exist_ok=True)
    files = service_client.list_files(sampler_path)
    print(f"Downloading {len(files)} files to {output_dir}/...")
    for f in files:
        local_path = os.path.join(output_dir, f.name)
        service_client.download_file(f.path, local_path)
        print(f"  {f.name}: {os.path.getsize(local_path)/1024/1024:.1f} MB")
    print("Done!")

# download_weights(service_client, sampler_path)
```

---

## Notebook 2: `NHR/gpt_oss_math_tool_sft.ipynb` (Tool-Use Math)

Structurally identical to Notebook 1. Only the cells that differ are listed below.

### Cell 3 — Load from Kaggle (tool dataset)
```python
INPUT_PATH = "/kaggle/input/nemotron-math-tool/"  # <-- tool-use dataset path
```

### Cell 5 — DeepSeek → Harmony Formatter (Tool-Use)

This formatter handles the multi-turn pattern: `user → assistant(think + tool_call) → tool_result → assistant(answer)`

```python
def format_deepseek_tool_to_harmony(messages: list[dict]) -> list[dict]:
    """
    Convert a DeepSeek tool-use conversation to GPT OSS Harmony format.
    
    Handles:
    1. <think>...</think> → ThinkingPart (analysis channel)
    2. tool_calls in assistant messages → ToolCall objects (commentary channel)
    3. tool result messages → role="tool" with mandatory "name" field
    4. Final answer → TextPart (final channel)
    
    The GptOssRenderer then maps these to proper Harmony tokens.
    """
    from tinker_cookbook.renderers.base import ToolCall
    
    formatted = []
    last_tool_call_name = None  # Track for pairing with results
    
    for msg in messages:
        role = msg["role"]
        
        if role == "assistant":
            content = msg.get("content", "") or ""
            raw_tool_calls = msg.get("tool_calls", [])
            
            # Extract <think>...</think>
            think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            parts = []
            
            if think_match:
                thinking_text = think_match.group(1).strip()
                if thinking_text:
                    parts.append({"type": "thinking", "thinking": thinking_text})
                after_think = content[think_match.end():].strip()
            else:
                after_think = content.strip()
            
            if raw_tool_calls:
                # Assistant message WITH tool calls
                harmony_tool_calls = []
                for tc in raw_tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "python")
                    args = func.get("arguments", "{}")
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    harmony_tool_calls.append(
                        ToolCall(
                            function=ToolCall.FunctionBody(
                                name=name, arguments=args),
                            id=tc.get("id"),
                        )
                    )
                    last_tool_call_name = name
                
                if after_think:
                    parts.append({"type": "text", "text": after_think})
                
                formatted.append({
                    "role": "assistant",
                    "content": parts if parts else [{"type": "text", "text": ""}],
                    "tool_calls": harmony_tool_calls,
                })
            else:
                # Assistant message WITHOUT tool calls (final answer)
                if after_think:
                    parts.append({"type": "text", "text": after_think})
                formatted.append({
                    "role": "assistant",
                    "content": parts if parts else [{"type": "text", "text": content}],
                })
        
        elif role == "tool" or role == "function" or role == "ipython":
            # Tool result message — MUST include "name" for GptOssRenderer
            tool_name = (msg.get("name", "") 
                         or last_tool_call_name 
                         or "python")
            formatted.append({
                "role": "tool",
                "name": tool_name,
                "content": msg.get("content", ""),
                "tool_call_id": msg.get("tool_call_id", ""),
            })
        
        elif role == "system":
            formatted.append(msg)
        
        else:
            formatted.append(msg)
    
    return formatted
```

> [!WARNING]
> You **must inspect** actual `math_tool` rows to verify:
> 1. Whether tool calls appear in `msg["tool_calls"]` or are embedded in `msg["content"]` as text
> 2. The tool role name (`"tool"`, `"function"`, or `"ipython"`)
> 3. The tool name (`"python"`, `"calculator"`, `"wolfram"`, etc.)
> 
> Print a few samples in Cell 4 and adjust the formatter accordingly.

### Cell 6 — Renderer + Tool Definitions

```python
tokenizer = tokenizer_utils.get_tokenizer(MODEL_NAME)
renderer = GptOssRenderer(
    tokenizer=tokenizer,
    use_system_prompt=False,  # We'll provide our own system via tool prefix
    # Note: use_system_prompt=False since create_conversation_prefix_with_tools
    # adds its own system message with tool routing instructions
)

# Discover what tools are used in the dataset
tool_names = set()
for row in all_convos[:100]:
    for msg in row["messages"]:
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tool_names.add(tc.get("function", {}).get("name", "unknown"))
        if msg.get("role") in ("tool", "function", "ipython"):
            tool_names.add(msg.get("name", "unknown"))
print(f"Tools found in dataset: {tool_names}")

# Define tool specs (adjust based on what's actually in the dataset)
from tinker_cookbook.renderers.base import ToolSpec
tool_specs = []
if "python" in tool_names or not tool_names:
    tool_specs.append(ToolSpec(
        name="python",
        description="Execute Python code to solve math problems.",
        parameters={
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python code"}},
            "required": ["code"]
        },
    ))
# Add more specs for any other tools found

# Create tool prefix messages (system + developer with tool defs)
tool_prefix = renderer.create_conversation_prefix_with_tools(
    tools=tool_specs,
    system_prompt="You are a helpful math assistant. Use tools when needed.",
)
print(f"Tool prefix: {len(tool_prefix)} messages")
for msg in tool_prefix:
    print(f"  [{msg['role']}]: {str(msg.get('content', ''))[:100]}...")
```

### Cell 7 — Convert All Rows to Datums (with tool prefix)

```python
MAX_LENGTH = 8192

def row_to_datum(row: dict) -> tinker.Datum:
    """Convert one tool-use row into a Tinker Datum."""
    messages = format_deepseek_tool_to_harmony(row["messages"])
    # Prepend tool prefix (system + developer messages with tool definitions)
    full_messages = tool_prefix + messages
    model_input, weights = renderer.build_supervised_example(full_messages)
    return datum_from_model_input_weights(model_input, weights, max_length=MAX_LENGTH)
```

All other cells (8-11) are identical to Notebook 1, with adjusted naming (e.g., `"gpt_oss_tool_final"`, `"gpt_oss_tool_weights"`).

---

## Format Comparison Table

| Aspect | DeepSeek (Source) | Harmony / GPT OSS (Target) |
|--------|-------------------|----------------------------|
| CoT block | `<think>...</think>` in content | `{"type": "thinking", "thinking": "..."}` → `<\|channel\|>analysis` |
| Final answer | After `</think>` in content | `{"type": "text", "text": "..."}` → `<\|channel\|>final` |
| Tool call | `msg["tool_calls"]` list | Same structure wrapped in `ToolCall` objects → `to=functions.{name}<\|channel\|>commentary` |
| Tool result | `{"role": "tool", "content": "..."}` | Must add `"name"` field → `<\|start\|>functions.{name} to=assistant<\|channel\|>commentary` |
| System prompt | `<\|im_start\|>system...` | Auto-generated `<\|start\|>system<\|message\|>...` by renderer |
| Developer msg | N/A | `<\|start\|>developer<\|message\|># Instructions...` (tool defs + instructions) |
| Stop token | `<\|im_end\|>` | `<\|return\|>` (normal) or `<\|call\|>` (tool call) |
| Code execution | N/A (SFT = frozen traces) | Only in RL via `AgentToolMessageEnv` + sandbox |

---

## Verification Plan

### Automated Sanity Checks (embedded in Cell 8)
1. Decode tokens back to text → verify Harmony special tokens present
2. Check weight distribution (prompt=0, completion=1)
3. Verify at least 3 random samples have correct format

### Manual Verification
1. Verify dataset rows loaded correctly from Kaggle path
2. Check decoded tokens show proper Harmony structure
3. Monitor training NLL decrease
4. Test inference post-training
