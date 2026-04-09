# RLVR Training Notebook for GPT OSS 120B via Tinker

## Background

Build a Kaggle notebook that trains `openai/gpt-oss-120b` using **Reinforcement Learning with Verifiable Rewards (RLVR)** via the Tinker `tinker_cookbook.rl` framework. This is a **no-tool, single-turn** math RL setup where the model generates reasoning+answer, and a verifier checks the final `\boxed{}` answer for correctness.

### Key inputs
- **Dataset**: `my_dataset.csv` (columns: `id`, `problem`, `answer`) — uploaded to Kaggle as a Kaggle Dataset.
- **Model**: `openai/gpt-oss-120b` via Tinker cloud (same model used in the SFT notebook).
- **Working SFT reference**: The existing `No tool.ipynb` which demonstrates `GptOssRenderer`, `ServiceClient`, `TrainingClient`, token rendering, etc.

---

## Proposed Changes

### [NEW] [rlvr_training.py](file:///d:/ALL%20CODES/AIMO%20Progress%20Prize%203/NHR/rlvr_training.py)

A pure Python script (not `.ipynb`) organized into clearly marked sections, intended to be copy-pasted into Kaggle notebook cells. Using `.py` avoids the `.ipynb` edit restriction.

---

## Architecture — How Tinker RL Works

From reading the `tinker_cookbook` source:

1. **`ProblemEnv`** — Single-turn Q&A environment. Subclass it, implement `get_question()`, `check_answer()`, `check_format()`, `get_reference_answer()`. The base class handles prompt rendering via `Renderer`, computes reward as `format_coef * (format - 1) + correct_answer`.
2. **`ProblemGroupBuilder`** — Creates `group_size` copies of the env for GRPO (multiple completions per problem).
3. **`RLDataset` / `RLDatasetBuilder`** — Provides batches of `EnvGroupBuilder` instances.
4. **`train.Config`** — Main config object: model, learning rate, max_tokens, loss_fn, dataset_builder, lora_rank, etc.
5. **`train.main(config)`** — Async orchestrator that creates ServiceClient, TrainingClient, builds dataset, runs sync/async training loop with rollouts, checkpointing, and evaluation.

### Our Implementation

We create a **custom `RLDatasetBuilder`** that:
- Loads `my_dataset.csv` from Kaggle
- Creates `MathEnv` instances with our `problem` and `answer` columns
- Uses the `GptOssRenderer` for proper Harmony format rendering

Then we configure `train.Config` and call `train.main(config)`.

---

## Notebook Cell Structure

### Cell 1 — Configuration

```python
class RLConfig:
    model_name = "openai/gpt-oss-120b"
    lora_rank = 32
    learning_rate = 1e-5            # Lower LR for RL (vs 2e-4 for SFT)
    max_tokens = 24576              # Generation budget per rollout
    batch_size = 8                  # Problems per training step
    group_size = 4                  # Completions per problem (GRPO)
    max_steps = 100                 # Total training iterations
    temperature = 1.0               # Sampling temperature for rollouts
    loss_fn = "importance_sampling" # Standard for GRPO
    eval_every = 10                 # Evaluate every N steps
    save_every = 25                 # Checkpoint every N steps
    format_coef = 0.1              # Weight for format reward
    log_path = "/kaggle/working/rl_logs"
```

### Cell 2 — Installs & Setup

```
!pip install -q tinker tinker-cookbook
```
Set `TINKER_API_KEY`.

### Cell 3 — Imports

Standard imports + Tinker cookbook RL modules.

### Cell 4 — Dataset Loading

Load `my_dataset.csv` from Kaggle input, parse `problem` and `answer` columns.

### Cell 5 — Custom MathEnv (ProblemEnv subclass)

Implements:
- `get_question()` → returns `problem + " Write your answer in \\boxed{} format."`
- `check_answer()` → extracts `\boxed{}` from response, compares with ground truth (using `grade_answer` from tinker_cookbook if available, else string matching)
- `check_format()` → checks if `\boxed{}` is present
- `get_reference_answer()` → returns ground truth

### Cell 6 — Custom RLDataset & RLDatasetBuilder

Creates the dataset that:
1. Shuffles problems
2. Provides batches of `ProblemGroupBuilder` instances
3. Each builder creates `group_size` copies of `AIMOMathEnv`

### Cell 7 — RL Training Launch

```python
import asyncio
from tinker_cookbook.rl import train

config = train.Config(
    model_name=RLConfig.model_name,
    learning_rate=RLConfig.learning_rate,
    dataset_builder=our_dataset_builder,
    max_tokens=RLConfig.max_tokens,
    log_path=RLConfig.log_path,
    lora_rank=RLConfig.lora_rank,
    loss_fn=RLConfig.loss_fn,
    temperature=RLConfig.temperature,
    eval_every=RLConfig.eval_every,
    save_every=RLConfig.save_every,
    max_steps=RLConfig.max_steps,
    renderer_name="gpt_oss",
)
asyncio.run(train.main(config))
```

### Cell 8 — Download Weights

Reuses the pattern from the SFT notebook to download the final checkpoint tar.

---

## User Review Required

> [!IMPORTANT]
> **No Tool Use**: This notebook implements **single-turn** RLVR without any code execution tools. The model generates reasoning and a final `\boxed{}` answer in one shot. The original implementation plan mentioned tool use with Modal sandboxes — we are deliberately NOT implementing that here for simplicity and cost efficiency.

> [!IMPORTANT]
> **Tinker Cookbook RL Framework**: Instead of manually implementing the GRPO training loop (forward_backward + optim_step), we leverage `tinker_cookbook.rl.train.main()` which handles:
> - Rollout generation (sampling completions)
> - Advantage computation (GRPO group-relative)
> - Policy optimization (importance sampling loss)  
> - Checkpointing & evaluation
> 
> This is the **officially supported** way to do RL with Tinker.

> [!WARNING]
> **Renderer**: We use `renderer_name="gpt_oss"` which maps to `GptOssRenderer` internally. This handles the Harmony format (`<|start|>`, `<|channel|>`, `<|message|>`, etc.) automatically for both prompt construction and response parsing.

> [!WARNING]
> **Answer Grading**: The Tinker cookbook includes a robust `grade_answer` function using SymPy for symbolic math comparison. We'll use this rather than simple string matching, which handles equivalent expressions like `\frac{1}{2}` vs `0.5`.

---

## Open Questions

> [!IMPORTANT]
> 1. **SFT Checkpoint Warm-Start**: Should we load the SFT checkpoint from the previous training run (`tinker://0399b1f6-...`) as a starting point for RL? This is standard practice (SFT → RL pipeline). If yes, we'll set `load_checkpoint_path` in the config. Please provide the checkpoint path if you want this.

> [!IMPORTANT]
> 2. **Dataset Size**: `my_dataset.csv` will be loaded from Kaggle. How many rows does it have? With `batch_size=8` and `group_size=4`, each step trains on 8 problems × 4 completions = 32 rollouts. At 100 steps, we need at least 800 problems.

> [!IMPORTANT]
> 3. **API Key**: The SFT notebook had the API key hardcoded. Should I use the same key or use a Kaggle secret?

---

## Verification Plan

### Automated Tests
- The script is syntactically valid Python
- Import structure matches the tinker-cookbook API
- Dataset loading and custom Env work correctly (can test locally with CSV)

### Manual Verification
- Upload to Kaggle, attach dataset, run the notebook
- Monitor Tinker training metrics (reward, format accuracy, correct accuracy) across steps
- Verify checkpoints are saved and can be downloaded
