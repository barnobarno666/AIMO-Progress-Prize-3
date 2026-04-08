# RLVR Training with Tool Use (Tinker Notebook) — Implementation Plan

This plan details the creation of a Kaggle notebook for Reinforcement Learning with Verifiable Rewards (RLVR) training using the `openai/gpt-oss-120b` model within the Tinker framework. The notebook will feature an online sandboxed Python execution tool, rewarding the model solely based on final answer correctness.

---

## Context & Key Decisions

> [!IMPORTANT]  
> **Framework & Model**: `openai/gpt-oss-120b` using Tinker's `AgentToolMessageEnv` for multi-turn RL training. We use the `tinker_cookbook.rl` and `tinker_cookbook.tool_use` libraries.

> [!IMPORTANT]
> **Tool Execution Sandbox**: The execution environment relies on Tinker Cloud for training compute. The code execution sandbox will use Tinker's built-in Modal cloud sandbox support (`SandboxBackend.MODAL`), avoiding any local Docker dependencies. The notebook itself will run efficiently on CPU, orchestrating the cloud compute via Tinker's API.

> [!IMPORTANT]
> **Dataset**: `nvidia/Nemotron-Math-v2`. We will filter for specific datapoints: `reason_high_with_tool` accuracy $\le$ 0.25 (meaning at most 2 out of 8 passes) and `reason_high_no_tool` accuracy = 0. This targets hard problems where tool use is required but current tool performance is low.

> [!IMPORTANT]
> **Reward Design**: 
> - **No complex intermediate step rewards** or explicit tool-calling penalties.
> - **Final Answer Reward**: Positive reward (+1.0 or higher) for extracting the correct integer answer from `\boxed{}`.
> - **Negative Penalty**: Negative reward (-1.0) for incorrect answers or failure to produce a valid `\boxed{}` answer format before the token limit is reached.
> - The model is naturally penalized for bad code because it wastes its token budget and time, leading to a failure to answer.

---

## Proposed Notebook Structure

**File**: `@[NHR/rlvr_tinker_training.ipynb]`

### Cell 1 — Configuration Map
Sets up hyperparams, batch sizes, and dataset filters.
```python
class Config:
    model_name = "openai/gpt-oss-120b"
    lora_rank = 32
    max_tokens = 24576          # Max generation tokens per episode
    learning_rate = 1e-5
    group_size = 4              # GRPO group size (completions per prompt)
    groups_per_batch = 8        # Number of problems per step
    max_steps = 100             # Total training steps
    
    # Dataset Filtering
    max_samples = 500           # Adjustable subset size
    max_tool_acc = 0.25         # reason_high_with_tool.accuracy <= 0.25
    req_notool_acc = 0.0        # reason_high_no_tool.accuracy == 0.0
```

### Cell 2 — Installs & Setup
Installs Tinker, unsloth, trl, vllm, and the modal sandbox dependency (`tinker-cookbook[modal]`). Ensure Tinker API keys and Modal tokens are configured.

### Cell 3 — Dataset Preparation
Loads `nvidia/Nemotron-Math-v2`.
Filters rows based on the user's JSON metadata requirements:
```python
def filter_row(row):
    try:
        meta = json.loads(row['metadata'])
        tool_acc = meta.get('reason_high_with_tool', {}).get('accuracy', 1.0)
        no_tool_acc = meta.get('reason_high_no_tool', {}).get('accuracy', 1.0)
        return tool_acc <= Config.max_tool_acc and no_tool_acc == Config.req_notool_acc
    except:
        return False
```
Extracts `problem` and `answer` pairs.

### Cell 4 — Tool Definition (`@tool` & Harmony Format)
Leverages Tinker's `tool_use` library to define a stateful Python execution tool. Follows OpenAI Harmony specification for function calling (using the `functions` namespace and `to=functions.execute_python` routing).
```python
from tinker_cookbook.tool_use import tool, simple_tool_result

class PythonExecutionTool:
    def __init__(self, backend_pool):
        self.backend_pool = backend_pool
        
    @tool
    async def execute_python(self, code: str) -> ToolResult:
        """Run python code and get the console output."""
        result = await self.backend_pool.run_in_workdir(
            files={"code.py": code}, 
            command=["python", "code.py"]
        )
        return simple_tool_result(result.stdout + "\n" + result.stderr)
```

### Cell 5 — Reward Function
Defines the final answer reward logic.
```python
def extract_boxed_answer(text):
    # Regex logic to find \boxed{X} 
    pass

class ExactMatchReward:
    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        # 1. Get final assistant message
        # 2. Extract \boxed{} answer
        # 3. Compare with Ground Truth associated with the episode
        # 4. Return +1.0 for match, -1.0 for mismatch/missing
        pass
```

### Cell 6 — Environment Builder
Uses `build_agent_tool_env` to wire together the dataset problem, the Python tool, and the reward function. Uses the Modal sandbox backend (`SandboxBackend.MODAL`).
```python
from tinker_cookbook.tool_use import build_agent_tool_env

# Factory to create environments for the InterleavedRLDatasetBuilder
```

### Cell 7 — RL Training Loop
Sets up the `Config` (from `tinker_cookbook.rl.train`) targeting Tinker's cloud training architecture (`forward_backward`, `optim_step`, `sample`). The notebook manages the rollout flow locally via the `Config` abstractions while Tinker handles the distributed model computation.


> [!WARNING]
> **Tokenizer / Renderer**: As seen in the SFT plan, GPT OSS uses custom Discord/Harmony tokens (`<|channel|>`, `<|message|>`, etc.). We will use Tinker's built-in `GptOssRenderer` combined with their standard tool handling to cleanly map to the Harmony `<|call|>` structures and `functions` namespaces.

Let me know if this updated structure looks right to you, and I will proceed with creating the notebook!
