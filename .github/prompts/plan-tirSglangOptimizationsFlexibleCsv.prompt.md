## Plan: Implement TIR, SGLang Optimizations, and Flexible CSV for Qwen3.5 AIMO3

The current notebook uses `qwen-agent`'s `Assistant` class which provides a *partial* form of TIR — the agent does invoke the `python_executor` tool in a multi-turn loop. However, it delegates tool-call routing to the library rather than leveraging SGLang's **native OpenAI-compatible function calling**. There's also a **critical bug**: the wrong tool-call-parser is being used. The plan replaces qwen-agent with a direct OpenAI API TIR loop, fixes SGLang server flags, adds H100 optimizations, and makes local eval work with any CSV.

---

### Phase 1: Fix SGLang Server Launch (Performance + Correctness)

1. **Fix tool-call-parser (BUG)** — Change `--tool-call-parser qwen3_coder` to `--tool-call-parser qwen`. Per [SGLang docs](https://docs.sglang.io/advanced_features/tool_parser.html), `qwen3_coder` is for Qwen3-Coder models only. The `qwen` parser covers "Qwen series except Qwen3-Coder", which includes Qwen3.5-35B-A3B.
2. **Add reasoning parser** — Add `--reasoning-parser qwen3` to properly separate thinking from final output.
3. **H100 single-GPU optimizations**:
   - `--kv-cache-dtype fp8_e4m3` — FP8 KV cache saves ~50% KV memory, ~1.5x more concurrent tokens on H100.
   - `--mem-fraction-static 0.88` — increase from 0.85, safe with FP8 KV reducing pressure.
   - `--language-only` — skip loading multimodal encoder (only text needed for AIMO).
   - `--chunked-prefill-size 8192` — make explicit.
   - `--disable-log-stats` — reduce overhead.

### Phase 2: Implement True TIR via Direct OpenAI API

4. **Replace `PythonExecutor` + `qwen-agent`** with a direct OpenAI API TIR loop (*depends on step 1-3*):
   - Remove `qwen_agent` imports (`Assistant`, `BaseTool`, `register_tool`)
   - Add `from openai import OpenAI`
   - Define `PYTHON_TOOL` in OpenAI function-calling JSON schema format
   - Convert `PythonExecutor.call()` to a standalone `execute_python(code: str) -> str` function (keep subprocess + 7s timeout)
5. **Implement explicit TIR message loop** in `_predict()` (*depends on step 4*):
   - `messages = [system_msg, user_msg]`
   - Loop up to `MAX_TURNS` (e.g., 128):
     - Check global time limit + per-problem timeout
     - Call `client.chat.completions.create(messages=messages, tools=[PYTHON_TOOL], ...)`
     - If `response.choices[0].message.tool_calls` → extract code → execute → append assistant + tool result messages → continue
     - If `finish_reason == 'stop'` → extract `\boxed{}` → break
   - Reference pattern: [NHR/qwen-3.5-inference.ipynb](NHR/qwen-3.5-inference.ipynb) `_process_attempt()` method
6. **Keep generation parameters**: `temperature=1.0`, `top_p=1.0`, `presence_penalty=2.0`, `max_tokens=24576`, `top_k=40`, `enable_thinking=False` via `extra_body`

### Phase 3: Flexible CSV Handling for Local Evaluation

7. **Make CSV path configurable** (*parallel with steps 1-6*): Add `LOCAL_CSV_PATH` variable, default to the original reference.csv path for backward compatibility.
8. **Handle arbitrary CSVs**: Read any CSV → only require `problem` column (+ optional `answer` for scoring). If no `id` column, generate sequential IDs. Strip `answer` to create mock CSV, pass `id` + `problem` to `inference_server.run_local_gateway()`.

---

**Relevant files**
- [Qwen3_5/qwen3-5-sglang-starts-here.ipynb](Qwen3_5/qwen3-5-sglang-starts-here.ipynb) — main file to modify (all phases)
- [NHR/qwen-3.5-inference.ipynb](NHR/qwen-3.5-inference.ipynb) — reference for TIR loop via OpenAI API
- [IDK/36-40-gpt-oss-120b-tir-dynamictime-pooling.ipynb](IDK/36-40-gpt-oss-120b-tir-dynamictime-pooling.ipynb) — reference for parallel sampling + Jupyter kernel execution

**Verification**
1. Check `server.log` for no errors after launch with new SGLang flags
2. Verify model returns `tool_calls` (not inline code) by checking `response.choices[0].message.tool_calls is not None`
3. Test with a simple math problem requiring computation → verify tool result is fed back and `\boxed{}` extracted
4. Test CSV handling with a custom CSV (extra columns, no `id` column, different ordering)

**Decisions**
- `qwen` parser (not `qwen3_coder`) per SGLang docs — this is a correctness fix
- FP8 KV cache — negligible quality impact, significant memory savings on H100
- Subprocess execution kept (not Jupyter kernel) — simpler, safer, sufficient for single-shot 7s runs
- `enable_thinking: False` kept — saves token budget for tool-use rounds

**Further Considerations**
1. **Parallel sampling (self-consistency)**: NHR notebooks show 8-parallel traces with majority voting → significantly improves accuracy. **Recommendation**: Add as a future Phase 4 enhancement.
2. **Jupyter kernel vs subprocess**: Reference implementations use persistent stateful kernels. Better for multi-step TIR where variables carry over between turns. **Recommendation**: Start with subprocess; upgrade if needed.
3. **`enable_thinking: True`**: Qwen3.5 supports thinking mode for better math reasoning, but uses more tokens. **Recommendation**: Test both; current `False` preserves token budget for tool rounds.
