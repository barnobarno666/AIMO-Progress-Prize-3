# Revise `Tinker RLVR.ipynb` With 10-Train / 1-Val Split

**Summary**
- Keep `group_size = 8`, `max_trajectory_tokens = 24576`, `max_tokens = 4096`, and `max_tool_iterations = 15` unchanged.
- Refactor the notebook so tool use is aligned with the `python`-recipient style in [stable_deepconf.ipynb](/home/nhr13/AIMO-Combined/AIMO-Progress-Prize-3/NHR/stable_deepconf.ipynb), while still accepting `functions.python` as a fallback.
- Reduce burn via fewer training steps, smaller batch size, and a tiny dataset split: train on the literal last 10 CSV rows, then pick exactly 1 validation row by seeded random sampling from the remaining rows.

**Key Changes**
- Update Cell 1 config:
  - `batch_size = 2`
  - `group_size = 8` unchanged
  - `max_steps = 30`
  - `max_tokens = 4096` unchanged
  - `max_trajectory_tokens = 24576` unchanged
  - `max_tool_iterations = 15` unchanged
  - `eval_every = 30`
  - `save_every = 15`
  - add split controls such as `train_last_n = 10`, `eval_size = 1`, `split_seed = 42`
- Replace the current thread-timeout `CodeExecutor` in Cell 5 with a persistent per-env worker-process executor:
  - state persists across tool calls within an episode
  - timeout kills and recreates the worker
  - `reset()` restores a clean namespace
  - this removes the runaway-daemon-thread failure mode from the current notebook
- Rewrite Cell 7 tool parsing:
  - canonical format is `recipient == "python"` / `name == "python"`
  - accept parsed Tinker `tool_calls` first
  - accept `functions.python` as compatibility fallback
  - recover malformed Harmony-style raw text patterns seen in the saved run
  - normalize every successful parse into one internal tool-intent object before execution
- Tighten Cell 7 episode control to stop wasting turns:
  - valid tool call executes and continues with `reward = 0.0`
  - malformed tool call gets negative feedback and at most one repair turn
  - empty `final` or non-boxed `final` ends the episode with negative reward
  - repeated reasoning-only turns without progress terminate as `stalled`
  - tool timeout or worker crash ends the episode immediately
- Update Cell 9 dataset builder:
  - no shuffle for training-set construction
  - training set = last 10 CSV rows in file order
  - candidate pool for validation = all earlier rows
  - validation set = exactly 1 row sampled from that pool using `split_seed`
  - all other non-training rows are excluded from both train and validation for this run
- Add a preflight cell before training:
  - tests for `python`, `functions.python`, malformed JSON, empty `final`, timeout code, and state persistence
  - fail fast before launching the 30-step run

**Public Interfaces / Types**
- Replace `CodeExecutor` with a worker-backed executor that guarantees hard timeout cleanup.
- Replace `_extract_code_from_message()` with a normalized parser returning structured tool intent.
- Extend environment metrics with:
  - `tool_call_ok`
  - `tool_format_error`
  - `fallback_parse_used`
  - `tool_timeout`
  - `invalid_final`
  - `stalled`
  - `canonical_python_recipient`

**Test Plan**
- Parser tests:
  - parsed `python` call succeeds
  - parsed `functions.python` call succeeds
  - malformed Harmony raw text is either recovered or explicitly penalized
  - empty `final` and non-boxed `final` terminate correctly
- Executor tests:
  - state persists across two calls in one episode
  - infinite loop times out, worker is replaced, and no stale execution survives
  - reset clears namespace
- Dataset tests:
  - training rows are exactly the last 10 CSV rows
  - validation row is exactly 1 sampled row from the earlier rows using the fixed seed
  - no other rows appear in train or validation
- Training acceptance:
  - a smoke run completes without hanging at the next sampling batch
  - the main run uses `2 x 8 = 16` rollouts per step
  - logs show successful `python` tool executions and fewer stalled / empty-final trajectories

**Assumptions**
- The dataset has at least 11 rows.
- “Train on 10” means the literal last 10 rows in CSV order.
- “Validation on 1” means one seeded-random row chosen from the remaining earlier rows, with all other rows ignored for this run.
- The notebook stays on the Tinker RL stack and is not rewritten into a standalone Harmony sampler loop.

**References**
- [Tinker RLVR notebook](/home/nhr13/AIMO-Combined/AIMO-Progress-Prize-3/Tinker%20Training%20SFT.ipynb/Tinker%20RLVR.ipynb)
- [stable_deepconf.ipynb](/home/nhr13/AIMO-Combined/AIMO-Progress-Prize-3/NHR/stable_deepconf.ipynb)
- Tinker rendering docs: https://tinker-docs.thinkingmachines.ai/rendering
- Tinker RL API docs: https://tinker-docs.thinkingmachines.ai/cookbook/api-reference/rl/
