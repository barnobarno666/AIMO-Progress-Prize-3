# Goal Description
Improve the `NEMOTRON V2(WORKING).ipynb` notebook configuration for the vLLM server to match the recommended best practices for the NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 (and FP8) models running on single GPUs, as detailed in the [vllm_cookbook.ipynb](file:///d:/ALL%20CODES/AIMO%20Progress%20Prize%203/Reference%20Code/vllm_cookbook.ipynb).

## Proposed Changes

### [Component: vLLM Server Setup]
#### [MODIFY] NEMOTRON V2(WORKING).ipynb
1. **Reduce GPU Memory Utilization Risk:**
   - Change `CFG.gpu_memory_utilization` from `0.96` to `0.9` to provide enough memory room during first-run compilation (especially for `TRITON_ATTN`).
2. **Add Explicit Parallelism Parameters to vLLM:**
   - Append `--pipeline-parallel-size`, `1`
   - Append `--data-parallel-size`, `1`
3. **Use Triton Attention Backend:**
   - Append `--attention-backend`, `TRITON_ATTN` (strongly recommended in the cookbook config for NVFP4/FP8 to maximize throughput).
4. **Setup proper reasoning plugin:**
   - Replace the default `--reasoning-parser` `nemotron_v3` with the dedicated `super_v3_reasoning_parser.py` as depicted in the cookbook.
   - We will insert a small python script segment before launching the server to resolve the parser plugin path (it will check the model dir, and fallback to downloading it if missing).
   - Change the `vllm serve` arguments to:
     `'--reasoning-parser-plugin', parser_path`
     `'--reasoning-parser', 'super_v3'`

## Verification Plan

### Manual Verification
- The user will need to run the notebook in their environment (Kaggle or otherwise) with adjusting dataset paths to confirm that the vLLM API server starts up successfully and serves requests without Out-of-Memory (OOM) errors during autotuning.
