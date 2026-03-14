# 🌌 Antigravity Development System Guide: ShadowKV

This document serves as the official knowledge base and development guide for **Antigravity** when operating on the `ShadowKV` repository. It provides an in-depth code-level breakdown, architectural insights, and actionable development workflows.

---

## 🏗️ 1. Architecture Overview

ShadowKV is a training-free, high-throughput long-context LLM inference framework. Its core innovation is keeping the massive KV cache in exactly tailored "shadows" (CPU memory / low-rank representations) rather than keeping full KV tensors on limited GPU VRAM.

### High-Level Workflow:
1. **Prefilling Phase (`prefill_kv_cache`)**:
   - Computes **SVD** of the Key cache to extract low-rank representations $U$ and $SV$.
   - Computes **landmarks** (mean of chunks) and calculates cosine similarity to isolate **outlier chunks**.
   - Outlier chunks and local chunks are kept on GPU as a sparse budget.
   - The Value cache is offloaded to the CPU (`v_cache_cpu`).
2. **Decoding Phase**:
   - **Chunk Retrieval**: Incoming query computes attention against $K$-landmarks to find the most relevant chunks (`get_retrieval_position_ids`).
   - **Value Fetching**: Highly optimized custom CUDA kernels fetch the necessary $V$ chunks from CPU to GPU (`get_value_cache`).
   - **Key Reconstruction**: CUTLASS-based `batch_gather_gemm` kernels reconstruct the $K$ cache on-the-fly using $U \times SV$, applying Rotary Positional Embedding (RoPE) simultaneously (`get_key_cache`).
   - Passes the gathered/reconstructed states to `flash_attn_with_kvcache` or MInference kernels.

---

## 📂 2. Directory Structure & Key Components

### Python Modeling Layer (`models/`)
- **`kv_cache.py`**: The heart of ShadowKV. 
  - `KV_Cache`: Baseline full attention wrapper.
  - `ShadowKVCache`: ShadowKV logic tailored for accuracy tests (batch size 1).
  - `ShadowKVCache_CPU`: Highly optimized CPU offloading version serving actual inference throughput. Manages pointers, offsets, buffers, and signals for asynchronous GPU/CPU memory transfers.
- **`base.py`**: The `LLM` base wrapper managing `inference()`, `prefill()`, and `generate()`. Wraps generation loops and hooks up the custom `kv_cache.py` configurations based on `attn_mode`.
- **`tensor_op.py`**: The bridge between Python and custom CUDA kernels. Defines functions like `apply_rotary_pos_emb_cuda` and `batch_gather_gemm_rotary_pos_emb_cuda`.

### CUDA Kernels Layer (`kernels/`)
Provides Python bindings via `setup.py` (`CUDAExtension`), compiling a module exposed as `kernels.shadowkv`.
- **`batch_gather_gemm.cu`**: Implements a customized `GemmUniversalBatchGatherIndices` via CUTLASS. Highly optimized to perform batched GEMM ($U \times SV$) while simultaneously gathering specifically selected indices for the sparse budget.
- **`gather_copy.cu`**: Handles fetching `v_cache` directly from CPU `pin_memory` to GPU buffers using multi-threading. Extensively utilizes optimized `PTYPE` (like `int4` and `int2`) memory mapping.
- **`rope.cu` & `rope_new.cu`**: Accelerated, customized RoPE application directly acting on fetched buffers to save VRAM trips.

---

## 🛠️ 3. Detailed Code Knowledge & Constraints

### 3.1 Python Integration Rules
- **Memory Management**: The framework operates under extreme VRAM constraints. You will often see `gc.collect()`, `torch.cuda.empty_cache()`, and `torch.cuda.synchronize()`. **DO NOT** remove these when modifying decoding loops, as offloading deeply depends on strict memory limits.
- **Data Types**: The primary datatype is `torch.bfloat16`. Always ensure fallback values and instantiated buffers match `self.dtype`.
- **SVD Buffers (`U` and `SV`)**: 
  - $U$ shape relies heavily on rank `rank` (typically 160). 
  - Handled dynamically. If patching context lengths, ensure `chunk_size` alignment holds true.

### 3.2 CUDA / CUTLASS Guidelines
- **Header modifications**: Modifying CUTLASS logic requires extreme precision. The CUTLASS `Gemm` object in `batch_gather_gemm.cu` uses `<128, 128, 32>` threadblock sizes and `Sm80` (Ampere). **Do not** attempt to arbitrarily change block sizes without calculating SMEM usage constraints (currently bounded at ~160KB in `gather_copy.cu`).
- **Memory Copy Kernels (`gather_copy.cu`)**: Copy mappings are strictly tuned for `BLOCK_SIZE_CP (128/256)`. If changing `map_size` (currently handles 128, 256, 512, 1024), you must explicitly register the `cudaFuncSetAttribute` for `cudaFuncAttributeMaxDynamicSharedMemorySize`. Failure to do so will result in CUDA kernel launch errors.

---

## 🚀 4. Development Workflow

### Building Kernels
Whenever you modify files in `kernels/`, you MUST rebuild the extension.
```bash
python setup.py build_ext --inplace
```

### Running Tests
**Accuracy Evaluation (RULER Benchmark)**:
**必须**使用 `scripts/run_ruler.sh` 启动 RULER 精度测试，**禁止**直接调用 `test/eval_acc.py`。详见 [`docs/run_ruler.md`](docs/run_ruler.md)。
```bash
bash scripts/run_ruler.sh \
  --model /home/zijie/models/Llama-3.1-8B-Instruct \
  --gpus 0,1 --method full --num_samples 10
```

**Efficiency/Throughput Evaluation**:
Executes on single GPU.
```bash
python test/e2e.py --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" --datalen "122k"
```

### Debugging Tips 🐜
- **CUDA Errors**: If a kernel modification causes a device assert or illegal memory access, use `CUDA_LAUNCH_BLOCKING=1 python test/e2e.py ...` to trace the exact kernel throwing the fault.
- **Offload Misses**: Look at `self.offsets`, `self.cnts` and `self.signals` in `ShadowKVCache_CPU`. Print `self.position_ids` to ensure chunk limits don't exceed `self.chunks`.

## 🤖 5. Antigravity Prompt Directives
- **Zero-Shot Assumptions**: When asked to add a new model architecture to ShadowKV, reference how `llama.py` or `qwen.py` inherets/implements the `LLM` class. 
- **Variable Auditing**: When modifying `ShadowKVCache_CPU::prefill_kv_cache()`, explicitly verify the tensor shape matching against `head_dim` and `chunk_size` logic.
- **Dependency Installation Rules**: `pip install` is permitted but **strictly limited** to the `shadowkv` conda environment. You **must not** use `pip install -e .` (or equivalent editable installs) for this repository. Instead, execution relies purely on `PYTHONPATH` to resolve importing the current directory. Rely on `flash-attn`, `minference`, and built-in Torch capabilities wherever possible.

## ⚙️ 6. Environment Execution Rules
- **Strict Conda Environment**: You **MUST ALWAYS** use the `shadowkv` conda environment when executing any Python scripts or running terminal commands within this repository. 
- **Execution Format**: Ensure the environment is active before execution. For example, use `conda run -n shadowkv python <script.py>` or chain the activation like `source ~/anaconda3/etc/profile.d/conda.sh && conda activate shadowkv && python <script.py>`. Do not use the base environment.

## 📝 7. Documentation Management Rules
- **Two-Tier Structure**: All technical documentation follows a two-tier structure:
  1. **`GEMINI.md`（本文件）**: 存放 **摘要引用**（Reference & Summary）。每篇文档在下方 Section 8 中以一行简述 + 链接的形式索引。
  2. **`docs/*.md`**: 存放 **详细技术描述**。每个主题一个 `.md` 文件，包含完整的代码分析、数据流、架构图等。
- **新增文档流程**:
  1. 在 `docs/` 下创建新的 `.md` 文件，撰写详细内容。
  2. 在本文件 Section 8 中添加对应的引用条目（标题 + 一句话总结 + 相对路径链接）。
- **命名规范**: `docs/` 下的文件使用 `snake_case.md` 命名，名称应简洁且具有描述性。
- **更新规则**: 当代码发生重大变更时，同步更新对应的 `docs/` 文档和本文件中的引用。

---

## 📚 8. Documentation References

| 主题 | 文档路径 | 摘要 |
|------|----------|------|
| KV Cache Offload & Load | [`docs/kv_cache_offload_load.md`](docs/kv_cache_offload_load.md) | 详述 Prefill 阶段的 V cache CPU offload、K cache SVD 压缩，以及 Decode 阶段的稀疏 Value 加载（`gather_copy_with_offsets`）和 Key 在线重建（CUTLASS GEMM + RoPE）的完整流程与 buffer 布局。 |
| K Cache 低秩分解 | [`docs/k_cache_low_rank_decomposition.md`](docs/k_cache_low_rank_decomposition.md) | 详述 Pre-RoPE Key 的 SVD 压缩原理（rank=160）、`ShadowKVCache` vs `ShadowKVCache_CPU` 的 U/SV 存储布局差异、CUTLASS `GemmUniversalBatchGatherIndices` kernel 的 Gather-GEMM-RoPE fused 重建流程，以及 6.4× 压缩比的存储分析。 |
| RULER Benchmark 测试 | [`docs/run_ruler.md`](docs/run_ruler.md) | `scripts/run_ruler.sh` 多 GPU 并行 RULER 精度测试脚本的使用指南，包括参数说明、可用任务列表、支持的模型、输出格式及常见用法。**所有 RULER 测试必须通过此脚本启动。** |
