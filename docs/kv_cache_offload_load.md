# KV Cache Offload & Load 机制详解

本文档详细描述 ShadowKV 中 KV Cache 的 Offload（GPU → CPU）和 Load（CPU → GPU）机制。

---

## 1. 整体架构

ShadowKV 的核心思想：Prefill 阶段将完整 KV Cache **压缩/卸载**到 CPU，Decoding 阶段只按需 **稀疏加载** 被 top-k 检索选中的 chunk。

```
Prefill:  GPU [full KV] ──offload──> CPU [v_cache_cpu (pinned)] + CPU [U, SV (SVD)]
                                     GPU [outlier_k/v + local_k/v (buffer)]

Decode:   CPU [v_cache_cpu] ──sparse load──> GPU [v_cache_buffer]
          GPU [U] × GPU [SV] ──GEMM+RoPE──> GPU [k_cache_buffer]
```

---

## 2. Offload（GPU → CPU）— Prefill 阶段

### 2.1 Value Cache Offload

**位置**: `ShadowKVCache_CPU.prefill_kv_cache()` — `models/kv_cache.py` L543

```python
self.v_cache_cpu[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :max_ctx_chunks].copy_(
    new_v_cache[:, :, :self.max_ctx_chunks_len].reshape(
        bsz, self.num_key_value_heads, max_ctx_chunks, self.chunk_size * self.head_dim
    ),
    non_blocking=True
)
```

**关键设计**:
- `v_cache_cpu` 在 `__init__` 中以 `pin_memory=True` 分配在 CPU 上（L400-409），形状为 `[num_layers, batch_size, num_kv_heads, max_chunks, chunk_size * head_dim]`
- 使用 `.copy_(non_blocking=True)` 实现异步 GPU→CPU 拷贝
- Value cache 按 chunk 重排存储（`chunk_size * head_dim` 合并为最后一维），便于后续按 chunk 稀疏读取

### 2.2 Key Cache 压缩（SVD）

**位置**: `ShadowKVCache_CPU.get_svd()` — `models/kv_cache.py` L480-513

Key cache **不直接 offload**，而是通过 SVD 分解为低秩表示：
- `U`: `[bsz, seq_len, rank]`，存 CPU
- `SV`: `[bsz, num_kv_heads, head_dim, rank]`，存 CPU

Prefill 结束后由 `H2D()` 统一搬到 GPU。

### 2.3 Outlier & Local Chunks（保留在 GPU）

Prefill 阶段还会：
1. 计算 landmark（chunk mean）与原始 key 的余弦相似度
2. 选出 `outlier_chunk` 个异常 chunk 的 K/V，直接写入 GPU buffer
3. 尾部 `local_chunk` 个 chunk 的 K/V 也直接写入 GPU buffer

这部分始终留在 GPU 上，不走 offload/load 流程。

---

## 3. Load（CPU → GPU）— Decoding 阶段

### 3.1 Chunk 检索（决定加载哪些 chunk）

**位置**: `ShadowKVCache_CPU.get_retrieval_position_ids()` — `models/kv_cache.py` L616-645

1. Query 与 `k_landmark` 做 fused GEMM + Softmax（`shadowkv.batch_gemm_softmax`）
2. Top-k 选出 `select_sets` 个 chunk
3. `shadowkv.reorder_keys_and_compute_offsets()` 计算 `offsets` 和 `cnts`，为后续 gather_copy 准备稀疏索引

### 3.2 Value Cache Load（CPU → GPU）

**位置**: `ShadowKVCache_CPU.get_value_cache()` — `models/kv_cache.py` L647-653

```python
shadowkv.gather_copy_with_offsets(
    self.v_cache_cpu[layer_idx],    # 源: CPU pinned memory
    self.v_cache_buffer[layer_idx], # 目标: GPU buffer
    self.temp, self.offsets, self.cnts, self.signals,
    self.batch_size, self.num_key_value_heads,
    int(self.max_ctx_chunks_len * self.head_dim),
    int(self.sparse_budget * self.head_dim),
    self.kernel_offset, self.kernel_stride, self.select_sets
)
```

**底层 CUDA 实现**: `kernels/gather_copy.cu` L239-312
- Kernel: `gather_copy_var_midpoint_BP`（`kernels/copy.cuh` L438-492）
- 利用共享内存 + `PTYPE`（`int4` / `int2`）实现高带宽的稀疏 CPU→GPU 拷贝
- 支持 midpoint 优化：如果部分数据已在 GPU buffer 中（cache hit），只搬缺失的部分
- `signals` 数组用于标记 cache hit/miss

### 3.3 Key Cache 重建（在线 GEMM + RoPE）

**位置**: `ShadowKVCache_CPU.get_key_cache()` — `models/kv_cache.py` L655-666

```python
# Step 1: D2D copy — 将已缓存的 outlier K 重排到正确位置
shadowkv.gather_copy_d2d_with_offsets(
    self.k_cache_buffer[layer_idx], self.offsets, self.cnts, ...
)

# Step 2: 在线重建 — U × SV + RoPE，结果直接写入 k_cache_buffer
batch_gather_gemm_rotary_pos_emb_cuda(
    u, sv, cos_sin_cache, position_ids, self.output,
    self.chunk_size, self.k_cache_buffer[layer_idx],
    self.sparse_start, self.sparse_end, self.cnts
)
```

**底层 CUDA 实现**:
- `gather_copy_d2d_with_offsets`: `kernels/gather_copy.cu` L112-175（GPU D2D 稀疏 copy）
- `batch_gather_gemm_rotary_pos_emb_cuda`: `kernels/batch_gather_gemm.cu`（CUTLASS batched GEMM + RoPE fused kernel）

---

## 4. Buffer 布局

GPU 上的 `k_cache_buffer` / `v_cache_buffer` 的布局如下：

```
|<-- prefill_local -->|<-- outlier_chunk * chunk_size -->|<-- sparse_budget -->|<-- gen tokens -->|
|     local K/V       |        outlier K/V              |   retrieved K/V     |  decode K/V      |
|                     |                                 |                     |                  |
0              prefill_local                      sparse_start          sparse_end        sparse_end+gen_offset
```

- `prefill_local`: 尾部 local chunk，始终保留在 GPU
- `outlier`: 异常 chunk，始终保留在 GPU
- `sparse_budget` 区域: 每步 decode 动态填充
- `gen tokens`: decode 阶段新生成的 token 的 KV

---

## 5. 关键数据结构

| 变量 | 位置 | 形状 | 说明 |
|------|------|------|------|
| `v_cache_cpu` | CPU (pinned) | `[layers, bsz, kv_heads, max_chunks, chunk_size*head_dim]` | 完整 V cache |
| `U` | CPU → GPU (H2D) | `[layers, bsz, seq_len, rank]` | SVD 左矩阵 |
| `SV` | CPU → GPU (H2D) | `[layers, bsz, kv_heads, head_dim, rank]` | SVD 右矩阵 (S·V^T) |
| `k_cache_buffer` | GPU | `[layers, bsz, kv_heads, budget+extra, head_dim]` | K cache GPU 工作区 |
| `v_cache_buffer` | GPU | `[layers, bsz, kv_heads, budget+extra, head_dim]` | V cache GPU 工作区 |
| `offsets` | GPU | `[block_num * select_sets]` | 稀疏 gather 偏移量 |
| `cnts` | GPU | `[block_num]` | cache hit 计数 |
| `signals` | GPU | `[block_num]` | hit/miss 标记 |
| `position_ids` | GPU | `[layers, bsz, kv_heads, select_sets]` | 当前选中的 chunk 索引 |
