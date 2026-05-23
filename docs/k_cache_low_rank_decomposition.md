# K Cache 低秩分解（SVD）压缩与重建

本文档详述 ShadowKV 中 Key Cache 的 SVD 低秩压缩原理、代码实现、以及 Decode 阶段基于 PyTorch gather/GEMM 的在线重建流程。

---

## 1. 核心思想

完整的 Key Cache 形状为 `[bsz, seq_len, num_kv_heads * head_dim]`（如 `[bsz, 128K, 1024]`）。直接存储显存占用巨大。ShadowKV 的做法是：

1. **Prefill 阶段**：对 Pre-RoPE 的 Key Cache 做 SVD 分解，只保留前 `rank` 个奇异值/向量（默认 rank=160）
2. **Decode 阶段**：依据检索到的稀疏 position_ids，从 $U$ 中 gather 对应行，再与 $SV$ 做矩阵乘 + RoPE，在线重建出 Post-RoPE 的 Key

**压缩率**：原始 Key 每个 token 需要 `num_kv_heads * head_dim = 1024` 个元素，SVD 后只需存储 $U$ 的 `rank=160` 个元素/token + 每层固定的 $SV$ 矩阵。压缩比约为 **6.4×**。

> **关键设计选择**：SVD 作用于 **Pre-RoPE key states**（即旋转位置编码之前）。这是因为 RoPE 是 position-dependent 的，如果压缩 Post-RoPE key，则 $U$ 矩阵中会混入位置信息，无法稀疏 gather 并正确重建。Pre-RoPE key 是 position-independent 的，因此 $U$ 的每一行对应一个 token，可以自由 gather。

---

## 2. SVD 压缩（Prefill 阶段）

### 2.1 调用链

```
base.py::layer_compute() L131
  └── kv_cache.get_svd(key_states, layer_idx)   # key_states 是 Pre-RoPE 的
  └── apply_rotary_pos_emb(query, key, position_ids)   # 之后才做 RoPE
  └── kv_cache.prefill_kv_cache(value, key_roped, ...)
```

**重点**：`get_svd()` 在 `apply_rotary_pos_emb()` **之前**被调用，确保输入是 Pre-RoPE key。

### 2.2 `ShadowKVCache.get_svd()` — 参考实现 (`kv_cache.py` L195-213)

```python
def get_svd(self, new_k_cache, layer_idx):
    # 输入: [bsz, num_kv_heads, prefill_len, head_dim] 或 [bsz, prefill_len, kv_heads*head_dim]
    # Step 1: 将多头合并为单个大矩阵
    if new_k_cache.shape[1] <= 32:
        k_cache = new_k_cache.transpose(1, 2).reshape(
            self.batch_size, -1, self.num_key_value_heads * self.head_dim
        )  # [bsz, seq_len, 1024]

    # Step 2: SVD 分解（float32 精度）
    u, s, v = torch.svd(k_cache.float())
    v = v.transpose(1, 2)  # v: [bsz, 1024, 1024] → [bsz, 1024, seq_len] 不对，pytorch svd 返回 V 不是 V^T

    # Step 3: 截断保留 top-rank 分量
    self.U[layer_idx] = u[:, :, :self.rank]      # [bsz, seq_len, rank]
    self.SV[layer_idx] = (diag(s[:rank]) @ v[:rank])  # [bsz, rank, 1024]
    #                    reshape → [bsz, kv_heads, rank, head_dim]
```

**数学表示**：
$$K_{pre-rope} \approx U_{:,\,:rank} \cdot \text{diag}(S_{:rank}) \cdot V_{:rank,\,:}^T = U \cdot SV$$

其中：
- $K_{pre-rope} \in \mathbb{R}^{seq\_len \times (kv\_heads \cdot head\_dim)}$
- $U \in \mathbb{R}^{seq\_len \times rank}$ — 每个 token 一行，position-independent
- $SV \in \mathbb{R}^{rank \times (kv\_heads \cdot head\_dim)}$ — 每层固定，reshape 为 `[kv_heads, rank, head_dim]`

---

## 3. K Cache 重建（Decode 阶段）

### 3.1 参考实现：`ShadowKVCache.get_key_cache()` (`kv_cache.py` L314-334)

纯 PyTorch 实现，逻辑清晰：

```python
def get_key_cache(self, layer_idx, position_ids, rope_func, cos_sin_cache):
    u = self.U[layer_idx]   # [bsz, seq_len, rank]
    sv = self.SV[layer_idx] # [bsz, kv_heads, rank, head_dim]

    # Step 1: 从 U 中按 position_ids gather 出对应行
    # position_ids: [bsz, kv_heads, sparse_budget]
    index_expanded = position_ids.unsqueeze(-1).expand(-1, -1, -1, rank)
    u_expand = u.unsqueeze(1).expand(-1, kv_heads, -1, -1)
    U_head = torch.gather(u_expand, 2, index_expanded)
    # U_head: [bsz, kv_heads, sparse_budget, rank]

    # Step 2: 矩阵乘重建 Pre-RoPE key
    result = torch.einsum('bhrk,bhkd->bhrd', U_head, sv)
    # result: [bsz, kv_heads, sparse_budget, head_dim]

    # Step 3: 对重建的 key 施加 RoPE
    result = rope_func(result, position_ids)

    # Step 4: 写入 GPU buffer
    self.k_cache_buffer[layer_idx][:, :, sparse_start:sparse_end].copy_(result)
```

**数学过程**：
$$K_{post-rope}[i] = \text{RoPE}(U[pos_i, :] \cdot SV, pos_i)$$

---

## 4. 数据流总览

```
Prefill:
  key_states (Pre-RoPE)
      │
      ├── get_svd() ──→ U [bsz, seq_len, rank] (GPU)
      │                   SV [bsz, kv_heads, rank, head_dim] (GPU)
      │
      ├── apply_rotary_pos_emb() ──→ key_states_roped (Post-RoPE)
      │
      └── prefill_kv_cache()
              ├── outlier chunks (Post-RoPE) ──→ k_cache_buffer (GPU, 常驻)
              └── local chunks (Post-RoPE) ──→ k_cache_buffer (GPU, 常驻)


Decode:
  get_retrieval_position_ids()
      └── position_ids: [bsz, kv_heads, select_sets] (chunk indices)

  get_key_cache()
      └── torch.gather + torch.einsum + RoPE
              ├── U[position_ids] × SV = Pre-RoPE K
              └── RoPE: Pre-RoPE K → Post-RoPE K → k_cache_buffer
```

---

## 5. 存储开销分析

以 Llama-3.1-8B（`kv_heads=8, head_dim=128, rank=160`）+ 128K context 为例：

| 组件 | 形状 | 每层大小 (bf16) | 说明 |
|------|------|-----------------|------|
| 原始 K cache | `[bsz, 128K, 8, 128]` | 128K × 1024 × 2B = **256 MB** | 如果直接存 |
| U | `[bsz, 128K, 160]` | 128K × 160 × 2B = **40 MB** | SVD 左矩阵 |
| SV | `[bsz, 8, 160, 128]` | 8 × 160 × 128 × 2B = **320 KB** | SVD 右矩阵（极小） |
| **总计** | — | **~40 MB** | **压缩比 ≈ 6.4×** |

> 注意：U 存储在 GPU 上，这是 Decode 阶段的主要显存开销。但相比原始 K cache 的 256 MB/层，40 MB/层 显著减少了显存占用。

---

## 6. 关键约束与注意事项

1. **Pre-RoPE 输入**：`get_svd()` **必须**在 `apply_rotary_pos_emb()` 之前调用，否则重建时 RoPE 会被双重施加。
2. **Rank 选择**：`rank=160` 是经验值，原始维度为 1024（`8 heads × 128 head_dim`）。更大的 rank 提高精度但增加存储和计算。
3. **SV 布局**：`ShadowKVCache` 存 `[kv_heads, rank, head_dim]`，decode 时通过 `torch.einsum` 与 gathered `U` 重建 K。
4. **Chunk-level 索引**：`position_ids` 由 chunk-level 选择展开为 token-level，并受 `active_select_sets` 边界保护。
5. **精度**：SVD 在 `float32` 下执行，结果转回 `bfloat16` 存储。
