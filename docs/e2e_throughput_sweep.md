# E2E Throughput Sweep 测试指南

本文档说明如何使用 `test/e2e_sweep.py` 和 `test/e2e_single_run.py` 复现论文 Table 4（Generation throughput under varying batch sizes and sequence lengths）。

---

## 脚本架构

| 脚本 | 角色 | 说明 |
|------|------|------|
| `test/e2e_sweep.py` | 编排器 | 遍历所有 `(method, batch_size, datalen)` 组合，分配 GPU，收集结果 |
| `test/e2e_single_run.py` | Worker | 运行单个测试点，输出 JSON 结果 |

### 设计要点

1. **子进程隔离**：每个测试点在独立子进程中运行，OOM 不会影响后续测试。
2. **OOM 跳过**：某个 `(method, datalen)` 在 batch_size=N 时 OOM 后，自动跳过所有 batch_size > N 的组合。
3. **多 GPU 并行**：通过 `--devices cuda:0 cuda:1` 将不同 method 分配到不同 GPU，使用 `CUDA_VISIBLE_DEVICES` 隔离。
4. **结果标记**：Worker 输出 `@@RESULT@@{json}@@END@@` 标记，编排器通过正则匹配解析。

---

## 支持的参数

### Context 长度（`--datalens`）

| datalen | prompt tokens | sparse_budget | RULER 数据集长度 |
|---------|--------------|---------------|-----------------|
| 4k      | 4,096        | 128           | 4,096           |
| 8k      | 8,192        | 128           | 8,192           |
| 16k     | 16,384       | 256           | 16,384          |
| 32k     | 32,768       | 512           | 32,768          |
| 48k     | 49,152       | 768           | 65,536（截断）   |
| 60k     | 61,440       | 1,024         | 65,536（截断）   |
| 64k     | 65,536       | 1,024         | 65,536          |
| 80k     | 81,920       | 1,280         | 131,072（截断）  |
| 96k     | 98,304       | 1,536         | 131,072（截断）  |
| 122k    | 124,928      | 2,048         | 131,072（截断）  |
| 244k    | 249,856      | 4,096         | 262,144（截断）  |
| 488k    | 499,712      | 8,192         | 524,288         |

> **注意**：较长 context（48k+）复用较大的 RULER 数据集并截断至目标长度。throughput 测试中 prompt 内容不影响结果。

### 其他参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_name` | 必选 | 模型名（用于 `choose_model_class`） |
| `--model_path` | 同 model_name | 本地模型权重路径 |
| `--devices` | `cuda:0` | GPU 列表（空格分隔） |
| `--methods` | `full shadowkv_cpu` | 测试方法 |
| `--batch_sizes` | `2 3 4 5 6 8 12 16 24 32 48` | Batch size 列表 |
| `--gen_len` | `100` | 生成 token 数 |
| `--timeout` | `1800` | 单次测试超时（秒） |
| `--output_dir` | `results/` | CSV 输出目录 |

---

## 用法示例

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate shadowkv

# 快速验证（4k context，batch 2-3）
python test/e2e_sweep.py \
  --model_name "/home/zijie/models/Llama-3.1-8B-Instruct" \
  --devices cuda:0 cuda:1 \
  --datalens 4k --batch_sizes 2 3

# 中等规模测试（16k-96k）
python test/e2e_sweep.py \
  --model_name "/home/zijie/models/Llama-3.1-8B-Instruct" \
  --devices cuda:0 cuda:1 \
  --datalens 16k 32k 48k 64k

# 完整 Table 4 复现（需 A100 80GB + Gradient-1048k 模型）
python test/e2e_sweep.py \
  --model_name "gradientai/Llama-3-8B-Instruct-Gradient-1048k" \
  --devices cuda:0 cuda:1 \
  --datalens 60k 122k 244k 488k
```

---

## 输出

1. **ASCII 表格**：输出到 stdout，格式对齐 Table 4。
2. **CSV 文件**：保存到 `results/table4_YYYYMMDD_HHMMSS.csv`，包含列：`method, context_length, batch_size, throughput_tok_s`。

---

## 硬件要求

| GPU | 可测试范围 |
|-----|-----------|
| RTX 3090 (24GB) | Full KV: 16k×bsz≤3; ShadowKV: 16k×bsz≤3, 48k×bsz≤3 |
| A100 (80GB) | 完整 Table 4 |

---

## RULER 数据准备

Worker 脚本使用 `ruler/niah_single_1` 数据集。若缺失某个长度的数据，需手动生成：

```bash
cd data/ruler
python prepare.py \
  --save_dir data/llama-3/<TARGET_LEN> \
  --benchmark synthetic --task niah_single_1 \
  --tokenizer_path /home/zijie/models/Llama-3.1-8B-Instruct \
  --tokenizer_type hf \
  --max_seq_length <TARGET_LEN> \
  --num_samples 100 \
  --model_template_type llama-3
```

其中 `<TARGET_LEN>` 必须为：4096, 8192, 16384, 32768, 65536, 131072, 262144 之一。

---

## Throughput 计算方式

吞吐量在 `models/base.py` 的 `batch_generate()` 中计算：

```
throughput = batch_size × gen_tokens / elapsed_time
```

仅计时 decode 阶段（prefill 不包含在内），与论文一致。
