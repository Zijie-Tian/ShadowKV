# ShadowKV - Codex 项目指南

## 项目概述

ShadowKV 是一个无需训练的高吞吐量长上下文 LLM 推理框架。核心创新是将海量 KV cache 存储在精确裁剪的"影子"（CPU 内存/低秩表示）中，而非保存在有限的 GPU VRAM 中。

**核心工作流程**:
1. **Prefill 阶段**: 对 K cache 进行 SVD 分解，用 landmarks 隔离异常 chunk，V cache offload 到 CPU
2. **Decode 阶段**: 查询与 K-landmarks 计算注意力获取相关 chunk，从 CPU 拉取 V，在线重建 K（U × SV + RoPE）

## 关键文件

| 文件 | 作用 |
|------|------|
| `models/kv_cache.py` | KV_Cache、ShadowKVCache 实现 |
| `models/base.py` | LLM 基类，管理 inference()、prefill()、generate() |
| `models/tensor_op.py` | CUDA kernel 绑定（RoPE、batch_gather_gemm） |
| `kernels/batch_gather_gemm.cu` | CUTLASS GEMM kernel，重建 K cache |
| `kernels/gather_copy.cu` | CPU→GPU V cache 高效传输 |

## 重要规则（详见 GEMINI.md）

**所有项目规则见 [`GEMINI.md`](GEMINI.md)**，核心要点：

- **环境**: 必须使用 `shadowkv` conda 环境，用 `conda run -n shadowkv ...` 或 `conda activate shadowkv`
- **依赖**: 仅在 shadowkv 环境中使用 `pip install`，禁止 `pip install -e .`，使用 PYTHONPATH 导入
- **内核修改**: 修改 `kernels/` 后必须 `python setup.py build_ext --inplace`
- **精度测试**: 必须用 `bash scripts/run_ruler.sh`，禁止直接调用 `test/eval_acc.py`
- **调试**: CUDA 错误用 `CUDA_LAUNCH_BLOCKING=1`
- **内存管理**: `gc.collect()`、`torch.cuda.empty_cache()` 不可随意移除

## 文档结构

- `GEMINI.md` - 规则和知识库入口（规则必须写在此文件）
- `docs/*.md` - 详细技术文档（snake_case 命名）

## 新增模型

参考 `llama.py` 或 `qwen.py` 继承/实现 `LLM` 类的方式。

## LongBench 测试（`scripts/run_longbench.sh`）

LongBench 必须通过 `scripts/run_longbench.sh` 启动，不要直接调用 `test/eval_acc.py`，以保证输出目录、预测格式和自动打分流程一致。

### 运行环境

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate shadowkv
```

GPU 调试默认只用 GPU0；命令中同时写 `CUDA_VISIBLE_DEVICES=0` 和 `--gpu 0`，避免占用其他 GPU。

### 流程与输出

`run_longbench.sh` 的流程参考 `~/Code/LUTAttn/scripts/run_exp.sh`：

1. 跳过校准阶段（ShadowKV 当前 LongBench 路径无需校准）。
2. 生成预测到模型专属目录。
3. 自动调用 `score_longbench.py` 打分并写出 `result.json`。

默认输出目录：

```text
archive/{model_basename}/long_bench/
```

例如本机默认模型会写到：

```text
archive/Llama-3.1-8B-Instruct/long_bench/<task>.jsonl
archive/Llama-3.1-8B-Instruct/long_bench/result.json
archive/Llama-3.1-8B-Instruct/long_bench/logs/score_*.log
```

每个 `<task>.jsonl` 使用与 LUTAttn LongBench scorer 兼容的逐样本格式：

```json
{"pred": "...", "answers": ["..."], "all_classes": [], "length": 3141, "score": 0.0}
```

### 常用参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model` | HuggingFace 模型路径或名称 | `/home/zijie/models/Llama-3.1-8B-Instruct` |
| `--gpu` / `--gpus` | 单 GPU 编号 | `0` |
| `--method` | `full` / `shadowkv` | `shadowkv` |
| `--tasks` | 逗号分隔 LongBench task，或 `all` | `all` |
| `--num_samples` | 每个 subtask 的样本数；`-1` 表示全部样本 | `2` |
| `--datalen` | prompt 中间截断后的最大 token 数 | `32768` |
| `--sparse_budget` | ShadowKV sparse budget | `512` |
| `--rank` | ShadowKV SVD rank | `160` |
| `--chunk_size` | ShadowKV chunk size | `8` |
| `--max_gen` | 覆盖每任务生成长度，仅用于 smoke/debug | 不覆盖 |
| `--output_dir` / `--pred_dir` | 覆盖预测与评分输出目录 | `archive/{model}/long_bench` |
| `--skip_score` | 只生成预测，不自动打分 | 关闭 |
| `--e` / `--longbench_e` | 使用 LongBench-E split | 关闭 |

### GPU0 全 subtask smoke

用于快速确认 21 个 subtask、预测输出和自动打分流程都能跑通：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_longbench.sh \
  --gpu 0 \
  --model /home/zijie/models/Llama-3.1-8B-Instruct \
  --method shadowkv \
  --tasks all \
  --num_samples 1 \
  --datalen 4096 \
  --sparse_budget 128 \
  --rank 160 \
  --chunk_size 8 \
  --max_gen 1
```

`--max_gen 1` 只用于 smoke，不能作为正式 LongBench 分数。

### GPU0 全量评测

正式跑全部 subtask 的全部样本时去掉 `--max_gen`，并使用任务默认生成长度：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_longbench.sh \
  --gpu 0 \
  --model /home/zijie/models/Llama-3.1-8B-Instruct \
  --method shadowkv \
  --tasks all \
  --num_samples -1 \
  --datalen 32768 \
  --sparse_budget 512 \
  --rank 160 \
  --chunk_size 8
```

### 单独重新打分

如果预测已经存在，只想重新计算分数：

```bash
python score_longbench.py \
  --pred_dir archive/Llama-3.1-8B-Instruct/long_bench \
  --datasets all
```
