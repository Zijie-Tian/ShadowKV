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
| `--gpu` / `--gpus` | 单 GPU 编号、逗号分隔多 GPU 编号，或 `all` | `0` |
| `--pp_size` / `--pp-size` | 每个 LongBench worker 使用的 GPU 数；`1` 为普通单卡，`2` 为当前 ShadowKV 多卡测试推荐 PP 配置 | 脚本默认 `1`；ShadowKV 多卡测试显式写 `2` |
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

### ShadowKV PP 默认策略与支持矩阵

ShadowKV LongBench 测试在有至少 2 张空闲 GPU、且模型已适配 PP 时，**默认推荐显式使用 `--pp_size 2`**。
脚本为了兼容旧单卡流程，CLI 默认值仍是 `1`，因此多卡 ShadowKV 命令必须主动写出：

```bash
--method shadowkv --pp_size 2
```

6 张 GPU 的标准编排是 3 个 2 卡 PP worker：

```text
worker 0: GPU 0,1
worker 1: GPU 2,3
worker 2: GPU 4,5
```

使用规则：

1. **ShadowKV 多卡测试默认写 `--pp_size 2`**，尤其是 7B/8B/9B 模型、`datalen=32768`、6 GPU 并行跑 LongBench 时。
2. 只有 1 张 GPU、做单卡回归、或模型不在下表支持范围内时，才使用 `--pp_size 1` 或省略。
3. `--pp_size 2` 表示每个 worker 内部用 2 张可见 GPU 做 layer-sharded PP；如果传 `--gpus 0,1,2,3,4,5`，脚本会自动切成 `0,1`、`2,3`、`4,5` 三组并行。
4. 不要在 6 GPU PP 命令前额外设置 `CUDA_VISIBLE_DEVICES=0`；由 `scripts/run_longbench.sh --gpus ... --pp_size 2` 统一编排。
5. PP 当前主要验证的是 LongBench `--method shadowkv` accuracy/debug 路径；`full + PP` 不是默认测试路径。

当前 PP 支持情况：

| 模型类 / family | 常用本地模型路径 | 推荐 `pp_size` | 状态 |
|---|---|---:|---|
| `GLM` / GLM-4 | `/home/zijie/models/GLM-4-9B-Chat-1M` | `2` | 已适配；用于避免 24GB 单卡 32k ShadowKV LongBench OOM；已通过 6 GPU / 3 worker / 32k smoke |
| `Llama` / Llama 3.1 | `/home/zijie/models/Llama-3.1-8B-Instruct` | `2` | 已适配；已通过 2 GPU 32k smoke 和 6 GPU / 3 worker / 32k smoke |
| `Llama` / Llama 2 | `/home/zijie/models/Llama-2-7b-chat-hf` | `2` | 已适配；已通过 2 GPU 32k smoke 和 6 GPU / 3 worker / 32k smoke |
| `Llama` / 小模型回归 | `/home/zijie/models/Llama-3.2-1B-Instruct` | `1` 或 `2` | Llama 类已支持 PP；小模型通常单卡即可，已做 `pp_size=1` 回归 |
| `Qwen` / `Phi` / 其他模型类 | 视本地模型而定 | `1` | 未适配 PP；`test/eval_acc.py` 会拒绝 `--pp_size != 1`，避免误用 |

### GLM-4-9B-Chat-1M：6 GPU + 2 卡 PP LongBench

GLM-4-9B-Chat-1M 在 24GB 单卡上跑 `datalen=32768` 的 ShadowKV LongBench 容易 OOM。当前推荐用
`--pp_size 2` 做 **2 卡一组的 layer-sharded PP**，再把 6 张 GPU 编排成 3 个并行 worker：

```text
worker 0: GPU 0,1
worker 1: GPU 2,3
worker 2: GPU 4,5
```

配置要点：

1. **必须通过 `scripts/run_longbench.sh` 启动**，不要直接调用 `test/eval_acc.py`。
2. **不要在命令前额外设置 `CUDA_VISIBLE_DEVICES=0`**。6 卡 PP 由 `--gpus 0,1,2,3,4,5 --pp_size 2`
   统一编排；脚本会在每个子 worker 内部设置正确的 `CUDA_VISIBLE_DEVICES=<两张卡>`。
3. `--pp_size 2` 当前支持 GLM 与 Llama 的 LongBench/ShadowKV 路径；Qwen/Phi 等未适配模型常规评测保持 `--pp_size 1` 或省略。
4. `SHADOWKV_LONGBENCH_DATASET=/home/zijie/data/LongBench` 用本机 LongBench 数据集，避免在线下载和 remote-code 差异。
5. `--max_gen 1` 只用于 smoke/debug，正式分数必须去掉。
6. 正式全量用 `--num_samples -1 --datalen 32768 --sparse_budget 512 --rank 160 --chunk_size 8`。
7. 为了避免覆盖失败的旧结果，GLM PP 推荐输出到独立目录，例如
   `archive/GLM-4-9B-Chat-1M/long_bench_shadowkv_pp`。

#### 6 GPU / 2 卡 PP / 32k smoke

先用每个 subtask 1 条样本、每题只生成 1 token 验证所有 21 个 LongBench subtask、3 个 PP worker、
预测输出和自动打分流程都能跑通：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate shadowkv

SHADOWKV_LONGBENCH_DATASET=/home/zijie/data/LongBench \
bash scripts/run_longbench.sh \
  --gpus 0,1,2,3,4,5 \
  --pp_size 2 \
  --model /home/zijie/models/GLM-4-9B-Chat-1M \
  --method shadowkv \
  --tasks all \
  --num_samples 1 \
  --datalen 32768 \
  --sparse_budget 512 \
  --rank 160 \
  --chunk_size 8 \
  --max_gen 1 \
  --output_dir archive/GLM-4-9B-Chat-1M/long_bench_shadowkv_pp_smoke_32k
```

smoke 成功后应看到：

```text
archive/GLM-4-9B-Chat-1M/long_bench_shadowkv_pp_smoke_32k/<task>.jsonl
archive/GLM-4-9B-Chat-1M/long_bench_shadowkv_pp_smoke_32k/result.json
archive/GLM-4-9B-Chat-1M/long_bench_shadowkv_pp_smoke_32k/logs/task_shards.tsv
archive/GLM-4-9B-Chat-1M/long_bench_shadowkv_pp_smoke_32k/logs/gpu0_1.log
archive/GLM-4-9B-Chat-1M/long_bench_shadowkv_pp_smoke_32k/logs/gpu2_3.log
archive/GLM-4-9B-Chat-1M/long_bench_shadowkv_pp_smoke_32k/logs/gpu4_5.log
archive/GLM-4-9B-Chat-1M/long_bench_shadowkv_pp_smoke_32k/logs/score_*.log
```

`logs/task_shards.tsv` 会记录每个 GPU pair 负责哪些 subtask；每个 `<task>.jsonl` 在 smoke 下应有 1 行。

#### 6 GPU / 2 卡 PP / GLM ShadowKV 全量 LongBench

smoke 通过后跑正式评测。正式命令不要加 `--max_gen`：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate shadowkv

SHADOWKV_LONGBENCH_DATASET=/home/zijie/data/LongBench \
bash scripts/run_longbench.sh \
  --gpus 0,1,2,3,4,5 \
  --pp_size 2 \
  --model /home/zijie/models/GLM-4-9B-Chat-1M \
  --method shadowkv \
  --tasks all \
  --num_samples -1 \
  --datalen 32768 \
  --sparse_budget 512 \
  --rank 160 \
  --chunk_size 8 \
  --output_dir archive/GLM-4-9B-Chat-1M/long_bench_shadowkv_pp
```

全量输出目录：

```text
archive/GLM-4-9B-Chat-1M/long_bench_shadowkv_pp/<task>.jsonl
archive/GLM-4-9B-Chat-1M/long_bench_shadowkv_pp/result.json
archive/GLM-4-9B-Chat-1M/long_bench_shadowkv_pp/logs/task_shards.tsv
archive/GLM-4-9B-Chat-1M/long_bench_shadowkv_pp/logs/gpu0_1.log
archive/GLM-4-9B-Chat-1M/long_bench_shadowkv_pp/logs/gpu2_3.log
archive/GLM-4-9B-Chat-1M/long_bench_shadowkv_pp/logs/gpu4_5.log
archive/GLM-4-9B-Chat-1M/long_bench_shadowkv_pp/logs/score_*.log
```

脚本会在 3 个 PP worker 全部结束后自动调用：

```bash
python score_longbench.py \
  --pred_dir archive/GLM-4-9B-Chat-1M/long_bench_shadowkv_pp \
  --datasets all
```

`score_longbench.py` 生成的 `result.json` 会按 subtask 名字排序，并把 `mean` 放在最后。

#### GLM/Llama PP 常见问题排查

- 如果某个 worker OOM 或异常退出，先看对应 pair 的日志，例如：

  ```bash
  tail -n 120 archive/GLM-4-9B-Chat-1M/long_bench_shadowkv_pp/logs/gpu0_1.log
  tail -n 120 archive/GLM-4-9B-Chat-1M/long_bench_shadowkv_pp/logs/gpu2_3.log
  tail -n 120 archive/GLM-4-9B-Chat-1M/long_bench_shadowkv_pp/logs/gpu4_5.log
  ```

- 如果需要确认 GPU 是否被旧进程占用：

  ```bash
  nvidia-smi
  ps -o pid,ppid,stat,etime,cmd -u "$USER" | grep -E 'GLM-4-9B-Chat-1M|eval_acc.py|run_longbench.sh' | grep -v grep
  ```

- 如果只想重新打分，不重新生成预测：

  ```bash
  python score_longbench.py \
    --pred_dir archive/GLM-4-9B-Chat-1M/long_bench_shadowkv_pp \
    --datasets all
  ```

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
