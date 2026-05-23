# RULER Benchmark 测试指南

> [!IMPORTANT]
> **必须使用 `scripts/run_ruler.sh` 启动 RULER 测试。** 禁止直接调用 `test/eval_acc.py` 运行 RULER，以确保数据自动生成、多 GPU 并行调度和结果归档的一致性。

---

## 快速开始

```bash
# 直接运行（使用脚本内部配置的任务列表和长度列表）
bash scripts/run_ruler.sh --gpus 0,1

# 指定模型和方法
bash scripts/run_ruler.sh \
  --model /home/zijie/models/Llama-3.1-8B-Instruct \
  --gpus 0,1 --method full --num_samples 10
```

---

## 脚本内部配置

测试的 **上下文长度** 和 **任务列表** 在脚本头部以 bash 数组形式配置，通过注释/取消注释控制：

```bash
# Context lengths to test (tokens)
DATALENS=(
    4096
    # 8192
    # 65536
    # 131072
)

# RULER tasks to evaluate
TASKS=(
    "niah_single_1"
    # "niah_single_2"
    "vt"
    "fwe"
    "qa_1"
    # "qa_2"
)
```

> [!TIP]
> 需要测试新的长度或任务？直接编辑 `scripts/run_ruler.sh` 头部的数组，取消注释即可。

---

## 命令行参数

以下参数从命令行传入，覆盖默认值：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型路径（HuggingFace 格式） | `/home/zijie/models/Llama-3.1-8B-Instruct` |
| `--models` | 多模型路径，逗号分隔 | 脚本内 `MODELS` 数组 |
| `--datalens` | 上下文长度，逗号分隔 | 脚本内 `DATALENS` 数组 |
| `--tasks` | RULER 任务名，逗号分隔（不带 `ruler/` 前缀） | 脚本内 `TASKS` 数组 |
| `--gpus` | 使用的 GPU 编号，逗号分隔 | `0,1` |
| `--method` | 注意力方式：`full` / `shadowkv` | `full` |
| `--num_samples` | 每任务测试样本数 | `10` |
| `--sparse_budget` | ShadowKV 稀疏预算 | `2048` |
| `--rank` | SVD 低秩分解秩 | `160` |
| `--chunk_size` | 分块大小 | `8` |
| `--data_samples` | 自动生成数据时的样本数 | `500` |

---

## 可用任务

来自 `data/ruler/synthetic.yaml`，共 11 个：

| 任务 | 类别 | 说明 |
|------|------|------|
| `niah_single_1` | NIAH | 单 needle，repeat haystack |
| `niah_single_2` | NIAH | 单 needle，essay haystack |
| `niah_single_3` | NIAH | 单 needle，UUID value |
| `niah_multikey_1` | NIAH | 多 key (4)，essay haystack |
| `niah_multikey_2` | NIAH | 多 key，needle haystack |
| `niah_multivalue` | NIAH | 多 value (4) |
| `niah_multiquery` | NIAH | 多 query (4) |
| `vt` | Variable Tracking | 变量追踪 |
| `fwe` | Freq Words Extraction | 高频词提取 |
| `qa_1` | QA | SQuAD 问答 |
| `qa_2` | QA | HotpotQA 问答 |

---

## 支持的模型

脚本根据路径关键字自动匹配模型模板：

| 关键字 | 模板 | 示例 |
|--------|------|------|
| `llama-3` / `llama3` | `llama-3` | `Llama-3.1-8B-Instruct` |
| `llama-2` / `llama2` | `llama-2` | `Llama-2-7b-chat-hf` |
| `yi` | `yi` | `Yi-9B-200K` |
| `glm` | `glm` | `GLM-4-9B-Chat-1M` |
| `qwen` | `qwen` | `Qwen2.5-7B-Instruct-1M` |
| `phi` | `phi` | `Phi-3-mini-128k-instruct` |

---

## 工作流程

脚本对每个 datalen 自动执行 3 步：

1. **数据检查与生成** — 检查 `data/ruler/data/{template}/{datalen}/{task}/validation.jsonl` 是否存在，缺失则调用 `prepare.py` 生成。
2. **Round-Robin 任务分配** — 将任务均匀分配到各 GPU，每 GPU 独立启动 `eval_acc.py`。
3. **结果聚合** — 等待所有 GPU 完成后，从 JSONL 文件汇总精度表格，按 datalen 分组输出。

---

## 输出位置

| 内容 | 路径 |
|------|------|
| 推理结果（JSONL） | `archive/{model}/ruler/{task}_{datalen}_{method}_{budget}_{rank}_{chunk}.jsonl` |
| GPU 日志 | `archive/{model}/logs/gpu{id}_{datalen}_{timestamp}.log` |

每条 JSONL 包含：
```json
{
  "prediction": ["模型输出"],
  "ground_truth": [["正确答案"]],
  "correct": [1.0, 0.0, ...],
  "avg_score": 0.85
}
```

---

## 常见用法

```bash
# 使用默认配置（编辑脚本头部的 DATALENS 和 TASKS 数组）
bash scripts/run_ruler.sh --gpus 0,1,2,3

# ShadowKV 测试
bash scripts/run_ruler.sh --method shadowkv --sparse_budget 2048 --gpus 0,1

# 快速冒烟测试（编辑脚本只保留一个 datalen 和少量 tasks）
bash scripts/run_ruler.sh --gpus 0 --num_samples 3

# GPU0 单模型/单任务/单长度冒烟测试（无需编辑脚本）
CUDA_VISIBLE_DEVICES=0 bash scripts/run_ruler.sh \
  --model /home/zijie/models/Llama-3.1-8B-Instruct \
  --gpus 0 --method full \
  --datalens 65536 --tasks niah_single_1 --num_samples 1
```

> [!NOTE]
> 部分 essay 类型任务（如 `niah_single_2`）的数据生成可能依赖外部语料，若生成失败脚本会自动跳过。
