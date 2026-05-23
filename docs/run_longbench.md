# LongBench 测试指南

本文档说明如何在 ShadowKV 中运行 LongBench accuracy smoke / debug。实现参考了 `~/Code/Quest/evaluation/LongBench` 的 prompt、max generation、metric 配置，并接入当前仓库的 `test/eval_acc.py` / `Evaluator` / `Dataset` 流程。

> [!IMPORTANT]
> 本仓库测试命令必须在 `shadowkv` conda 环境中执行。单卡 GPU 调试时请显式限制 `CUDA_VISIBLE_DEVICES=0`，避免占用其他 GPU。

## 快速开始：GPU0 两样本调试

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate shadowkv

CUDA_VISIBLE_DEVICES=0 bash scripts/run_longbench.sh \
  --gpu 0 \
  --model /home/zijie/models/Llama-3.1-8B-Instruct \
  --method shadowkv \
  --num_samples 2 \
  --datalen 32768 \
  --sparse_budget 512 \
  --rank 160 \
  --chunk_size 8 \
  --max_gen 8
```

`--max_gen 8` 适合 smoke/debug，会覆盖 LongBench 官方每任务生成长度以缩短运行时间；正式评测时去掉该参数，使用 `data/longbench/config/dataset2maxlen.json` 中的任务默认值。

## 常用参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model` | HuggingFace 模型路径或名称 | `/home/zijie/models/Llama-3.1-8B-Instruct` |
| `--gpu` / `--gpus` | 单 GPU 编号、逗号分隔多 GPU 编号，或 `all` | `0` |
| `--method` | `full` / `shadowkv` | `shadowkv` |
| `--tasks` | 逗号分隔任务名，或 `all` | `all` |
| `--num_samples` | 每个 LongBench subtask 取样数量 | `2` |
| `--datalen` | prompt 中间截断后的最大 token 数 | `32768` |
| `--sparse_budget` | ShadowKV sparse budget | `512` |
| `--rank` | ShadowKV SVD rank | `160` |
| `--chunk_size` | ShadowKV chunk size | `8` |
| `--max_gen` | 覆盖每任务生成长度，调试时建议小值 | 不覆盖 |
| `--e` | 使用 LongBench-E (`*_e`) split | 关闭 |

## 任务列表

任务来自 `data/longbench/config/dataset2prompt.json`，包括：

- `narrativeqa`, `qasper`, `multifieldqa_en`, `multifieldqa_zh`
- `hotpotqa`, `2wikimqa`, `musique`, `dureader`
- `gov_report`, `qmsum`, `multi_news`, `vcsum`
- `trec`, `triviaqa`, `samsum`, `lsht`
- `passage_count`, `passage_retrieval_en`, `passage_retrieval_zh`
- `lcc`, `repobench-p`

只跑部分任务示例：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_longbench.sh \
  --gpu 0 --method full \
  --tasks qasper,narrativeqa,hotpotqa \
  --num_samples 2 --max_gen 8
```

## 多 GPU task 分片

`--gpus` 传入多个 GPU（例如 `0,1,2,3,4,5`）或 `all` 时，`scripts/run_longbench.sh`
会在脚本内部按 task round-robin 分片，每张 GPU 启动一个单卡子进程，所有子进程写入同一个输出目录中互不重叠的
`<task>.jsonl`，最后由父进程统一调用 `score_longbench.py` 生成 `result.json`。

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate shadowkv

SHADOWKV_LONGBENCH_DATASET=/home/zijie/data/LongBench \
bash scripts/run_longbench.sh \
  --gpus all \
  --model /home/zijie/models/Llama-3.2-1B-Instruct \
  --method shadowkv \
  --tasks all \
  --num_samples 1 \
  --datalen 4096 \
  --sparse_budget 128 \
  --rank 160 \
  --chunk_size 8 \
  --max_gen 1
```

多 GPU 输出目录包含：

```text
logs/task_shards.tsv
logs/gpu0.log
logs/gpu1.log
...
logs/score_*.log
result.json
```

`SHADOWKV_LONGBENCH_DATASET` 可用于指定本机 LongBench 数据集路径；未设置时默认仍使用 `THUDM/LongBench`。

## 输出位置

`test/eval_acc.py` 会按 dataset name 写入：

```text
archive/{model}/long_bench/{task}.jsonl
archive/{model}/long_bench/result.json
```

每个子任务写入一个 `<task>.jsonl`，每条样本 JSONL 使用与 LUTAttn LongBench scorer 兼容的字段：`pred`、`answers`、`all_classes`、`length`、`score`。脚本结束后会自动调用 `score_longbench.py`，在同一目录生成 `result.json`。


## 打分流程

`scripts/run_longbench.sh` 参考 LUTAttn `scripts/run_exp.sh` 的流程：

1. 跳过校准阶段（ShadowKV 当前 LongBench 路径无需校准）。
2. 生成预测，输出到 `archive/{model}/long_bench/<task>.jsonl`。
3. 自动执行：

```bash
python score_longbench.py --pred_dir archive/Llama-3.1-8B-Instruct/long_bench --datasets all
```

`result.json` 会保存在同一个模型目录下。若只想生成预测不打分，可传 `--skip_score`。如需覆盖输出目录，可传 `--output_dir` / `--pred_dir`。

## 实现入口

- `data/dataset.py`：新增 `long_bench/<task>` loader，加载 `THUDM/LongBench`，套用 Quest prompt config，按 `--datalen` 做中间截断。
- `data/longbench/config/`：从 Quest 参考复制的 `dataset2prompt.json` 和 `dataset2maxlen.json`。
- `data/longbench/metrics.py`：从 Quest 参考复制 LongBench metrics，并保留 dataset-to-metric 映射。
- `test/evaluator.py`：新增 LongBench 逐样本 scoring/output 字段。
- `scripts/run_longbench.sh`：GPU-scoped wrapper，便于每个 subtask 两样本 debug。

## 注意事项

- 这一路径直接使用 ShadowKV 自己的 `LLM.generate()`，不会复刻 Quest 中“context prefill + question token decode”的所有内部细节；主要复用 Quest 的 prompt/max-gen/metric 配置。
- 对 `shadowkv` 方法，较短的 LongBench 样本在去掉 local/outlier chunk 后，可检索 landmark 数可能小于 `sparse_budget / chunk_size`；`ShadowKVCache` 会按当前样本的可用 landmark 数动态收缩 active sparse budget，避免 `torch.topk` 越界。
- `--max_gen` 会影响分数，只能用于快速 debug；正式结果不要使用小 `--max_gen`。
