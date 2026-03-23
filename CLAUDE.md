# ShadowKV - Claude Code 项目指南

## 项目概述

ShadowKV 是一个无需训练的高吞吐量长上下文 LLM 推理框架。核心创新是将海量 KV cache 存储在精确裁剪的"影子"（CPU 内存/低秩表示）中，而非保存在有限的 GPU VRAM 中。

**核心工作流程**:
1. **Prefill 阶段**: 对 K cache 进行 SVD 分解，用 landmarks 隔离异常 chunk，V cache offload 到 CPU
2. **Decode 阶段**: 查询与 K-landmarks 计算注意力获取相关 chunk，从 CPU 拉取 V，在线重建 K（U × SV + RoPE）

## 关键文件

| 文件 | 作用 |
|------|------|
| `models/kv_cache.py` | KV_Cache、ShadowKVCache、ShadowKVCache_CPU 实现 |
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
