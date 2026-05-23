#!/bin/bash
################################################################################
# GPU-scoped LongBench runner for ShadowKV.
#
# Examples:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_longbench.sh --gpu 0 --method shadowkv --num_samples 2 --max_gen 8
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_longbench.sh --gpu 0 --tasks qasper,narrativeqa --method full
#   bash scripts/run_longbench.sh --gpus 0,1,2,3,4,5 --tasks all --num_samples 1 --max_gen 1
#   bash scripts/run_longbench.sh --gpus all --tasks all --num_samples 1 --max_gen 1
################################################################################

set -euo pipefail

MODEL="/home/zijie/models/Llama-3.1-8B-Instruct"
GPU="0"
METHOD="shadowkv"
NUM_SAMPLES=2
DATALEN=32768
SPARSE_BUDGET=512
RANK=160
CHUNK_SIZE=8
MAX_GEN=""
LONG_BENCH_E=false
TASKS="all"
OUTPUT_DIR=""
SKIP_SCORE=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

ALL_TASKS=$(python - <<'PY'
import json
from pathlib import Path
p = Path('data/longbench/config/dataset2prompt.json')
print(','.join(json.load(open(p, encoding='utf-8')).keys()))
PY
)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)         MODEL="$2";          shift 2 ;;
        --gpu|--gpus)    GPU="$2";            shift 2 ;;
        --method)        METHOD="$2";         shift 2 ;;
        --tasks)         TASKS="$2";          shift 2 ;;
        --output_dir|--pred_dir)
                         OUTPUT_DIR="$2";     shift 2 ;;
        --skip_score)    SKIP_SCORE=true;      shift ;;
        --num_samples)   NUM_SAMPLES="$2";    shift 2 ;;
        --datalen)       DATALEN="$2";        shift 2 ;;
        --sparse_budget) SPARSE_BUDGET="$2";  shift 2 ;;
        --rank)          RANK="$2";           shift 2 ;;
        --chunk_size)    CHUNK_SIZE="$2";     shift 2 ;;
        --max_gen|--longbench_max_gen)
                         MAX_GEN="$2";        shift 2 ;;
        --e|--longbench_e)
                         LONG_BENCH_E=true;   shift ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ "${TASKS}" == "all" ]]; then
    TASKS="${ALL_TASKS}"
fi

IFS=',' read -ra TASK_ARRAY <<< "${TASKS}"
MODEL_SHORT="$(basename "${MODEL%/}")"
if [[ -z "${OUTPUT_DIR}" ]]; then
    if [[ "${LONG_BENCH_E}" == true ]]; then
        OUTPUT_DIR="archive/${MODEL_SHORT}/long_bench_e"
    else
        OUTPUT_DIR="archive/${MODEL_SHORT}/long_bench"
    fi
fi
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

resolve_gpu_list() {
    local requested="$1"
    if [[ "${requested}" == "all" ]]; then
        if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
            echo "${CUDA_VISIBLE_DEVICES}"
        elif command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi --query-gpu=index --format=csv,noheader,nounits | paste -sd, -
        else
            echo "0"
        fi
    else
        echo "${requested}"
    fi
}

GPU_LIST="$(resolve_gpu_list "${GPU}")"
IFS=',' read -ra RAW_GPU_ARRAY <<< "${GPU_LIST}"
GPU_ARRAY=()
for GPU_ID in "${RAW_GPU_ARRAY[@]}"; do
    GPU_ID="$(echo "${GPU_ID}" | xargs)"
    [[ -z "${GPU_ID}" ]] && continue
    GPU_ARRAY+=("${GPU_ID}")
done
if [[ "${#GPU_ARRAY[@]}" -eq 0 ]]; then
    echo "No GPUs resolved from --gpu/--gpus '${GPU}'" >&2
    exit 1
fi

if [[ "${#GPU_ARRAY[@]}" -gt 1 ]]; then
    SHARD_TASKS=()
    for IDX in "${!GPU_ARRAY[@]}"; do
        SHARD_TASKS[IDX]=""
    done

    TASK_IDX=0
    for TASK in "${TASK_ARRAY[@]}"; do
        [[ -z "${TASK}" ]] && continue
        SHARD_IDX=$((TASK_IDX % ${#GPU_ARRAY[@]}))
        if [[ -z "${SHARD_TASKS[SHARD_IDX]}" ]]; then
            SHARD_TASKS[SHARD_IDX]="${TASK}"
        else
            SHARD_TASKS[SHARD_IDX]="${SHARD_TASKS[SHARD_IDX]},${TASK}"
        fi
        TASK_IDX=$((TASK_IDX + 1))
    done

    SHARD_FILE="${LOG_DIR}/task_shards.tsv"
    : > "${SHARD_FILE}"

    cat <<EOF
╔══════════════════════════════════════════════════════════════╗
║          ShadowKV LongBench Multi-GPU Runner                ║
╠══════════════════════════════════════════════════════════════╣
║  Model:    ${MODEL}
║  GPUs:     ${GPU_ARRAY[*]}
║  Method:   ${METHOD}
║  Samples:  ${NUM_SAMPLES}
║  DataLen:  ${DATALEN}
║  Budget:   ${SPARSE_BUDGET}
║  Rank:     ${RANK}
║  ChunkSz:  ${CHUNK_SIZE}
║  MaxGen:   ${MAX_GEN:-task defaults}
║  Tasks:    ${TASKS}
║  PredDir:  ${OUTPUT_DIR}
║  Score:    $([[ "${SKIP_SCORE}" == true ]] && echo "skip" || echo "enabled")
╚══════════════════════════════════════════════════════════════╝
EOF

    echo "[1/3] 校准阶段已跳过（ShadowKV 无需校准）"
    echo "[2/3] 开始 LongBench 多 GPU 预测生成..."

    PIDS=()
    PID_LABELS=()
    for IDX in "${!GPU_ARRAY[@]}"; do
        CHILD_GPU="${GPU_ARRAY[IDX]}"
        CHILD_TASKS="${SHARD_TASKS[IDX]}"
        [[ -z "${CHILD_TASKS}" ]] && continue

        SAFE_GPU="$(printf '%s' "${CHILD_GPU}" | tr -c 'A-Za-z0-9_.-' '_')"
        CHILD_LOG="${LOG_DIR}/gpu${SAFE_GPU}.log"
        printf "%s\t%s\n" "${CHILD_GPU}" "${CHILD_TASKS}" >> "${SHARD_FILE}"
        echo "[launch] GPU ${CHILD_GPU}: ${CHILD_TASKS} -> ${CHILD_LOG}"

        CHILD_CMD=(bash "${SCRIPT_DIR}/run_longbench.sh"
            --gpu "${CHILD_GPU}"
            --model "${MODEL}"
            --method "${METHOD}"
            --tasks "${CHILD_TASKS}"
            --num_samples "${NUM_SAMPLES}"
            --datalen "${DATALEN}"
            --sparse_budget "${SPARSE_BUDGET}"
            --rank "${RANK}"
            --chunk_size "${CHUNK_SIZE}"
            --output_dir "${OUTPUT_DIR}"
            --skip_score
        )
        if [[ -n "${MAX_GEN}" ]]; then
            CHILD_CMD+=(--max_gen "${MAX_GEN}")
        fi
        if [[ "${LONG_BENCH_E}" == true ]]; then
            CHILD_CMD+=(--e)
        fi

        (
            cd "${ROOT_DIR}"
            "${CHILD_CMD[@]}"
        ) > "${CHILD_LOG}" 2>&1 &
        PIDS+=("$!")
        PID_LABELS+=("GPU ${CHILD_GPU}")
    done

    printf "%s\n" "${PIDS[@]}" > "${LOG_DIR}/pids.txt"

    FAILED=0
    for IDX in "${!PIDS[@]}"; do
        PID="${PIDS[IDX]}"
        LABEL="${PID_LABELS[IDX]}"
        if wait "${PID}"; then
            echo "[ok] ${LABEL} completed"
        else
            echo "[fail] ${LABEL} failed (pid ${PID})" >&2
            FAILED=1
        fi
    done

    if [[ "${FAILED}" -ne 0 ]]; then
        echo "[summary] one or more prediction shards failed; tailing logs" >&2
        for LOG in "${LOG_DIR}"/gpu*.log; do
            echo "===== ${LOG} ====="
            tail -n 120 "${LOG}"
        done
        exit 1
    fi

    echo "[2/3] 预测完成"
    echo "结果目录: ${OUTPUT_DIR}"

    if [[ "${SKIP_SCORE}" == true ]]; then
        echo "[3/3] 跳过分数计算 (--skip_score)"
    else
        SCORE_LOG="${LOG_DIR}/score_$(date +%Y%m%d_%H%M%S).log"
        echo "[3/3] 开始计算分数..."
        SCORE_CMD=(python score_longbench.py --pred_dir "${OUTPUT_DIR}" --datasets "${TASKS}")
        if [[ "${LONG_BENCH_E}" == true ]]; then
            SCORE_CMD+=(--e)
        fi
        echo "评分脚本: ${SCORE_CMD[*]}"
        "${SCORE_CMD[@]}" > "${SCORE_LOG}" 2>&1
        cat "${SCORE_LOG}"
        if [[ -f "${OUTPUT_DIR}/result.json" ]]; then
            echo "✓ 分数计算完成"
            echo "结果文件: ${OUTPUT_DIR}/result.json"
            echo "评分日志: ${SCORE_LOG}"
        else
            echo "分数计算失败，详见 ${SCORE_LOG}" >&2
            exit 1
        fi
    fi
    exit 0
fi

RUN_GPU="${GPU_ARRAY[0]}"
DATASET_NAMES=""
for TASK in "${TASK_ARRAY[@]}"; do
    [[ -z "${TASK}" ]] && continue
    if [[ -z "${DATASET_NAMES}" ]]; then
        DATASET_NAMES="long_bench/${TASK}"
    else
        DATASET_NAMES="${DATASET_NAMES},long_bench/${TASK}"
    fi
done

CMD=(python test/eval_acc.py
    --model_name "${MODEL}"
    --dataset_name "${DATASET_NAMES}"
    --num_samples "${NUM_SAMPLES}"
    --datalen "${DATALEN}"
    --method "${METHOD}"
    --sparse_budget "${SPARSE_BUDGET}"
    --rank "${RANK}"
    --chunk_size "${CHUNK_SIZE}"
    --longbench_output_dir "${OUTPUT_DIR}"
)

if [[ -n "${MAX_GEN}" ]]; then
    CMD+=(--longbench_max_gen "${MAX_GEN}")
fi
if [[ "${LONG_BENCH_E}" == true ]]; then
    CMD+=(--longbench_e)
fi

cat <<EOF
╔══════════════════════════════════════════════════════════════╗
║              ShadowKV LongBench Runner                      ║
╠══════════════════════════════════════════════════════════════╣
║  Model:    ${MODEL}
║  GPU:      ${RUN_GPU}
║  Method:   ${METHOD}
║  Samples:  ${NUM_SAMPLES}
║  DataLen:  ${DATALEN}
║  Budget:   ${SPARSE_BUDGET}
║  Rank:     ${RANK}
║  ChunkSz:  ${CHUNK_SIZE}
║  MaxGen:   ${MAX_GEN:-task defaults}
║  Tasks:    ${TASKS}
║  PredDir:  ${OUTPUT_DIR}
║  Score:    $([[ "${SKIP_SCORE}" == true ]] && echo "skip" || echo "enabled")
╚══════════════════════════════════════════════════════════════╝
EOF

echo "[1/3] 校准阶段已跳过（ShadowKV 无需校准）"
echo "[2/3] 开始 LongBench 预测生成..."
CUDA_VISIBLE_DEVICES="${RUN_GPU}" PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" "${CMD[@]}"

echo "[2/3] 预测完成"
echo "结果目录: ${OUTPUT_DIR}"

if [[ "${SKIP_SCORE}" == true ]]; then
    echo "[3/3] 跳过分数计算 (--skip_score)"
else
    SCORE_LOG="${LOG_DIR}/score_$(date +%Y%m%d_%H%M%S).log"
    echo "[3/3] 开始计算分数..."
    SCORE_CMD=(python score_longbench.py --pred_dir "${OUTPUT_DIR}" --datasets "${TASKS}")
    if [[ "${LONG_BENCH_E}" == true ]]; then
        SCORE_CMD+=(--e)
    fi
    echo "评分脚本: ${SCORE_CMD[*]}"
    "${SCORE_CMD[@]}" > "${SCORE_LOG}" 2>&1
    cat "${SCORE_LOG}"
    if [[ -f "${OUTPUT_DIR}/result.json" ]]; then
        echo "✓ 分数计算完成"
        echo "结果文件: ${OUTPUT_DIR}/result.json"
        echo "评分日志: ${SCORE_LOG}"
    else
        echo "分数计算失败，详见 ${SCORE_LOG}" >&2
        exit 1
    fi
fi
