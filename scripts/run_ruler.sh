#!/bin/bash
################################################################################
#
# Multi-GPU RULER Benchmark Runner for ShadowKV
#
# Distributes RULER tasks across multiple GPUs in parallel (round-robin).
# Auto-generates missing RULER data, runs eval_acc.py per GPU, and aggregates
# results into a summary table.
#
# Usage:
#   bash scripts/run_ruler.sh \
#     --model /home/zijie/models/Llama-3.1-8B-Instruct \
#     --gpus 0,1 --method full --num_samples 10
#
################################################################################

set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# Configuration Lists — Edit these to control what gets tested
# ══════════════════════════════════════════════════════════════════════════════

# Context lengths to test (tokens)
DATALENS=(
    4096
    # 8192
    # 16384
    # 32768
    # 65536
    # 131072
    # 262144
)

# RULER tasks to evaluate
TASKS=(
    "niah_single_1"
    # "niah_single_2"
    # "niah_single_3"
    # "niah_multikey_1"
    # "niah_multikey_2"
    # "niah_multivalue"
    # "niah_multiquery"
    "vt"
    "fwe"
    "qa_1"
    # "qa_2"
)

# ══════════════════════════════════════════════════════════════════════════════
# Defaults (override via command-line arguments)
# ══════════════════════════════════════════════════════════════════════════════
MODEL="/home/zijie/models/Llama-3.1-8B-Instruct"
GPUS="0,1"
METHOD="full"
NUM_SAMPLES=10
SPARSE_BUDGET=2048
RANK=160
CHUNK_SIZE=8
DATA_SAMPLES=500  # number of samples to generate in RULER data

# ─── Parse Arguments ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)         MODEL="$2";          shift 2 ;;
        --gpus)          GPUS="$2";           shift 2 ;;
        --method)        METHOD="$2";         shift 2 ;;
        --num_samples)   NUM_SAMPLES="$2";    shift 2 ;;
        --sparse_budget) SPARSE_BUDGET="$2";  shift 2 ;;
        --rank)          RANK="$2";           shift 2 ;;
        --chunk_size)    CHUNK_SIZE="$2";     shift 2 ;;
        --data_samples)  DATA_SAMPLES="$2";   shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ─── Resolve paths ───────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# ─── Detect model template ──────────────────────────────────────────────────
MODEL_LOWER=$(echo "${MODEL}" | tr '[:upper:]' '[:lower:]')
if [[ "${MODEL_LOWER}" == *"llama-3"* ]] || [[ "${MODEL_LOWER}" == *"llama_3"* ]] || [[ "${MODEL_LOWER}" == *"llama3"* ]]; then
    TEMPLATE="llama-3"
elif [[ "${MODEL_LOWER}" == *"llama-2"* ]] || [[ "${MODEL_LOWER}" == *"llama_2"* ]] || [[ "${MODEL_LOWER}" == *"llama2"* ]]; then
    TEMPLATE="llama-2"
elif [[ "${MODEL_LOWER}" == *"yi"* ]]; then
    TEMPLATE="yi"
elif [[ "${MODEL_LOWER}" == *"glm"* ]]; then
    TEMPLATE="glm"
elif [[ "${MODEL_LOWER}" == *"qwen"* ]]; then
    TEMPLATE="qwen"
elif [[ "${MODEL_LOWER}" == *"phi"* ]]; then
    TEMPLATE="phi"
else
    echo "ERROR: Cannot auto-detect model template from '${MODEL}'"
    echo "Supported: llama-3, llama-2, yi, glm, qwen, phi"
    exit 1
fi

# ─── Derived variables ──────────────────────────────────────────────────────
IFS=',' read -ra GPU_ARRAY <<< "${GPUS}"
NUM_GPUS=${#GPU_ARRAY[@]}
NUM_TASKS=${#TASKS[@]}
NUM_LENS=${#DATALENS[@]}
MODEL_SHORT=$(basename "${MODEL}")
LOG_DIR="archive/${MODEL_SHORT}/logs"
mkdir -p "${LOG_DIR}"

# ─── Print configuration ────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              ShadowKV RULER Benchmark Runner                ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  Model:    %-48s║\n" "${MODEL_SHORT}"
printf "║  Template: %-48s║\n" "${TEMPLATE}"
printf "║  GPUs:     %-48s║\n" "${GPUS} (${NUM_GPUS} devices)"
printf "║  Tasks:    %-48s║\n" "${NUM_TASKS} tasks"
printf "║  DataLens: %-48s║\n" "${DATALENS[*]}"
printf "║  Method:   %-48s║\n" "${METHOD}"
printf "║  Samples:  %-48s║\n" "${NUM_SAMPLES}"
printf "║  Budget:   %-48s║\n" "${SPARSE_BUDGET}"
printf "║  Rank:     %-48s║\n" "${RANK}"
printf "║  ChunkSz:  %-48s║\n" "${CHUNK_SIZE}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Main loop: iterate over each datalen
# ═══════════════════════════════════════════════════════════════════════════
TOTAL_FAILURES=0

for DATALEN in "${DATALENS[@]}"; do

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  DataLen: ${DATALEN}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ─── Step 1: Auto-generate missing RULER data ────────────────────────────
echo "── [1/3] Checking & generating RULER data ──"
DATA_DIR="data/ruler/data/${TEMPLATE}/${DATALEN}"

for TASK in "${TASKS[@]}"; do
    JSONL="${DATA_DIR}/${TASK}/validation.jsonl"
    if [[ -f "${JSONL}" ]]; then
        LINES=$(wc -l < "${JSONL}")
        echo "  ✓ ${TASK} — exists (${LINES} samples)"
    else
        echo "  ✗ ${TASK} — generating (${DATA_SAMPLES} samples)..."
        mkdir -p "${DATA_DIR}"
        python data/ruler/prepare.py \
            --save_dir "${DATA_DIR}" \
            --benchmark synthetic \
            --task "${TASK}" \
            --tokenizer_path "${MODEL}" \
            --tokenizer_type hf \
            --max_seq_length "${DATALEN}" \
            --model_template_type "${TEMPLATE}" \
            --num_samples "${DATA_SAMPLES}" \
            2>&1 | sed 's/^/    /'
        if [[ -f "${JSONL}" ]]; then
            LINES=$(wc -l < "${JSONL}")
            echo "  ✓ ${TASK} — generated (${LINES} samples)"
        else
            echo "  ⚠ ${TASK} — generation failed, skipping"
        fi
    fi
done

# Rebuild task list with only available tasks
AVAIL_TASKS=()
for TASK in "${TASKS[@]}"; do
    JSONL="${DATA_DIR}/${TASK}/validation.jsonl"
    if [[ -f "${JSONL}" ]]; then
        AVAIL_TASKS+=("${TASK}")
    fi
done
NUM_AVAIL=${#AVAIL_TASKS[@]}

if [[ ${NUM_AVAIL} -eq 0 ]]; then
    echo "  ERROR: No RULER data available for datalen=${DATALEN}. Skipping."
    continue
fi

# ─── Step 2: Round-robin distribute tasks to GPUs ────────────────────────
echo ""
echo "── [2/3] Distributing ${NUM_AVAIL} tasks across ${NUM_GPUS} GPUs ──"

declare -a GPU_TASK_LISTS
for ((g = 0; g < NUM_GPUS; g++)); do
    GPU_TASK_LISTS[$g]=""
done

for ((i = 0; i < NUM_AVAIL; i++)); do
    g=$((i % NUM_GPUS))
    if [[ -z "${GPU_TASK_LISTS[$g]}" ]]; then
        GPU_TASK_LISTS[$g]="ruler/${AVAIL_TASKS[$i]}"
    else
        GPU_TASK_LISTS[$g]="${GPU_TASK_LISTS[$g]},ruler/${AVAIL_TASKS[$i]}"
    fi
done

for ((g = 0; g < NUM_GPUS; g++)); do
    if [[ -n "${GPU_TASK_LISTS[$g]}" ]]; then
        echo "  GPU ${GPU_ARRAY[$g]}: ${GPU_TASK_LISTS[$g]}"
    fi
done

# ─── Step 3: Launch parallel eval_acc.py processes ───────────────────────
echo ""
echo "── [3/3] Running evaluation (datalen=${DATALEN}) ──"

PIDS=()
ACTIVE_GPUS=()

for ((g = 0; g < NUM_GPUS; g++)); do
    TASK_LIST="${GPU_TASK_LISTS[$g]}"
    if [[ -z "${TASK_LIST}" ]]; then
        continue
    fi
    GPU_ID="${GPU_ARRAY[$g]}"
    LOG_FILE="${LOG_DIR}/gpu${GPU_ID}_${DATALEN}_$(date +%Y%m%d_%H%M%S).log"

    echo "  Launching GPU ${GPU_ID}: tasks=[${TASK_LIST}] → ${LOG_FILE}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" python test/eval_acc.py \
        --model_name "${MODEL}" \
        --dataset_name "${TASK_LIST}" \
        --num_samples "${NUM_SAMPLES}" \
        --datalen "${DATALEN}" \
        --method "${METHOD}" \
        --sparse_budget "${SPARSE_BUDGET}" \
        --rank "${RANK}" \
        --chunk_size "${CHUNK_SIZE}" \
        > "${LOG_FILE}" 2>&1 &

    PIDS+=($!)
    ACTIVE_GPUS+=("${GPU_ID}")
done

echo ""
echo "  Waiting for ${#PIDS[@]} GPU processes..."

FAILURES=0
for ((i = 0; i < ${#PIDS[@]}; i++)); do
    PID=${PIDS[$i]}
    GPU_ID=${ACTIVE_GPUS[$i]}
    if wait "${PID}"; then
        echo "  ✓ GPU ${GPU_ID} (PID ${PID}) — done"
    else
        echo "  ✗ GPU ${GPU_ID} (PID ${PID}) — FAILED (exit code: $?)"
        FAILURES=$((FAILURES + 1))
    fi
done
TOTAL_FAILURES=$((TOTAL_FAILURES + FAILURES))

if [[ ${FAILURES} -gt 0 ]]; then
    echo ""
    echo "  ⚠ ${FAILURES} GPU(s) failed for datalen=${DATALEN}. Check logs:"
    for ((g = 0; g < NUM_GPUS; g++)); do
        if [[ -n "${GPU_TASK_LISTS[$g]}" ]]; then
            GPU_ID="${GPU_ARRAY[$g]}"
            LATEST_LOG=$(ls -t "${LOG_DIR}/gpu${GPU_ID}_${DATALEN}_"*.log 2>/dev/null | head -1)
            if [[ -n "${LATEST_LOG}" ]]; then
                echo "  [GPU ${GPU_ID}] tail ${LATEST_LOG}:"
                tail -5 "${LATEST_LOG}" | sed 's/^/    /'
            fi
        fi
    done
fi

echo ""
done  # end datalen loop

# ══════════════════════════════════════════════════════════════════════════════
# Final Summary — aggregate ALL results across all datalens
# ══════════════════════════════════════════════════════════════════════════════
echo "════════════════════════ Results Summary ════════════════════════"
echo ""

python3 -c "
import json, os, glob, sys

model_short = '${MODEL_SHORT}'
method = '${METHOD}'
sparse_budget = '${SPARSE_BUDGET}'
rank = '${RANK}'
chunk_size = '${CHUNK_SIZE}'
datalens = [$(IFS=,; echo "${DATALENS[*]}")]

archive_dir = f'archive/{model_short}'

all_results = {}  # {datalen: [(task, samples, acc)]}

for datalen in datalens:
    pattern = f'{archive_dir}/ruler/*_{datalen}_{method}_{sparse_budget}_{rank}_{chunk_size}.jsonl'
    files = sorted(glob.glob(pattern))
    results = []
    for f in files:
        task = os.path.basename(f).replace(f'_{datalen}_{method}_{sparse_budget}_{rank}_{chunk_size}.jsonl', '')
        scores = []
        with open(f) as fh:
            for line in fh:
                data = json.loads(line)
                scores.extend(data.get('correct', []))
        if scores:
            avg = sum(scores) / len(scores)
            results.append((task, len(scores), avg))
    if results:
        all_results[datalen] = results

if not all_results:
    print('No result files found.')
    sys.exit(0)

print(f'Model: {model_short} | Method: {method}')
print()

for datalen in sorted(all_results.keys()):
    results = all_results[datalen]
    print(f'### DataLen: {datalen}')
    print()
    print('| Dataset | Samples | Accuracy |')
    print('|:--------|--------:|---------:|')
    total_score = 0
    for task, samples, acc in results:
        print(f'| ruler/{task} | {samples} | {acc:.4f} |')
        total_score += acc
    avg_all = total_score / len(results)
    print(f'| **Average** | - | **{avg_all:.4f}** |')
    print()
"

echo "════════════════════════════════════════════════════════════════"
echo "Full logs:    ${LOG_DIR}/"
echo "Result files: archive/${MODEL_SHORT}/ruler/"
if [[ ${TOTAL_FAILURES} -gt 0 ]]; then
    echo "⚠ Total failures: ${TOTAL_FAILURES}"
fi
echo "════════════════════════════════════════════════════════════════"
