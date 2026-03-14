#!/bin/bash
################################################################################
#
# Multi-GPU RULER Benchmark Runner for ShadowKV
#
# Iterates over configured models × datalens × tasks, distributing work
# across multiple GPUs in parallel (round-robin). Auto-generates missing
# RULER data, runs eval_acc.py per GPU, and aggregates results into a
# summary table + CSV.
#
# Usage:
#   bash scripts/run_ruler.sh --gpus 0,1 --method full --num_samples 10
#
################################################################################

set -uo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# Configuration Lists — Edit these to control what gets tested
# ══════════════════════════════════════════════════════════════════════════════

# Models to evaluate (local paths)
MODELS=(
    "/home/zijie/models/Llama-3.1-8B-Instruct"
    "/home/zijie/models/GLM-4-9B-Chat-1M"
    "/home/zijie/models/Qwen2.5-7B-Instruct-1M"
)

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

# ─── Template detection function ────────────────────────────────────────────
detect_template() {
    local model_lower
    model_lower=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    if [[ "${model_lower}" == *"llama-3"* ]] || [[ "${model_lower}" == *"llama_3"* ]] || [[ "${model_lower}" == *"llama3"* ]]; then
        echo "llama-3"
    elif [[ "${model_lower}" == *"llama-2"* ]] || [[ "${model_lower}" == *"llama_2"* ]] || [[ "${model_lower}" == *"llama2"* ]]; then
        echo "llama-2"
    elif [[ "${model_lower}" == *"yi"* ]]; then
        echo "yi"
    elif [[ "${model_lower}" == *"glm"* ]]; then
        echo "glm"
    elif [[ "${model_lower}" == *"qwen"* ]]; then
        echo "qwen"
    elif [[ "${model_lower}" == *"phi"* ]]; then
        echo "phi"
    else
        echo ""
    fi
}

# ─── Derived variables ──────────────────────────────────────────────────────
IFS=',' read -ra GPU_ARRAY <<< "${GPUS}"
NUM_GPUS=${#GPU_ARRAY[@]}
NUM_TASKS=${#TASKS[@]}
NUM_LENS=${#DATALENS[@]}
NUM_MODELS=${#MODELS[@]}

# ─── Print configuration ────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              ShadowKV RULER Benchmark Runner                ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  Models:   %-48s║\n" "${NUM_MODELS} models"
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
for M in "${MODELS[@]}"; do
    printf "  • %s\n" "$(basename "${M}")"
done
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Main loop: iterate over each model × datalen
# ═══════════════════════════════════════════════════════════════════════════
TOTAL_FAILURES=0

for MODEL in "${MODELS[@]}"; do

MODEL_SHORT=$(basename "${MODEL}")
TEMPLATE=$(detect_template "${MODEL}")

if [[ -z "${TEMPLATE}" ]]; then
    echo "⚠ Skipping ${MODEL_SHORT}: cannot detect template"
    continue
fi

LOG_DIR="archive/${MODEL_SHORT}/logs"
mkdir -p "${LOG_DIR}"

echo "╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍"
echo "  Model: ${MODEL_SHORT}  (template: ${TEMPLATE})"
echo "╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍"

for DATALEN in "${DATALENS[@]}"; do

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ${MODEL_SHORT} | DataLen: ${DATALEN}"
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
    echo "  ERROR: No RULER data available for ${MODEL_SHORT} datalen=${DATALEN}. Skipping."
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
echo "── [3/3] Running evaluation ──"

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
HAS_OOM=false
for ((i = 0; i < ${#PIDS[@]}; i++)); do
    PID=${PIDS[$i]}
    GPU_ID=${ACTIVE_GPUS[$i]}
    if wait "${PID}"; then
        echo "  ✓ GPU ${GPU_ID} (PID ${PID}) — done"
    else
        EXIT_CODE=$?
        # Check if failure was due to OOM
        LATEST_LOG=$(ls -t "${LOG_DIR}/gpu${GPU_ID}_${DATALEN}_"*.log 2>/dev/null | head -1)
        if [[ -n "${LATEST_LOG}" ]] && grep -qE "CUDA out of memory|OutOfMemoryError|torch.cuda.OutOfMemoryError" "${LATEST_LOG}"; then
            echo "  ⚠ GPU ${GPU_ID} (PID ${PID}) — OOM (out of memory)"
            HAS_OOM=true
        else
            echo "  ✗ GPU ${GPU_ID} (PID ${PID}) — FAILED (exit code: ${EXIT_CODE})"
        fi
        FAILURES=$((FAILURES + 1))
    fi
done
TOTAL_FAILURES=$((TOTAL_FAILURES + FAILURES))

# Write OOM marker if any GPU hit OOM for this model+datalen
if [[ "${HAS_OOM}" == true ]]; then
    OOM_MARKER="archive/${MODEL_SHORT}/ruler/.oom_${DATALEN}_${METHOD}_${SPARSE_BUDGET}_${RANK}_${CHUNK_SIZE}"
    mkdir -p "$(dirname "${OOM_MARKER}")"
    touch "${OOM_MARKER}"
    echo "  → OOM marker written: ${OOM_MARKER}"
fi

if [[ ${FAILURES} -gt 0 && "${HAS_OOM}" == false ]]; then
    echo ""
    echo "  ⚠ ${FAILURES} GPU(s) failed for ${MODEL_SHORT} datalen=${DATALEN}. Check logs:"
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

done  # end datalen loop
done  # end model loop

# ══════════════════════════════════════════════════════════════════════════════
# Final Summary — aggregate ALL results across all models × datalens
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════ Results Summary ════════════════════════"
echo ""

python3 scripts/summarize_ruler.py \
    --models "${MODELS[@]}" \
    --method "${METHOD}" \
    --sparse_budget "${SPARSE_BUDGET}" \
    --rank "${RANK}" \
    --chunk_size "${CHUNK_SIZE}" \
    --datalens "${DATALENS[@]}"

echo "════════════════════════════════════════════════════════════════"
echo "Result files: archive/*/ruler/"
if [[ ${TOTAL_FAILURES} -gt 0 ]]; then
    echo "⚠ Total failures: ${TOTAL_FAILURES}"
fi
echo "════════════════════════════════════════════════════════════════"
