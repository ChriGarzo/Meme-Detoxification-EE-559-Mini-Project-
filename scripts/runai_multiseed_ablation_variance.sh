#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Multi-seed BART/proxy ablation study with variance aggregation.
#
# Default seeds: 1 2 3 4 5
# Default conditions: full target_only visual_only none
#
# Typical use:
#   bash scripts/runai_multiseed_ablation_variance.sh [UID]
#   PHASE=train_proxy bash scripts/runai_multiseed_ablation_variance.sh [UID]
#   PHASE=evaluate bash scripts/runai_multiseed_ablation_variance.sh [UID]
#   PHASE=aggregate bash scripts/runai_multiseed_ablation_variance.sh [UID]
#
# PHASE options:
#   train_bart     Submit BART fine-tuning jobs. Default: one job per seed+condition.
#   train_proxy    Submit one proxy-training job per seed, after full BART is done.
#   evaluate       Submit one test inference/evaluation job per seed.
#   aggregate      Aggregate per-seed evaluation summaries locally; no GPU.
#   seed_pipeline  Submit one long job per seed. It trains all conditions and proxy
#                  sequentially. Set RUN_EVAL_AFTER_TRAINING=1 to evaluate too.
#
# Job granularity for PHASE=train_bart:
#   JOB_GRANULARITY=ablation  -> 5 seeds x 4 conditions = 20 jobs (default)
#   JOB_GRANULARITY=seed      -> 5 jobs, each loops over all conditions
# =============================================================================

if [ "$#" -gt 1 ]; then
    echo "Usage: bash $0 [UID_NUMBER]"
    exit 1
fi

USERNAME="${USER}"
GROUP_NUM="${GROUP_NUM:-31}"
IMAGE="${IMAGE:-registry.rcp.epfl.ch/ee-559-garzone/hmr:v0.1}"
REPO_ROOT_LOCAL="$(cd "$(dirname "$0")/.." && pwd)"

PHASE="${PHASE:-train_bart}"
JOB_GRANULARITY="${JOB_GRANULARITY:-ablation}"
SEEDS_STR="${SEEDS:-1 2 3 4 5}"
CONDITIONS_STR="${CONDITIONS:-full target_only visual_only none}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-hmr_multiseed_explicit_detox}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-_explicit_detox}"
INPUT_FORMAT="${INPUT_FORMAT:-explicit_detox}"
TASK_PREFIX="${TASK_PREFIX:-}"
HF_CACHE="${HF_CACHE:-/scratch/hf_cache}"
STAGES_ROOT="${STAGES_ROOT:-/scratch/stages}"
EVAL_ROOT="${EVAL_ROOT:-/scratch/eval_results}"
DATASET_DIR="${DATASET_DIR:-${STAGES_ROOT}/hmr_stage2_dataset}"
STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${STAGES_ROOT}/hmr_stage1_output}"
VALIDATION_JSONL="${VALIDATION_JSONL:-${DATASET_DIR}/test.jsonl}"
BASE_MODEL="${BASE_MODEL:-facebook/bart-large}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

PROXY_NUM_TRAIN_EPOCHS="${PROXY_NUM_TRAIN_EPOCHS:-20}"
PROXY_BATCH_SIZE="${PROXY_BATCH_SIZE:-64}"
PROXY_LEARNING_RATE="${PROXY_LEARNING_RATE:-1e-3}"
PROXY_NUM_SOFT_TOKENS="${PROXY_NUM_SOFT_TOKENS:-16}"
PROXY_TEXT_PROMPT_FORMAT="${PROXY_TEXT_PROMPT_FORMAT:-none_explicit_detox}"

BART_EVAL_BATCH_SIZE="${BART_EVAL_BATCH_SIZE:-4}"
PROXY_EVAL_BATCH_SIZE="${PROXY_EVAL_BATCH_SIZE:-8}"
NUM_BEAMS="${NUM_BEAMS:-4}"
MAX_LENGTH="${MAX_LENGTH:-64}"
INCLUDE_BART_BASE="${INCLUDE_BART_BASE:-0}"
INCLUDE_DETOXLLM="${INCLUDE_DETOXLLM:-0}"
BASELINE_SEED="${BASELINE_SEED:-42}"
SKIP_PROXY="${SKIP_PROXY:-0}"
SKIP_CLIPSCORE="${SKIP_CLIPSCORE:-0}"
RUN_EVAL_AFTER_TRAINING="${RUN_EVAL_AFTER_TRAINING:-0}"
DEBUG="${DEBUG:-0}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"

if [ -n "${1:-}" ]; then
    UID_NUM="$1"
else
    UID_NUM="$(id -u)"
fi

if [[ "${REPO_ROOT_LOCAL}" == /mnt/course-ee-559/* || "${REPO_ROOT_LOCAL}" == /scratch/* ]]; then
    CODE_ROOT="/scratch/hateful_meme_rewriting"
    MODE_LABEL="scratch"
else
    CODE_ROOT="/home/${USERNAME}/hateful_meme_rewriting"
    MODE_LABEL="home"
fi

PY_SCRIPT="${CODE_ROOT}/scripts/run_multiseed_ablation_variance.py"
if [ ! -f "${PY_SCRIPT}" ]; then
    if [ -f "${REPO_ROOT_LOCAL}/scripts/run_multiseed_ablation_variance.py" ]; then
        if [ "${PHASE}" = "aggregate" ]; then
            CODE_ROOT="${REPO_ROOT_LOCAL}"
            PY_SCRIPT="${REPO_ROOT_LOCAL}/scripts/run_multiseed_ablation_variance.py"
        else
            echo "Note: ${PY_SCRIPT} is not visible on this node."
            echo "      RunAI pods will use /scratch/hateful_meme_rewriting."
            CODE_ROOT="/scratch/hateful_meme_rewriting"
            PY_SCRIPT="${CODE_ROOT}/scripts/run_multiseed_ablation_variance.py"
        fi
    else
        echo "ERROR: Python orchestrator not found at ${PY_SCRIPT}"
        exit 1
    fi
fi

if [ "${PHASE}" = "aggregate" ] && [[ "${REPO_ROOT_LOCAL}" == /mnt/course-ee-559/* ]]; then
    LOCAL_SCRATCH_ROOT="$(cd "${REPO_ROOT_LOCAL}/.." && pwd)"

    localize_scratch_path() {
        local path_value="$1"
        if [ "${path_value}" = "/scratch" ]; then
            printf '%s\n' "${LOCAL_SCRATCH_ROOT}"
        elif [[ "${path_value}" == /scratch/* ]]; then
            printf '%s/%s\n' "${LOCAL_SCRATCH_ROOT}" "${path_value#/scratch/}"
        else
            printf '%s\n' "${path_value}"
        fi
    }

    HF_CACHE="$(localize_scratch_path "${HF_CACHE}")"
    STAGES_ROOT="$(localize_scratch_path "${STAGES_ROOT}")"
    EVAL_ROOT="$(localize_scratch_path "${EVAL_ROOT}")"
    DATASET_DIR="$(localize_scratch_path "${DATASET_DIR}")"
    STAGE1_OUTPUT_DIR="$(localize_scratch_path "${STAGE1_OUTPUT_DIR}")"
    VALIDATION_JSONL="$(localize_scratch_path "${VALIDATION_JSONL}")"
fi

SEEDS=(${SEEDS_STR})
CONDITIONS=(${CONDITIONS_STR})
JOB_SUFFIX="${OUTPUT_SUFFIX//_/-}"

COMMON_ARGS=(
    --repo_root "${CODE_ROOT}"
    --stages_root "${STAGES_ROOT}"
    --eval_root "${EVAL_ROOT}"
    --hf_cache "${HF_CACHE}"
    --dataset_dir "${DATASET_DIR}"
    --stage1_output_dir "${STAGE1_OUTPUT_DIR}"
    --validation_jsonl "${VALIDATION_JSONL}"
    --experiment_name "${EXPERIMENT_NAME}"
    --output_suffix "${OUTPUT_SUFFIX}"
    --input_format "${INPUT_FORMAT}"
)
if [ -n "${TASK_PREFIX}" ]; then
    COMMON_ARGS+=(--task_prefix "${TASK_PREFIX}")
fi
if [ "${DEBUG}" = "1" ]; then
    COMMON_ARGS+=(--debug)
fi
if [ "${DRY_RUN}" = "1" ]; then
    COMMON_ARGS+=(--dry_run)
fi
if [ "${FORCE}" = "1" ]; then
    COMMON_ARGS+=(--force)
fi

TRAIN_ARGS=(
    --base_model "${BASE_MODEL}"
    --num_train_epochs "${NUM_TRAIN_EPOCHS}"
    --per_device_train_batch_size "${TRAIN_BATCH_SIZE}"
    --learning_rate "${LEARNING_RATE}"
    --warmup_steps "${WARMUP_STEPS}"
    --weight_decay "${WEIGHT_DECAY}"
    --lora_r "${LORA_R}"
    --lora_alpha "${LORA_ALPHA}"
    --lora_dropout "${LORA_DROPOUT}"
)

PROXY_ARGS=(
    --proxy_num_train_epochs "${PROXY_NUM_TRAIN_EPOCHS}"
    --proxy_batch_size "${PROXY_BATCH_SIZE}"
    --proxy_learning_rate "${PROXY_LEARNING_RATE}"
    --proxy_num_soft_tokens "${PROXY_NUM_SOFT_TOKENS}"
)
SKIP_PROXY_ARGS=()
if [ "${SKIP_PROXY}" = "1" ]; then
    SKIP_PROXY_ARGS+=(--skip_proxy)
fi

EVAL_ARGS=(
    --conditions "${CONDITIONS[@]}"
    --bart_eval_batch_size "${BART_EVAL_BATCH_SIZE}"
    --proxy_eval_batch_size "${PROXY_EVAL_BATCH_SIZE}"
    --num_beams "${NUM_BEAMS}"
    --max_length "${MAX_LENGTH}"
    --proxy_text_prompt_format "${PROXY_TEXT_PROMPT_FORMAT}"
    --baseline_seed "${BASELINE_SEED}"
)
if [ "${SKIP_PROXY}" = "1" ]; then
    EVAL_ARGS+=(--skip_proxy)
fi
if [ "${INCLUDE_BART_BASE}" = "1" ]; then
    EVAL_ARGS+=(--include_bart_base)
fi
if [ "${INCLUDE_DETOXLLM}" = "1" ]; then
    EVAL_ARGS+=(--include_detoxllm)
fi
if [ "${SKIP_CLIPSCORE}" = "1" ]; then
    EVAL_ARGS+=(--skip_clipscore)
fi

print_header() {
    echo "=== Multi-seed ablation variance ==="
    echo "  Phase:        ${PHASE}"
    echo "  User:         ${USERNAME} (UID: ${UID_NUM})"
    echo "  Mode:         ${MODE_LABEL}"
    echo "  Code root:    ${CODE_ROOT}"
    echo "  Experiment:   ${EXPERIMENT_NAME}"
    echo "  Seeds:        ${SEEDS[*]}"
    echo "  Conditions:   ${CONDITIONS[*]}"
    echo "  Input format: ${INPUT_FORMAT}"
    echo "  Output suffix:${OUTPUT_SUFFIX}"
    echo "  Test JSONL:   ${VALIDATION_JSONL}"
    echo "  Image:        ${IMAGE}"
    echo ""
}

submit_gpu_job() {
    local job_name="$1"
    shift
    echo "Submitting ${job_name}"
    local submit_output
    if ! submit_output="$(runai submit "${job_name}" \
        --run-as-uid "${UID_NUM}" \
        --image "${IMAGE}" \
        --node-pools a100-40g \
        --gpu 1 \
        --cpu 8 \
        --memory 40Gi \
        --existing-pvc claimname=home,path=/home/${USERNAME} \
        --existing-pvc claimname=course-ee-559-scratch-g${GROUP_NUM},path=/scratch \
        --existing-pvc claimname=course-ee-559-shared-ro,path=/shared-ro \
        --existing-pvc claimname=course-ee-559-shared-rw,path=/shared-rw \
        --command -- python3 "${PY_SCRIPT}" "$@" 2>&1)"; then
        echo "${submit_output}"
        if echo "${submit_output}" | grep -Eiq "invalid_grant|Token is not active|failed to refresh token"; then
            echo ""
            echo "RunAI authentication has expired. Re-login on the login node, then rerun this script:"
            echo "  runai login"
            echo "  runai config project course-ee-559-${USERNAME}"
            echo "  bash scripts/runai_multiseed_ablation_variance.sh ${UID_NUM}"
        fi
        exit 1
    fi
    echo "${submit_output}"
}

print_header

case "${PHASE}" in
    train_bart)
        if [ "${JOB_GRANULARITY}" = "seed" ]; then
            for SEED in "${SEEDS[@]}"; do
                submit_gpu_job "hmr-ms-s${SEED}-train${JOB_SUFFIX}" \
                    train-seed \
                    "${COMMON_ARGS[@]}" \
                    "${TRAIN_ARGS[@]}" \
                    "${PROXY_ARGS[@]}" \
                    "${SKIP_PROXY_ARGS[@]}" \
                    --seed "${SEED}" \
                    --conditions "${CONDITIONS[@]}"
            done
        elif [ "${JOB_GRANULARITY}" = "ablation" ]; then
            for SEED in "${SEEDS[@]}"; do
                for CONDITION in "${CONDITIONS[@]}"; do
                    SAFE_CONDITION="${CONDITION//_/-}"
                    submit_gpu_job "hmr-ms-s${SEED}-${SAFE_CONDITION}${JOB_SUFFIX}" \
                        train-bart \
                        "${COMMON_ARGS[@]}" \
                        "${TRAIN_ARGS[@]}" \
                        --seed "${SEED}" \
                        --condition "${CONDITION}"
                done
            done
        else
            echo "ERROR: JOB_GRANULARITY must be 'ablation' or 'seed'."
            exit 1
        fi
        ;;

    train_proxy)
        if [ "${SKIP_PROXY}" = "1" ]; then
            echo "SKIP_PROXY=1, so no proxy training jobs were submitted."
            exit 0
        fi
        for SEED in "${SEEDS[@]}"; do
            submit_gpu_job "hmr-ms-s${SEED}-proxy${JOB_SUFFIX}" \
                train-proxy \
                "${COMMON_ARGS[@]}" \
                "${PROXY_ARGS[@]}" \
                --seed "${SEED}"
        done
        ;;

    evaluate)
        for SEED in "${SEEDS[@]}"; do
            submit_gpu_job "hmr-ms-s${SEED}-eval${JOB_SUFFIX}" \
                evaluate-seed \
                "${COMMON_ARGS[@]}" \
                "${EVAL_ARGS[@]}" \
                --seed "${SEED}"
        done
        ;;

    seed_pipeline)
        for SEED in "${SEEDS[@]}"; do
            PIPELINE_ARGS=(
                seed-pipeline
                "${COMMON_ARGS[@]}"
                "${TRAIN_ARGS[@]}"
                "${PROXY_ARGS[@]}"
                "${EVAL_ARGS[@]}"
                "${SKIP_PROXY_ARGS[@]}"
                --seed "${SEED}"
            )
            if [ "${RUN_EVAL_AFTER_TRAINING}" = "1" ]; then
                PIPELINE_ARGS+=(--run_eval_after_training)
            fi
            submit_gpu_job "hmr-ms-s${SEED}-pipeline${JOB_SUFFIX}" "${PIPELINE_ARGS[@]}"
        done
        ;;

    aggregate)
        python3 "${REPO_ROOT_LOCAL}/scripts/run_multiseed_ablation_variance.py" \
            aggregate \
            --repo_root "${REPO_ROOT_LOCAL}" \
            --stages_root "${STAGES_ROOT}" \
            --eval_root "${EVAL_ROOT}" \
            --hf_cache "${HF_CACHE}" \
            --dataset_dir "${DATASET_DIR}" \
            --stage1_output_dir "${STAGE1_OUTPUT_DIR}" \
            --validation_jsonl "${VALIDATION_JSONL}" \
            --experiment_name "${EXPERIMENT_NAME}" \
            --output_suffix "${OUTPUT_SUFFIX}" \
            --input_format "${INPUT_FORMAT}" \
            --seeds "${SEEDS[@]}"
        ;;

    *)
        echo "ERROR: Unknown PHASE '${PHASE}'."
        echo "Valid phases: train_bart train_proxy evaluate aggregate seed_pipeline"
        exit 1
        ;;
esac

echo ""
echo "Done with PHASE=${PHASE}."
if [ "${PHASE}" = "train_bart" ] && [ "${JOB_GRANULARITY}" = "ablation" ]; then
    echo "After all BART jobs finish, run:"
    echo "  PHASE=train_proxy bash scripts/runai_multiseed_ablation_variance.sh ${1:-}"
elif [ "${PHASE}" = "train_proxy" ]; then
    echo "After all proxy jobs finish, run:"
    echo "  PHASE=evaluate bash scripts/runai_multiseed_ablation_variance.sh ${1:-}"
elif [ "${PHASE}" = "evaluate" ]; then
    echo "After all evaluation jobs finish, run:"
    echo "  PHASE=aggregate bash scripts/runai_multiseed_ablation_variance.sh ${1:-}"
fi
