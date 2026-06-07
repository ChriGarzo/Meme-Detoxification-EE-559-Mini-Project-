#!/usr/bin/env bash
set -euo pipefail

# Unified evaluation job (runs inside the RunAI pod).
# Sequentially runs all inference steps with skip-if-complete logic, then
# calls evaluate.py once with all systems combined.
#
# Systems evaluated:
#   - LLaVA teacher (from validation JSONL)
#   - BART base (non-finetuned, 4 conditions)
#   - BART finetuned (4 conditions)
#   - DetoxLLM (UBC-NLP/DetoxLLM-7B)
#   - Proxy + BART full
#
# Output: ${OUTPUT_ROOT}/hmr_eval_results${EVAL_SUFFIX}/

CODE_ROOT="${1:-/scratch/hateful_meme_rewriting}"
HF_CACHE="${HF_CACHE:-/scratch/hf_cache}"
STAGES_ROOT="${STAGES_ROOT:-/scratch/stages}"
EVAL_ROOT="${EVAL_ROOT:-/scratch/eval_results}"
VALIDATION_JSONL="${VALIDATION_JSONL:-${STAGES_ROOT}/hmr_stage2_dataset/test.jsonl}"
INPUT_FORMAT="${INPUT_FORMAT:-explicit_detox}"
TASK_PREFIX="${TASK_PREFIX:-}"
CHECKPOINT_SUFFIX="${CHECKPOINT_SUFFIX:-_explicit_detox}"
EVAL_SUFFIX="${EVAL_SUFFIX:-_explicit_detox}"
PROXY_CHECKPOINT="${PROXY_CHECKPOINT:-${STAGES_ROOT}/hmr_proxy_checkpoint_explicit_detox/best_proxy.pt}"
PROXY_TEXT_PROMPT_FORMAT="${PROXY_TEXT_PROMPT_FORMAT:-none_explicit_detox}"

TASK_PREFIX_ARGS=()
if [ -n "${TASK_PREFIX}" ]; then
    TASK_PREFIX_ARGS=(--task_prefix "${TASK_PREFIX}")
fi

cd "${CODE_ROOT}"

echo "=== Unified evaluation job ==="
echo "Code root:       ${CODE_ROOT}"
echo "HF cache:        ${HF_CACHE}"
echo "Val JSONL:       ${VALIDATION_JSONL}"
echo "Input fmt:       ${INPUT_FORMAT}"
echo "Ckpt suffix:     ${CHECKPOINT_SUFFIX}"
echo "Eval suffix:     ${EVAL_SUFFIX}"
echo "Proxy ckpt:      ${PROXY_CHECKPOINT}"
echo "Proxy prompt:    ${PROXY_TEXT_PROMPT_FORMAT}"
echo "Stages root:     ${STAGES_ROOT}"
echo "Eval root:       ${EVAL_ROOT}"
echo "Output:          ${EVAL_ROOT}/hmr_eval_results${EVAL_SUFFIX}"
echo ""

VAL_COUNT="$(wc -l < "${VALIDATION_JSONL}")"
echo "Validation examples: ${VAL_COUNT}"
echo ""

# ---------------------------------------------------------------------------
# Helper: run stage2 BART inference if output is missing or incomplete
# ---------------------------------------------------------------------------
run_stage2_if_needed() {
    local cond="$1"
    local checkpoint_dir="$2"
    local output_dir="$3"
    local output_file="${output_dir}/stage2_rewrites_${cond}.jsonl"

    if [ -f "${output_file}" ]; then
        local output_count
        output_count="$(wc -l < "${output_file}")"
        if [ "${output_count}" = "${VAL_COUNT}" ]; then
            echo "--- ${cond}: found ${output_file} (${output_count}/${VAL_COUNT}); skipping ---"
            return 0
        fi
        echo "--- ${cond}: incomplete ${output_file} (${output_count}/${VAL_COUNT}); recomputing ---"
    else
        echo "--- ${cond}: not found; running inference ---"
    fi

    python3 "${CODE_ROOT}/inference/run_stage2.py" \
        --condition "${cond}" \
        --checkpoint_dir "${checkpoint_dir}" \
        --input_jsonl "${VALIDATION_JSONL}" \
        --output_dir "${output_dir}" \
        --hf_cache "${HF_CACHE}" \
        --input_format "${INPUT_FORMAT}" \
        "${TASK_PREFIX_ARGS[@]}" \
        --batch_size 4
}

# ---------------------------------------------------------------------------
# Step 1: BART base inference (4 conditions)
# ---------------------------------------------------------------------------
echo "=== BART base inference ==="
for COND in full target_only visual_only none; do
    run_stage2_if_needed \
        "${COND}" \
        "facebook/bart-large" \
        "${EVAL_ROOT}/hmr_eval_bart_base_${COND}${EVAL_SUFFIX}"
done

# ---------------------------------------------------------------------------
# Step 2: Finetuned BART inference (4 conditions)
# ---------------------------------------------------------------------------
echo "=== Finetuned BART inference ==="
for COND in full target_only visual_only none; do
    run_stage2_if_needed \
        "${COND}" \
        "${STAGES_ROOT}/hmr_stage2_${COND}${CHECKPOINT_SUFFIX}_checkpoint" \
        "${EVAL_ROOT}/hmr_eval_stage2_${COND}${EVAL_SUFFIX}"
done

# ---------------------------------------------------------------------------
# Step 3: DetoxLLM inference
# ---------------------------------------------------------------------------
DETOXLLM_OUTPUT_DIR="${EVAL_ROOT}/hmr_eval_detoxllm${EVAL_SUFFIX}"
DETOXLLM_OUTPUT_FILE="${DETOXLLM_OUTPUT_DIR}/detoxllm_rewrites.jsonl"

echo "=== DetoxLLM inference ==="
if [ -f "${DETOXLLM_OUTPUT_FILE}" ]; then
    DETOX_COUNT="$(wc -l < "${DETOXLLM_OUTPUT_FILE}")"
    if [ "${DETOX_COUNT}" = "${VAL_COUNT}" ]; then
        echo "Found complete DetoxLLM output (${DETOX_COUNT}/${VAL_COUNT}); skipping."
    else
        echo "Incomplete DetoxLLM output (${DETOX_COUNT}/${VAL_COUNT}); recomputing."
        python3 "${CODE_ROOT}/baselines/run_detoxllm_baseline.py" \
            --validation_jsonl "${VALIDATION_JSONL}" \
            --output_dir "${DETOXLLM_OUTPUT_DIR}" \
            --hf_cache "${HF_CACHE}"
    fi
else
    echo "DetoxLLM output not found; running inference."
    python3 "${CODE_ROOT}/baselines/run_detoxllm_baseline.py" \
        --validation_jsonl "${VALIDATION_JSONL}" \
        --output_dir "${DETOXLLM_OUTPUT_DIR}" \
        --hf_cache "${HF_CACHE}"
fi

# ---------------------------------------------------------------------------
# Step 4: Proxy + BART inference
# ---------------------------------------------------------------------------
PROXY_OUTPUT_DIR="${EVAL_ROOT}/hmr_eval_clip_proxy_bart_full${EVAL_SUFFIX}"
PROXY_OUTPUT_FILE="${PROXY_OUTPUT_DIR}/stage2_rewrites_clip_proxy_bart_full.jsonl"
BART_CHECKPOINT="${STAGES_ROOT}/hmr_stage2_full${CHECKPOINT_SUFFIX}_checkpoint"

echo "=== Proxy + BART inference ==="
NEED_PROXY_RECOMPUTE=0
if [ -f "${PROXY_OUTPUT_FILE}" ]; then
    PROXY_COUNT="$(wc -l < "${PROXY_OUTPUT_FILE}")"
    NONEMPTY_REWRITES="$(python3 -c "
import json, sys
count = sum(1 for line in open('${PROXY_OUTPUT_FILE}') if str(json.loads(line).get('rewrite') or '').strip())
print(count)
")"
    if [ "${PROXY_CHECKPOINT}" -nt "${PROXY_OUTPUT_FILE}" ]; then
        echo "Proxy checkpoint newer than output; recomputing."
        NEED_PROXY_RECOMPUTE=1
    elif [ "${PROXY_COUNT}" != "${VAL_COUNT}" ] || [ "${NONEMPTY_REWRITES}" != "${VAL_COUNT}" ]; then
        echo "Incomplete proxy output (${PROXY_COUNT}/${VAL_COUNT}, non-empty=${NONEMPTY_REWRITES}); recomputing."
        NEED_PROXY_RECOMPUTE=1
    else
        echo "Found complete proxy output (${PROXY_COUNT}/${VAL_COUNT}); skipping."
    fi
else
    echo "Proxy output not found; running inference."
    NEED_PROXY_RECOMPUTE=1
fi

if [ "${NEED_PROXY_RECOMPUTE}" = "1" ]; then
    python3 "${CODE_ROOT}/inference/run_proxy_pipeline.py" \
        --input_jsonl "${VALIDATION_JSONL}" \
        --bart_checkpoint "${BART_CHECKPOINT}" \
        --proxy_checkpoint "${PROXY_CHECKPOINT}" \
        --output_dir "${PROXY_OUTPUT_DIR}" \
        --hf_cache "${HF_CACHE}" \
        --text_prompt_format "${PROXY_TEXT_PROMPT_FORMAT}" \
        --batch_size 8
fi

# ---------------------------------------------------------------------------
# Step 5: Single evaluate.py call with all systems
# ---------------------------------------------------------------------------
echo "=== Final evaluation: all systems ==="
python3 "${CODE_ROOT}/evaluation/evaluate.py" \
    --validation_jsonl "${VALIDATION_JSONL}" \
    --bart_base_output_dirs \
        "${EVAL_ROOT}/hmr_eval_bart_base_full${EVAL_SUFFIX}" \
        "${EVAL_ROOT}/hmr_eval_bart_base_target_only${EVAL_SUFFIX}" \
        "${EVAL_ROOT}/hmr_eval_bart_base_visual_only${EVAL_SUFFIX}" \
        "${EVAL_ROOT}/hmr_eval_bart_base_none${EVAL_SUFFIX}" \
    --bart_finetuned_output_dirs \
        "${EVAL_ROOT}/hmr_eval_stage2_full${EVAL_SUFFIX}" \
        "${EVAL_ROOT}/hmr_eval_stage2_target_only${EVAL_SUFFIX}" \
        "${EVAL_ROOT}/hmr_eval_stage2_visual_only${EVAL_SUFFIX}" \
        "${EVAL_ROOT}/hmr_eval_stage2_none${EVAL_SUFFIX}" \
    --detoxllm_output_path "${DETOXLLM_OUTPUT_DIR}" \
    --proxy_output_dirs "${PROXY_OUTPUT_DIR}" \
    --output_dir "${EVAL_ROOT}/hmr_eval_results${EVAL_SUFFIX}" \
    --hf_cache "${HF_CACHE}"

echo "=== Evaluation complete ==="
echo "Results: ${EVAL_ROOT}/hmr_eval_results${EVAL_SUFFIX}"
