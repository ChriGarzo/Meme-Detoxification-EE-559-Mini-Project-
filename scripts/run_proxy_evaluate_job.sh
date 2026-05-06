#!/usr/bin/env bash
set -euo pipefail

# Runs inside the RunAI pod. Generates proxy+BART rewrites on the held-out
# Stage 2 validation split, then evaluates them with the standard metrics.

CODE_ROOT="${1:-/scratch/hateful_meme_rewriting}"
HF_CACHE="${HF_CACHE:-/scratch/hf_cache}"
VALIDATION_JSONL="${VALIDATION_JSONL:-/scratch/hmr_stage2_dataset/val.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch}"
CHECKPOINT_SUFFIX="${CHECKPOINT_SUFFIX:-}"
PROXY_CHECKPOINT="${PROXY_CHECKPOINT:-/scratch/hmr_proxy_checkpoint/best_proxy.pt}"
PROXY_TEXT_PROMPT_FORMAT="${PROXY_TEXT_PROMPT_FORMAT:-none_legacy}"

cd "${CODE_ROOT}"

if [ -z "${EVAL_SUFFIX+x}" ]; then
    if [ "${PROXY_TEXT_PROMPT_FORMAT}" = "none_legacy" ]; then
        EVAL_SUFFIX="_proxy"
    else
        EVAL_SUFFIX="_proxy_text_explicit"
    fi
fi

PROXY_OUTPUT_DIR="${OUTPUT_ROOT}/hmr_eval_proxy_bart_full${EVAL_SUFFIX}"
EVAL_OUTPUT_DIR="${OUTPUT_ROOT}/hmr_eval_results_proxy${EVAL_SUFFIX}"
BART_CHECKPOINT="${OUTPUT_ROOT}/hmr_stage2_phase2_full${CHECKPOINT_SUFFIX}_checkpoint"

echo "=== Proxy+BART evaluation job ==="
echo "Code root:       ${CODE_ROOT}"
echo "HF cache:        ${HF_CACHE}"
echo "Val JSONL:       ${VALIDATION_JSONL}"
echo "BART checkpoint: ${BART_CHECKPOINT}"
echo "Proxy ckpt:      ${PROXY_CHECKPOINT}"
echo "Proxy output:    ${PROXY_OUTPUT_DIR}"
echo "Eval output:     ${EVAL_OUTPUT_DIR}"
echo "Text prompt fmt: ${PROXY_TEXT_PROMPT_FORMAT}"
echo ""

VAL_COUNT="$(wc -l < "${VALIDATION_JSONL}")"
OUTPUT_FILE="${PROXY_OUTPUT_DIR}/stage2_rewrites_proxy_bart_full.jsonl"

if [ -f "${OUTPUT_FILE}" ]; then
    OUTPUT_COUNT="$(wc -l < "${OUTPUT_FILE}")"
    NONEMPTY_REWRITES="$(python3 - "${OUTPUT_FILE}" <<'PY'
import json
import sys
count = 0
with open(sys.argv[1], encoding="utf-8", errors="replace") as f:
    for line in f:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(row.get("rewrite") or "").strip():
            count += 1
print(count)
PY
)"
    if [ "${PROXY_CHECKPOINT}" -nt "${OUTPUT_FILE}" ]; then
        echo "Proxy checkpoint is newer than existing output; recomputing."
        NEED_RECOMPUTE=1
    elif [ "${CODE_ROOT}/inference/run_proxy_pipeline.py" -nt "${OUTPUT_FILE}" ]; then
        echo "Proxy inference script is newer than existing output; recomputing."
        NEED_RECOMPUTE=1
    else
        NEED_RECOMPUTE=0
    fi
    if [ "${NEED_RECOMPUTE}" = "0" ] && [ "${OUTPUT_COUNT}" = "${VAL_COUNT}" ] && [ "${NONEMPTY_REWRITES}" = "${VAL_COUNT}" ]; then
        echo "Found complete proxy output (${OUTPUT_COUNT}/${VAL_COUNT}, non-empty rewrites=${NONEMPTY_REWRITES}); skipping inference."
    else
        echo "Found incomplete/invalid proxy output (${OUTPUT_COUNT}/${VAL_COUNT}, non-empty rewrites=${NONEMPTY_REWRITES}); recomputing."
        python3 "${CODE_ROOT}/inference/run_proxy_pipeline.py" \
            --input_jsonl "${VALIDATION_JSONL}" \
            --bart_checkpoint "${BART_CHECKPOINT}" \
            --proxy_checkpoint "${PROXY_CHECKPOINT}" \
            --output_dir "${PROXY_OUTPUT_DIR}" \
            --hf_cache "${HF_CACHE}" \
            --text_prompt_format "${PROXY_TEXT_PROMPT_FORMAT}" \
            --batch_size 8
    fi
else
    echo "Proxy output not found; running inference."
    python3 "${CODE_ROOT}/inference/run_proxy_pipeline.py" \
        --input_jsonl "${VALIDATION_JSONL}" \
        --bart_checkpoint "${BART_CHECKPOINT}" \
        --proxy_checkpoint "${PROXY_CHECKPOINT}" \
        --output_dir "${PROXY_OUTPUT_DIR}" \
        --hf_cache "${HF_CACHE}" \
        --text_prompt_format "${PROXY_TEXT_PROMPT_FORMAT}" \
        --batch_size 8
fi

echo "=== Evaluating proxy+BART outputs ==="
python3 "${CODE_ROOT}/evaluation/evaluate.py" \
    --validation_jsonl "${VALIDATION_JSONL}" \
    --proxy_output_dirs "${PROXY_OUTPUT_DIR}" \
    --output_dir "${EVAL_OUTPUT_DIR}" \
    --hf_cache "${HF_CACHE}"

echo "=== Proxy+BART evaluation complete ==="
