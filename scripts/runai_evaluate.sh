#!/usr/bin/env bash
set -e

# =============================================================================
# Stage 3: Full evaluation (GPU A100-40G)
#
# Usage:
#   bash scripts/runai_evaluate.sh <UID_NUMBER>   # home code path
#   bash scripts/runai_evaluate.sh                # scratch code path
#
# Example:
#   bash scripts/runai_evaluate.sh 123456
#
# Note: Run this AFTER all Stage 2 Phase 2 jobs and proxy training have completed.
#       Runs inference for all 4 BART conditions + proxy pipeline + baselines,
#       then computes STA/SIM/CLIPScore/RewritePrecision/J metrics.
# =============================================================================

# --- Validate args ---
if [ "$#" -gt 1 ]; then
    echo "ERROR: Too many arguments."
    echo "Usage:"
    echo "  bash $0 <UID_NUMBER>   # use /home/\${USER}/hateful_meme_rewriting"
    echo "  bash $0                # use /scratch/hateful_meme_rewriting"
    exit 1
fi

# --- Configuration (do NOT edit these) ---
USERNAME="${USER}"
GROUP_NUM="31"
IMAGE="registry.rcp.epfl.ch/ee-559-garzone/hmr:v0.1"
REPO_ROOT_LOCAL="$(cd "$(dirname "$0")/.." && pwd)"

# --- Path/UID mode selection ---
if [ -n "$1" ]; then
    UID_NUM="$1"
    CODE_ROOT="/home/${USERNAME}/hateful_meme_rewriting"
    MODE_LABEL="home"
else
    UID_NUM="$(id -u)"
    CODE_ROOT="/scratch/hateful_meme_rewriting"
    MODE_LABEL="scratch"
fi

EVAL_SCRIPT="${CODE_ROOT}/evaluation/evaluate.py"
if [ ! -f "${EVAL_SCRIPT}" ]; then
    if [ "${MODE_LABEL}" = "scratch" ] && [ -f "${REPO_ROOT_LOCAL}/evaluation/evaluate.py" ]; then
        echo "Note: /scratch path not visible on this node; using local repo check at ${REPO_ROOT_LOCAL}."
    else
        echo "ERROR: Evaluation script not found at: ${EVAL_SCRIPT}"
        echo "Check that the repository exists at ${CODE_ROOT}."
        exit 1
    fi
fi

echo "=== Stage 3: Full Evaluation ==="
echo "  User:  ${USERNAME} (UID: ${UID_NUM})"
echo "  Mode:  ${MODE_LABEL}"
echo "  Code:  ${CODE_ROOT}"
echo "  Group: ${GROUP_NUM}"
echo "  Image: ${IMAGE}"
echo ""

runai submit hmr-evaluate \
    --run-as-uid ${UID_NUM} \
    --image ${IMAGE} \
    --node-pools a100-40g \
    --gpu 1 \
    --cpu 8 \
    --memory 40Gi \
    --existing-pvc claimname=home,path=/home/${USERNAME} \
    --existing-pvc claimname=course-ee-559-scratch-g${GROUP_NUM},path=/scratch \
    --existing-pvc claimname=course-ee-559-shared-ro,path=/shared-ro \
    --existing-pvc claimname=course-ee-559-shared-rw,path=/shared-rw \
    --command -- bash -c "\
        echo '=== BART inference — all 4 conditions ===' && \
        for COND in full target_only attack_only none; do \
            python3 ${CODE_ROOT}/inference/run_stage2.py \
                --condition \${COND} \
                --checkpoint_dir /scratch/hmr_stage2_phase2_\${COND}_checkpoint \
                --stage1_output_dir /scratch/hmr_stage1_output \
                --output_dir /scratch/hmr_eval_stage2_\${COND} \
                --hf_cache /scratch/hf_cache ; \
        done && \
        echo '=== Proxy pipeline (no LLaVA) ===' && \
        python3 ${CODE_ROOT}/inference/run_proxy_pipeline.py \
            --proxy_checkpoint_dir /scratch/hmr_proxy_checkpoint \
            --stage1_output_dir /scratch/hmr_stage1_output \
            --output_dir /scratch/hmr_eval_proxy \
            --hf_cache /scratch/hf_cache && \
        echo '=== LLaVA end-to-end baseline ===' && \
        python3 ${CODE_ROOT}/baselines/run_llava_baseline.py \
            --mode end_to_end \
            --stage1_output_dir /scratch/hmr_stage1_output \
            --output_dir /scratch/hmr_eval_llava_e2e \
            --hf_cache /scratch/hf_cache && \
        echo '=== LLaVA structured-prompt baseline ===' && \
        python3 ${CODE_ROOT}/baselines/run_llava_baseline.py \
            --mode structured_prompt \
            --stage1_output_dir /scratch/hmr_stage1_output \
            --output_dir /scratch/hmr_eval_llava_struct \
            --hf_cache /scratch/hf_cache && \
        echo '=== DetoxLLM text-only baseline ===' && \
        python3 ${CODE_ROOT}/baselines/run_detoxllm_baseline.py \
            --stage1_output_dir /scratch/hmr_stage1_output \
            --output_dir /scratch/hmr_eval_detoxllm \
            --hf_cache /scratch/hf_cache && \
        echo '=== Final evaluation ===' && \
        python3 ${EVAL_SCRIPT} \
            --stage2_output_dirs \
                /scratch/hmr_eval_stage2_full \
                /scratch/hmr_eval_stage2_target_only \
                /scratch/hmr_eval_stage2_attack_only \
                /scratch/hmr_eval_stage2_none \
            --proxy_output_dir /scratch/hmr_eval_proxy \
            --llava_output_dir /scratch/hmr_eval_llava_e2e \
            --llava_struct_output_dir /scratch/hmr_eval_llava_struct \
            --detoxllm_output_dir /scratch/hmr_eval_detoxllm \
            --output_dir /scratch/hmr_eval_results \
            --hf_cache /scratch/hf_cache \
    "

echo "Full evaluation job submitted."
