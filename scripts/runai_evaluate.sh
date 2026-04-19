#!/usr/bin/env bash
set -e

# =============================================================================
# Stage 3: Full evaluation (GPU A100-40G)
#
# Usage: bash scripts/runai_evaluate.sh <UID_NUMBER>
#
#   UID_NUMBER  Your numeric Unix UID. Get it with: id -u
#
# Example:
#   bash scripts/runai_evaluate.sh 123456
#
# Note: Run this AFTER all Stage 2 Phase 2 jobs and proxy training have completed.
#       Runs inference for all 4 BART conditions + proxy pipeline + baselines,
#       then computes STA/SIM/CLIPScore/RewritePrecision/J metrics.
# =============================================================================

# --- Validate args ---
if [ -z "$1" ]; then
    echo "ERROR: Missing UID_NUMBER argument."
    echo "Usage: bash $0 <UID_NUMBER>"
    echo "  Get your UID with: id -u"
    exit 1
fi

UID_NUM="$1"

# --- Configuration (do NOT edit these) ---
USERNAME="${USER}"
GROUP_NUM="31"
IMAGE="registry.rcp.epfl.ch/ee-559-garzone/hmr:v0.1"

echo "=== Stage 3: Full Evaluation ==="
echo "  User:  ${USERNAME} (UID: ${UID_NUM})"
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
            python3 /home/${USERNAME}/hateful_meme_rewriting/inference/run_stage2.py \
                --condition \${COND} \
                --checkpoint_dir /scratch/hmr_stage2_phase2_\${COND}_checkpoint \
                --stage1_output_dir /scratch/hmr_stage1_output \
                --output_dir /scratch/hmr_eval_stage2_\${COND} \
                --hf_cache /scratch/hf_cache ; \
        done && \
        echo '=== Proxy pipeline (no LLaVA) ===' && \
        python3 /home/${USERNAME}/hateful_meme_rewriting/inference/run_proxy_pipeline.py \
            --proxy_checkpoint_dir /scratch/hmr_proxy_checkpoint \
            --stage1_output_dir /scratch/hmr_stage1_output \
            --output_dir /scratch/hmr_eval_proxy \
            --hf_cache /scratch/hf_cache && \
        echo '=== LLaVA end-to-end baseline ===' && \
        python3 /home/${USERNAME}/hateful_meme_rewriting/baselines/run_llava_baseline.py \
            --mode end_to_end \
            --stage1_output_dir /scratch/hmr_stage1_output \
            --output_dir /scratch/hmr_eval_llava_e2e \
            --hf_cache /scratch/hf_cache && \
        echo '=== LLaVA structured-prompt baseline ===' && \
        python3 /home/${USERNAME}/hateful_meme_rewriting/baselines/run_llava_baseline.py \
            --mode structured_prompt \
            --stage1_output_dir /scratch/hmr_stage1_output \
            --output_dir /scratch/hmr_eval_llava_struct \
            --hf_cache /scratch/hf_cache && \
        echo '=== DetoxLLM text-only baseline ===' && \
        python3 /home/${USERNAME}/hateful_meme_rewriting/baselines/run_detoxllm_baseline.py \
            --stage1_output_dir /scratch/hmr_stage1_output \
            --output_dir /scratch/hmr_eval_detoxllm \
            --hf_cache /scratch/hf_cache && \
        echo '=== Final evaluation ===' && \
        python3 /home/${USERNAME}/hateful_meme_rewriting/evaluation/evaluate.py \
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
