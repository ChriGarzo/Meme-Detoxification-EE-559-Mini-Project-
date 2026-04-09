#!/usr/bin/env bash
set -e

# === Configuration (edit these) ===
USERNAME="garzone"
UID_NUM="<YOUR_UID>"      # Get with: ssh garzone@jumphost.rcp.epfl.ch then: id
GROUP_NUM="31"
IMAGE="registry.rcp.epfl.ch/ee-559-${USERNAME}/hmr:v0.1"

# Full evaluation (GPU A100-40G)
# Runs: BART inference for all conditions, proxy pipeline, LLaVA baselines, DetoxLLM baseline, then evaluate.py

runai submit hmr-evaluate \
    --run-as-uid ${UID_NUM} \
    --image ${IMAGE} \
    --node-pools a100-40g \
    --gpu 1 \
    --cpu-request 8 \
    --memory-request 40Gi \
    --existing-pvc claimname=home,path=/home/${USERNAME} \
    --existing-pvc claimname=course-ee-559-scratch-g${GROUP_NUM},path=/scratch \
    --existing-pvc claimname=course-ee-559-shared-ro,path=/shared-ro \
    --existing-pvc claimname=course-ee-559-shared-rw,path=/shared-rw \
    --command -- bash -c "\
        echo '=== Stage 2 BART inference for all conditions ===' && \
        python3 /home/${USERNAME}/hateful_meme_rewriting/inference/run_stage2.py \
            --condition text_only \
            --checkpoint_dir /scratch/hmr_stage2_phase2_text_only_checkpoint \
            --dataset_path /scratch/hmr_stage2_dataset \
            --output_dir /scratch/hmr_eval_stage2_text_only \
            --hf_cache /scratch/hf_cache && \
        python3 /home/${USERNAME}/hateful_meme_rewriting/inference/run_stage2.py \
            --condition image_text \
            --checkpoint_dir /scratch/hmr_stage2_phase2_image_text_checkpoint \
            --dataset_path /scratch/hmr_stage2_dataset \
            --output_dir /scratch/hmr_eval_stage2_image_text \
            --hf_cache /scratch/hf_cache && \
        python3 /home/${USERNAME}/hateful_meme_rewriting/inference/run_stage2.py \
            --condition image_embedding \
            --checkpoint_dir /scratch/hmr_stage2_phase2_image_embedding_checkpoint \
            --dataset_path /scratch/hmr_stage2_dataset \
            --output_dir /scratch/hmr_eval_stage2_image_embedding \
            --hf_cache /scratch/hf_cache && \
        python3 /home/${USERNAME}/hateful_meme_rewriting/inference/run_stage2.py \
            --condition full \
            --checkpoint_dir /scratch/hmr_stage2_phase2_full_checkpoint \
            --dataset_path /scratch/hmr_stage2_dataset \
            --output_dir /scratch/hmr_eval_stage2_full \
            --hf_cache /scratch/hf_cache && \
        echo '=== Proxy pipeline inference (no LLaVA) ===' && \
        python3 /home/${USERNAME}/hateful_meme_rewriting/inference/run_proxy_pipeline.py \
            --proxy_checkpoint_dir /scratch/hmr_proxy_checkpoint \
            --stage1_dataset_path /scratch/hmr_stage2_dataset \
            --output_dir /scratch/hmr_eval_proxy \
            --hf_cache /scratch/hf_cache && \
        echo '=== LLaVA end-to-end baseline ===' && \
        python3 /home/${USERNAME}/hateful_meme_rewriting/baselines/run_llava_baseline.py \
            --mode end_to_end \
            --stage1_output_dir /scratch/hmr_stage1_output \
            --output_dir /scratch/hmr_eval_llava_e2e \
            --hf_cache /scratch/hf_cache && \
        echo '=== DetoxLLM baseline ===' && \
        python3 /home/${USERNAME}/hateful_meme_rewriting/baselines/run_detoxllm_baseline.py \
            --images_dir /scratch/hmr_data/harmeme/images \
            --output_dir /scratch/hmr_eval_detoxllm \
            --hf_cache /scratch/hf_cache && \
        echo '=== Final evaluation ===' && \
        python3 /home/${USERNAME}/hateful_meme_rewriting/evaluation/evaluate.py \
            --stage2_output_dirs /scratch/hmr_eval_stage2_text_only /scratch/hmr_eval_stage2_image_text /scratch/hmr_eval_stage2_image_embedding /scratch/hmr_eval_stage2_full \
            --proxy_output_dir /scratch/hmr_eval_proxy \
            --llava_output_dir /scratch/hmr_eval_llava_e2e \
            --detoxllm_output_dir /scratch/hmr_eval_detoxllm \
            --output_dir /scratch/hmr_eval_results \
            --hf_cache /scratch/hf_cache \
    "

echo "Full evaluation job submitted."
