#!/usr/bin/env bash
# =============================================================================
# Stage 2 full pipeline: Phase 1 (ParaDetox warm-up) -> Phase 2 (all 4
# conditions) running sequentially on a single GPU node.
#
# This script is meant to be launched via runai_stage2_pipeline.sh and runs
# inside the cluster job.  Do NOT run it directly on the login node.
# =============================================================================
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_CACHE="${HF_CACHE:-/scratch/hf_cache}"
DATASET_DIR="${DATASET_DIR:-/scratch/hmr_stage2_dataset}"
PHASE1_OUT="/scratch/hmr_stage2_phase1_checkpoint"

echo ""
echo "======================================================================"
echo "  Stage 2 full pipeline"
echo "  Repo:        ${REPO_DIR}"
echo "  HF cache:    ${HF_CACHE}"
echo "  Dataset dir: ${DATASET_DIR}"
echo "  Phase 1 out: ${PHASE1_OUT}"
echo "======================================================================"
echo ""

cd "${REPO_DIR}"

# ------------------------------------------------------------
# Phase 1 — ParaDetox warm-up
# Teaches BART the detoxification task and the [T:]/[A:]/[M:]
# conditioning prefix format before seeing any meme data.
# ------------------------------------------------------------
echo ">>> PHASE 1: ParaDetox warm-up"
python3 training/train_stage2_phase1.py \
    --output_dir      "${PHASE1_OUT}" \
    --hf_cache        "${HF_CACHE}" \
    --num_train_epochs            3 \
    --per_device_train_batch_size 16 \
    --learning_rate   3e-5 \
    --warmup_steps    500 \
    --weight_decay    0.01 \
    --bertscore_min   0.5 \
    --seed            42

echo ""
echo ">>> Phase 1 complete. Checkpoint: ${PHASE1_OUT}"
echo ""

# ------------------------------------------------------------
# Phase 2 — Meme conditioning fine-tune (4 ablation conditions)
#
# Changes vs. the old phase2-only run:
#   --phase1_checkpoint_dir  start from the warm-up checkpoint
#   --warmup_steps 200       longer warmup to prevent forgetting
#   --paradetox_mix_ratio 0  Phase 1 already provides the detox prior
# ------------------------------------------------------------
for CONDITION in full target_only attack_only none; do
    PHASE2_OUT="/scratch/hmr_stage2_phase2_${CONDITION}_checkpoint"
    echo ">>> PHASE 2 [${CONDITION}]"

    python3 training/train_stage2_phase2.py \
        --condition               "${CONDITION}" \
        --phase1_checkpoint_dir   "${PHASE1_OUT}" \
        --dataset_dir             "${DATASET_DIR}" \
        --output_dir              "${PHASE2_OUT}" \
        --hf_cache                "${HF_CACHE}" \
        --num_train_epochs            5 \
        --per_device_train_batch_size 8 \
        --learning_rate           2e-5 \
        --warmup_steps            200 \
        --weight_decay            0.01 \
        --paradetox_mix_ratio     0.0 \
        --seed                    42

    echo ""
    echo ">>> Phase 2 [${CONDITION}] complete. Checkpoint: ${PHASE2_OUT}"
    echo ""
done

echo "======================================================================"
echo "  Full Stage 2 pipeline COMPLETE."
echo "  Phase 1 checkpoint : ${PHASE1_OUT}"
echo "  Phase 2 checkpoints: /scratch/hmr_stage2_phase2_{full,target_only,attack_only,none}_checkpoint"
echo "======================================================================"
