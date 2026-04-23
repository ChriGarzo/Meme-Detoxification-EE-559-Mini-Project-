#!/usr/bin/env bash
set -e

# =============================================================================
# Stage 2: BART conditioning fine-tune, all 4 conditions (GPU A100-40G)
#
# Usage:
#   bash scripts/runai_stage2_phase2.sh <UID_NUMBER>   # personal home code path
#   bash scripts/runai_stage2_phase2.sh                # scratch code path
#
#   If UID_NUMBER is provided:
#     - uses code from /home/${USER}/hateful_meme_rewriting
#     - uses the provided UID for --run-as-uid
#
#   If UID_NUMBER is omitted:
#     - uses code from /scratch/hateful_meme_rewriting
#     - auto-detects UID with `id -u` for --run-as-uid
#
# Note: Run this AFTER runai_build_stage2_dataset.sh has completed.
#       Submits 4 jobs (one per conditioning condition).
#       Conditions: full | target_only | attack_only | none
#         full        — uses all explanation fields: target_group + attack_type + implicit_meaning
#         target_only — uses only the target_group field
#         attack_only — uses only the attack_type field
#         none        — no explanation conditioning (text-only baseline)
#
#       ParaDetox mixing: 20% of each training set is drawn from
#       s-nlp/paradetox (clean toxic→neutral pairs) to provide a
#       detoxification prior without a separate warm-up phase.
#       Set --paradetox_mix_ratio 0.0 to disable.
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
# If UID is provided -> run code from personal home.
# If UID is omitted  -> run code from shared scratch path.
if [ -n "$1" ]; then
    UID_NUM="$1"
    CODE_ROOT="/home/${USERNAME}/hateful_meme_rewriting"
    MODE_LABEL="home"
else
    UID_NUM="$(id -u)"
    CODE_ROOT="/scratch/hateful_meme_rewriting"
    MODE_LABEL="scratch"
fi

TRAIN_SCRIPT="${CODE_ROOT}/training/train_stage2_phase2.py"
if [ ! -f "${TRAIN_SCRIPT}" ]; then
    if [ "${MODE_LABEL}" = "scratch" ] && [ -f "${REPO_ROOT_LOCAL}/training/train_stage2_phase2.py" ]; then
        echo "Note: /scratch path not visible on this node; using local repo check at ${REPO_ROOT_LOCAL}."
    else
        echo "ERROR: Training script not found at: ${TRAIN_SCRIPT}"
        echo "Check that the repository exists at ${CODE_ROOT}."
        exit 1
    fi
fi

# Ablation conditions (must match MemeRewriter.format_input and run_stage2.py)
CONDITIONS=("full" "target_only" "attack_only" "none")

echo "=== Stage 2: BART Conditioning Fine-tune (with ParaDetox mixing) ==="
echo "  User:       ${USERNAME} (UID: ${UID_NUM})"
echo "  Mode:       ${MODE_LABEL}"
echo "  Code root:  ${CODE_ROOT}"
echo "  Group:      ${GROUP_NUM}"
echo "  Conditions: ${CONDITIONS[*]}"
echo "  Image:      ${IMAGE}"
echo ""

for CONDITION in "${CONDITIONS[@]}"; do
    SAFE_CONDITION="${CONDITION//_/-}"
    echo "Submitting Stage 2 for condition: ${CONDITION}"

    runai submit hmr-stage2-phase2-${SAFE_CONDITION} \
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
        --command -- python3 ${TRAIN_SCRIPT} \
            --condition ${CONDITION} \
            --dataset_dir /scratch/hmr_stage2_dataset \
            --output_dir /scratch/hmr_stage2_phase2_${CONDITION}_checkpoint \
            --hf_cache /scratch/hf_cache \
            --num_train_epochs 5 \
            --per_device_train_batch_size 8 \
            --learning_rate 1e-5 \
            --warmup_steps 200 \
            --weight_decay 0.01 \
            --paradetox_mix_ratio 0.2 \
            --seed 42
done

echo ""
echo "All Stage 2 jobs submitted."
