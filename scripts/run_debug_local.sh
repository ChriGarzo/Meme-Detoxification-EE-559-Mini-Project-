#!/usr/bin/env bash
set -e

# === Local debug script (no cluster) ===
# Sets DEBUG=1, runs full pipeline on CPU with synthetic data
# Useful for testing before submitting to RCP

export DEBUG=1
export HF_HOME="./.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "=== [DEBUG] Step 1: Filter images (skipped in debug) ==="
python3 data/preprocess/filter_meme_images.py \
    --dataset harmeme \
    --images_dir outputs/debug_run/images \
    --output_manifest outputs/debug_run/harmeme_manifest.csv \
    --save_examples outputs/debug_run/filter_examples \
    --debug

echo "=== [DEBUG] Step 2: Run Stage 1 — LLaVA explain + pseudo-rewrite ==="
python3 inference/run_stage1.py --dataset harmeme --debug

echo "=== [DEBUG] Step 3: Build Stage 2 dataset ==="
python3 data/preprocess/build_stage2_dataset.py \
    --stage1_dir outputs/debug_run/stage1 \
    --output_dir outputs/debug_run/stage2_dataset \
    --debug

echo "=== [DEBUG] Step 4: Train Stage 2 (full conditioning, with ParaDetox mixing) ==="
python3 training/train_stage2_phase2.py --condition full --debug

echo "=== [DEBUG] Step 5: Run Stage 2 inference ==="
python3 inference/run_stage2.py --condition full --debug

echo "=== [DEBUG] Step 6: Train proxy network ==="
python3 training/train_proxy.py --debug

echo "=== [DEBUG] Step 7: Run proxy pipeline inference (no LLaVA) ==="
python3 inference/run_proxy_pipeline.py --debug

echo "=== [DEBUG] Step 8: Run LLaVA baseline ==="
python3 baselines/run_llava_baseline.py --mode end_to_end --debug

echo "=== [DEBUG] Step 9: Evaluate (all systems including proxy) ==="
python3 evaluation/evaluate.py --debug

echo ""
echo "=== [DEBUG] All steps completed. If you see this, the pipeline is ready for RCP. ==="
