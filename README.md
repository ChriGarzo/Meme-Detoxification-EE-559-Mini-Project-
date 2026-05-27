# Explain-then-Rewrite: Leveraging Hate Explanation Generation for Targeted Meme Text Detoxification

**EE-559 Deep Learning — EPFL, Group 31**

A pipeline that uses LLaVA-Next (7B) as a teacher model to generate structured hate explanations and pseudo-rewrites, then fine-tunes BART-large (400M) directly as a lightweight student for meme text detoxification. Four conditioning ablations are evaluated: `full`, `target_only`, `visual_only`, `none`.

---

## Pipeline Overview

```
Stage 0 ──── OCR + CLIP filtering  [per dataset, GPU]
               │  EasyOCR extracts meme text (10–300 chars)
               │  CLIP filter (dataset-specific — see Preprocessing section)
               │  Outputs: /scratch/hmr_data/<dataset>/manifest.csv
               ▼
  Build Unified Splits
               │  Reads Stage 0 manifests (kept=True only)
               │  Stratified 80/10/10 split by (dataset, hateful)
               │  All 3 datasets represented in every split
               │  Outputs: unified_train.csv / unified_val.csv / unified_test.csv
               ▼
Stage 1 ──── LLaVA-Next explanations + pseudo-rewrites (sharded)  [GPU]
               │  Run on all 3 splits (train: 8 shards, val/test: 2 shards each)
               │  Each shard: explanation generation → rewrite generation → text STA + BERTScore filter
               │  All records written (passed + failed) with passed_stage1_filters flag
               │  Output: {split}_pseudo_rewrites_shard{XX}of{NN}.jsonl
               ▼
  Merge pseudo-rewrite shards (per split)
               │  Output: {split}_pseudo_rewrites_merged.jsonl  (train / val / test)
               ▼
   Build Stage 2 Dataset
               │  train.jsonl  ← from train_pseudo_rewrites_merged.jsonl (quality-filtered)
               │  val.jsonl    ← from val_pseudo_rewrites_merged.jsonl   (quality-filtered)
               │  test.jsonl   ← from test_pseudo_rewrites_merged.jsonl  (quality-filtered)
               │  Input format: "Task: rewrite the original meme text to be non-toxic while
               │    preserving the meme topic and intended meaning. Context: target group =
               │    {target}; visual evidence = {visual_evidence}; implicit harmful meaning =
               │    {meaning}. Original meme text to detoxify: {text}"
               ▼
Stage 2 ───── BART LoRA meme fine-tuning (×4 conditions in parallel)  [GPU]
               │  Conditions: full | target_only | visual_only | none
               │  LoRA: r=32, alpha=64, dropout=0.05 on q/k/v/out_proj + fc1/fc2
               │  ~17M trainable / 400M total parameters (~4.3%)
               │  5 epochs, lr=1e-4, starts directly from facebook/bart-large
               │  Validation: val.jsonl (proper held-out split from unified_val.csv)
               │  Evaluation metrics per checkpoint:
               │    ROUGE-1/2/L, collapse rate, text STA (RoBERTa),
               │    multimodal STA (VisualBERT image+text, eval-only)
               │  5 qualitative (original → generated → reference) examples logged
               │  Output: merged BART checkpoint + lora_adapter/ subdirectory
               ▼
Stage 4 ──── Proxy network training  [GPU]
               │  3-layer MLP: concat(CLIP image + CLIP text) [1536-dim]
               │    → short BART encoder-memory sequence [K × 1024]
               │  Enables VLM-free inference at deployment
               ▼
Stage 3 ──── Evaluation  [GPU]
               BART base + finetuned + DetoxLLM + Proxy — all on test.jsonl
               Metrics: text STA, BERTScore, CLIPScore, VisualBERT multimodal toxicity
```

**Models used:**
- `llava-hf/llava-v1.6-mistral-7b-hf` — Stage 1 teacher
- `facebook/bart-large` — Stage 2 student (LoRA fine-tuned, ~4.3% trainable params)
- `UBC-NLP/DetoxLLM-7B` — external text-only detoxification baseline (fair comparison target)
- `openai/clip-vit-large-patch14` — Stage 0 filter + Stage 4 proxy + Stage 2 eval visual features
- `chiragmittal92/visualbert-hateful-memes-finetuned-model` — Stage 2 multimodal STA metric (eval-only)
- `s-nlp/roberta_toxicity_classifier` — Stage 2 text STA metric

---

## Project Structure

```
hateful_meme_rewriting/
├── README.md
├── requirements.txt
│
├── docker/
│   ├── Dockerfile
│   └── requirements_docker.txt
│
├── data/
│   └── preprocess/
│       ├── filter_meme_images.py      ← Stage 0: OCR + CLIP filter (per dataset)
│       ├── sample_filter_examples.py  ← visual QC: samples 50 kept/discarded per dataset
│       ├── build_unified_splits.py    ← builds 80/10/10 splits from Stage 0 manifests
│       └── build_stage2_dataset.py    ← merges Stage 1 outputs → train/val/test JSONL
│
├── models/
│   ├── explainer.py                   ← LLaVA-Next wrapper
│   ├── rewriter.py                    ← BART wrapper (+ generate_from_formatted)
│   └── proxy.py                       ← CLIP → BART soft-token ExplanationProxy
│
├── inference/
│   ├── run_stage1_sharded.py          ← Stage 1: explanations + pseudo-rewrites (sharded)
│   ├── merge_stage1_rewrites_shards.py ← merges pseudo-rewrite shards
│   ├── run_stage2.py                  ← BART inference over filtered memes
│   └── run_proxy_pipeline.py          ← VLM-free proxy → BART-full inference
│
├── training/
│   ├── train_stage2_phase1.py         ← ParaDetox warm-up (kept for reference, not run in current pipeline)
│   ├── train_stage2_phase2.py         ← LoRA meme fine-tuning (×4 conditions, starts from bart-large directly)
│   └── train_proxy.py                 ← Proxy MLP training
│
├── evaluation/
│   ├── evaluate.py
│   └── metrics.py
│
├── baselines/
│   ├── run_llava_baseline.py          ← LLaVA structured-prompt baseline
│   └── run_detoxllm_baseline.py
│
├── analysis/
│   ├── aggregate_pipeline_co2.py      ← aggregate emissions CSVs + estimate training CO2
│   ├── benchmark_single_inference.py  ← per-model single-step latency + CO2
│   ├── recover_training_metrics.py
│   ├── compare_stage2_outputs.py
│   └── plot_proxy_training.py
│
├── utils/
│   ├── bertscore_utils.py             ← BERTScore batch helper
│   └── debug.py
│
├── configs/
│   ├── stage1_inference.yaml
│   ├── stage2_training.yaml
│   └── stage2_training_debug.yaml
│
└── scripts/
    ├── run_debug_local.sh             ← full pipeline locally (no GPU needed)
    ├── setup_scratch.sh               ← creates /scratch/ layout + downloads HarMeme
    ├── move_datasets_to_scratch.sh    ← moves MAMI/MMHS150K from home → scratch
    ├── runai_download_datasets.sh     ← RunAI wrapper for setup_scratch.sh
    ├── runai_move_datasets.sh         ← RunAI wrapper for move_datasets_to_scratch.sh
    ├── runai_stage0_filter.sh         ← Stage 0 per dataset (GPU)
    ├── runai_sample_filter_examples.sh ← QC sampling after Stage 0 (no GPU needed)
    ├── runai_build_unified_splits.sh  ← builds unified splits after Stage 0
    ├── runai_stage1_sharded.sh        ← Stage 1: explanations + rewrites, all splits (GPU)
    ├── runai_build_stage2_dataset.sh
    ├── runai_stage2_phase2.sh
    ├── runai_train_proxy.sh
    ├── runai_plot_proxy_curves.sh
    ├── runai_evaluate_all.sh          ← unified evaluation: BART + DetoxLLM + Proxy (GPU)
    ├── run_evaluate_all_job.sh        ← inner job script for runai_evaluate_all.sh
    ├── runai_pipeline_co2.sh          ← aggregate pipeline CO2 from existing CSVs (no GPU)
    └── runai_benchmark_inference.sh   ← per-model single-inference latency + CO2 (GPU)
```

---

## Local Debug Run (no GPU required)

To verify the full pipeline end-to-end on a small subset without a GPU:

```bash
bash scripts/run_debug_local.sh
```

This runs all stages on a tiny batch, outputs results to `outputs/debug_run/`, and prints:
```
[DEBUG] Pipeline complete. Results saved to outputs/debug_run/final_results.json
```

---

## Dataset Setup

### HarMeme
```bash
bash data/download_datasets.sh
```
You will be prompted to confirm dataset usage permissions.

### MAMI
Manual request required: https://forms.gle/AGWMiGicBHiQx4q98

### MMHS150K
Download from: https://gombru.github.io/2019/10/09/MMHS/

### ParaDetox
Not used in the current pipeline. Stage 2 trains exclusively on the meme pseudo-rewrite dataset produced by Stage 1. LoRA's small parameter footprint reduces overfitting risk without requiring external data mixing.

---

## Stage 0 Preprocessing

All three datasets go through the same two-stage filter in `filter_meme_images.py`, but the CLIP decision rule differs per dataset to account for their different origins.

**Stage 1 — OCR (identical for all datasets)**
EasyOCR extracts text from each image. Images with fewer than 10 or more than 300 characters are discarded. This removes images with no readable text (plain photos, blank images) and images that are mostly text (dense articles, long chat threads).

**Stage 2 — CLIP (dataset-specific)**

| Dataset | Origin | CLIP logic |
|---------|--------|------------|
| HarMeme | Curated COVID-19 meme collection | Binary: 2 prompts. Keep if `meme_score > screenshot_score`. |
| MAMI | Curated misogynous meme collection | Binary: 2 prompts. Keep if `meme_score > screenshot_score`. |
| MMHS150K | Raw Twitter posts | Multi-class: 5 prompts. Keep only if the meme prompt scores highest among all 5 **and** reaches a minimum threshold of 0.45. |

HarMeme and MAMI are curated datasets where images are already overwhelmingly memes, so the simple binary check is sufficient. MMHS150K is scraped directly from Twitter and contains a large proportion of non-meme content — plain photos of people, social media video thumbnails, and phone UI screenshots — that score close enough to the meme prompt to pass the binary filter. The stricter multi-class check uses four targeted negative prompts to catch these cases:

- `"a screenshot of a text message, tweet, or text conversation"`
- `"a screenshot of a social media video post or video thumbnail"`
- `"a plain photograph of a person or scene without any overlaid text"`
- `"a screenshot of a mobile phone or social media app interface"`

The threshold of 0.45 can be adjusted at runtime via `--mmhs150k_clip_threshold` if needed.

Each run produces a `manifest.csv` with one row per image containing the OCR text, CLIP scores, and a `kept` boolean. MMHS150K manifests additionally include `clip_best_negative` (which negative class scored highest) and `clip_threshold_used` for debugging. The manifests are then consumed by `build_unified_splits.py` to assemble the final training data.

---

## Cluster Workflow (EPFL RCP — Group 31)

### Infrastructure

| Resource | Value |
|---|---|
| Group scratch PVC | `course-ee-559-scratch-g31` mounted at `/scratch/` |
| Personal home PVC | `home` mounted at `/home/${USER}/` |
| Docker registry | `registry.rcp.epfl.ch/ee-559-garzone/hmr:v0.1` (shared, public) |
| RunAI project | `course-ee-559-<username>` |

All datasets, model checkpoints, and HuggingFace caches live on `/scratch/` (shared by all group members). Code lives in each member's personal `/home/${USER}/`. **Datasets and checkpoints only need to be produced once by any one member.**

---

### Step 1 — Docker image (each member, once)

See `docker/TUTORIAL_DOCKER_IT.md` for the full step-by-step guide in Italian.

Short version:
```bash
# On your laptop (Linux/Mac):
docker build --platform linux/amd64 -t registry.rcp.epfl.ch/ee-559-${USER}/hmr:v0.1 docker/

# On Windows PowerShell:
docker build --platform linux/amd64 -t registry.rcp.epfl.ch/ee-559-<username>/hmr:v0.1 docker/

docker login registry.rcp.epfl.ch        # GASPAR credentials
docker push registry.rcp.epfl.ch/ee-559-<username>/hmr:v0.1
```
The Harbor project `ee-559-<username>` must be created as **Public** at https://registry.rcp.epfl.ch before pushing.

---

### Step 2 — SSH into the cluster and configure RunAI

```bash
ssh <username>@jumphost.rcp.epfl.ch
runai login
runai config project course-ee-559-<username>
```

---

### Step 3 — Get your numeric UID

All RunAI scripts require your Unix UID as the first argument (needed to set file ownership inside containers):

```bash
id -u    # e.g. 123456
```

---

### Step 4 — Clone the repository

```bash
cd /home/${USER}/
git clone https://github.com/ChriGarzo/Meme-Detoxification-EE-559-Mini-Project-.git hateful_meme_rewriting
cd hateful_meme_rewriting
```

---

### Step 5 — Download and transfer datasets (once for the whole group)

**HarMeme** — downloaded automatically:
```bash
bash scripts/runai_download_datasets.sh <UID>
runai logs hmr-download-datasets -p course-ee-559-<username> --follow
```

**MAMI** — request access at https://forms.gle/AGWMiGicBHiQx4q98, then from Windows PowerShell:
```powershell
ssh <username>@jumphost.rcp.epfl.ch "mkdir -p ~/datasets_upload"
scp -r "C:\path\to\MAMI" <username>@jumphost.rcp.epfl.ch:/home/<username>/datasets_upload/
scp -r "C:\path\to\MMHS150K dataset" <username>@jumphost.rcp.epfl.ch:/home/<username>/datasets_upload/
```

Then move both from home to scratch (only accessible inside a RunAI container):
```bash
bash scripts/runai_move_datasets.sh <UID>
runai logs hmr-move-datasets -p course-ee-559-<username> --follow
```

---

### Step 6 — Run the pipeline (sequential stages)

All scripts follow the same pattern:
```bash
bash scripts/runai_<stage>.sh <UID> [optional args]
```

Your `$USER` is read automatically from the environment — you never edit the scripts.

**Stage 0 — Filter memes** (GPU; run once per dataset; outputs shared on /scratch/)
```bash
bash scripts/runai_stage0_filter.sh <UID> harmeme
bash scripts/runai_stage0_filter.sh <UID> mami
bash scripts/runai_stage0_filter.sh <UID> mmhs150k
```
Each job outputs a `manifest.csv` with OCR/CLIP scores and a `kept` flag. To visually inspect the results after all three jobs complete:
```bash
bash scripts/runai_sample_filter_examples.sh <UID>
runai logs hmr-sample-filter-examples -p course-ee-559-<username> --follow
```
This produces `/scratch/hmr_data/filtering_results/` with 50 kept and 50 discarded images per dataset. Copy it to your laptop with:
```bash
scp -r <username>@jumphost.rcp.epfl.ch:~/filtering_results/ .
```

**Build unified splits** (run after ALL three Stage 0 jobs complete)
```bash
bash scripts/runai_build_unified_splits.sh <UID>
runai logs hmr-build-unified-splits -p course-ee-559-<username> --follow
```
Reads the three Stage 0 manifests, filters to `kept=True` images only, then creates stratified 80/10/10 splits ensuring all three datasets are represented in every split. Outputs `unified_train.csv`, `unified_val.csv`, `unified_test.csv` to `/scratch/hmr_data/unified_splits/`.

**Stage 1 — LLaVA explanations + pseudo-rewrites (sharded)**
Run for all three splits. Train uses 8 shards; val and test use 2 shards each:
```bash
# Train (8 shards in parallel)
for i in $(seq 0 7); do
  SPLIT=train SHARD_ID=$i NUM_SHARDS=8 bash scripts/runai_stage1_sharded.sh
done

# Val and test (2 shards each, in parallel)
for i in 0 1; do
  SPLIT=val  SHARD_ID=$i NUM_SHARDS=2 bash scripts/runai_stage1_sharded.sh
  SPLIT=test SHARD_ID=$i NUM_SHARDS=2 bash scripts/runai_stage1_sharded.sh
done
```

**Merge pseudo-rewrite shards** (run once per split after all shards for that split finish):
```bash
for SPLIT in train val test; do
  python3 inference/merge_stage1_rewrites_shards.py \
    --dataset ${SPLIT} \
    --input_dir /mnt/course-ee-559/rcp-caas-ee-559-g31/scratch-g31/hmr_stage1_output \
    --num_shards $([ "$SPLIT" = "train" ] && echo 8 || echo 2) \
    --output_path /mnt/course-ee-559/rcp-caas-ee-559-g31/scratch-g31/hmr_stage1_output/${SPLIT}_pseudo_rewrites_merged.jsonl
done
```

**Build Stage 2 dataset** (wait for all Stage 1 jobs and merges to complete)
```bash
bash scripts/runai_build_stage2_dataset.sh <UID>
```
Produces `train.jsonl`, `val.jsonl`, and `test.jsonl` in `/scratch/hmr_stage2_dataset/`, each using the corresponding proper split from `unified_splits/`.

**Stage 2 — BART LoRA meme fine-tuning** (4 jobs submitted in parallel, one per condition)
```bash
bash scripts/runai_stage2_phase2.sh <UID>
```
Trains four separate LoRA-adapted checkpoints starting directly from `facebook/bart-large`: `full`, `target_only`, `visual_only`, `none`. Validation during training uses `val.jsonl` (proper held-out split from `unified_val.csv`).

Each job applies LoRA (r=32, alpha=64, dropout=0.05) to all attention projections and FFN layers, giving ~17M trainable parameters out of 400M total. At the end of training, the adapter is merged back into the base model and saved to the checkpoint directory. The raw LoRA adapter weights are preserved in `lora_adapter/` for potential future reuse.

Metrics tracked at every eval checkpoint: ROUGE-1/2/L, collapse rate, text STA (RoBERTa toxicity), and multimodal STA (VisualBERT with CLIP image features — images are used for this metric only and never influence gradients). Five qualitative examples (original → generated → reference) are logged at each eval step.

Note: Stage 2 Phase 1 (ParaDetox warm-up) is kept in `training/train_stage2_phase1.py` for reference but is not run in the current pipeline. Phase 2 starts directly from `facebook/bart-large`.

**Stage 4 — Train proxy network** (wait for Stage 2 Phase 2 `full` to complete)
```bash
bash scripts/runai_train_proxy.sh <UID>
```
The current proxy predicts a short sequence of BART encoder soft tokens rather than a single pooled hidden vector. By default `K=16`; try a longer memory with:
```bash
PROXY_NUM_SOFT_TOKENS=32 bash scripts/runai_train_proxy.sh <UID>
```

**Stage 3 — Full evaluation** (wait for Stage 2 Phase 2 all conditions + proxy training)
```bash
bash scripts/runai_evaluate_all.sh <UID>
```
Runs all inference and evaluation in a single job: BART base (4 conditions), finetuned BART (4 conditions), DetoxLLM, and proxy+BART — all on `test.jsonl`. Results are written to `/scratch/hmr_eval_results/`.

---

### Storage layout on `/scratch/` (shared by all group members)

```
/scratch/
├── hf_cache/                               ← HuggingFace model cache (downloaded once)
├── hmr_data/
│   ├── harmeme/
│   │   ├── images/                         ← raw images
│   │   ├── annotations/                    ← train/val/test.jsonl
│   │   └── manifest.csv                    ← Stage 0 output (OCR/CLIP scores + kept flag)
│   ├── mami/   (same structure)
│   ├── mmhs150k/  (same structure)
│   ├── filtering_results/                  ← QC output from runai_sample_filter_examples.sh
│   │   ├── kept/
│   │   └── discarded/
│   └── unified_splits/                     ← built after all Stage 0 jobs complete
│       ├── unified_train.csv
│       ├── unified_val.csv
│       ├── unified_test.csv
│       └── split_stats.json
│
├── stages/                                 ← active pipeline outputs (explicit_detox format)
│   ├── hmr_stage1_output/
│   │   ├── train_pseudo_rewrites_shard00of08.jsonl
│   │   ├── ... train_pseudo_rewrites_shard07of08.jsonl
│   │   ├── train_pseudo_rewrites_merged.jsonl
│   │   ├── val_pseudo_rewrites_shard00of02.jsonl
│   │   ├── val_pseudo_rewrites_shard01of02.jsonl
│   │   ├── val_pseudo_rewrites_merged.jsonl
│   │   ├── test_pseudo_rewrites_shard00of02.jsonl
│   │   ├── test_pseudo_rewrites_shard01of02.jsonl
│   │   └── test_pseudo_rewrites_merged.jsonl
│   ├── hmr_stage2_dataset/
│   │   ├── train.jsonl                     ← 3,578 examples (from unified_train.csv)
│   │   ├── val.jsonl                       ← 269 examples  (from unified_val.csv)
│   │   ├── test.jsonl                      ← 280 examples  (from unified_test.csv)
│   │   └── dataset_statistics.json
│   ├── hmr_stage2_phase2_full_explicit_detox_checkpoint/
│   ├── hmr_stage2_phase2_target_only_explicit_detox_checkpoint/
│   ├── hmr_stage2_phase2_visual_only_explicit_detox_checkpoint/
│   ├── hmr_stage2_phase2_none_explicit_detox_checkpoint/
│   └── hmr_proxy_checkpoint_explicit_detox/
│
├── eval_results/                           ← evaluation outputs (explicit_detox format)
│   ├── hmr_eval_results_explicit_detox/    ← final summary TSV/JSON
│   ├── hmr_eval_bart_base_{full,target_only,visual_only,none}_explicit_detox/
│   ├── hmr_eval_stage2_{full,target_only,visual_only,none}_explicit_detox/
│   ├── hmr_eval_clip_proxy_bart_full_explicit_detox/
│   └── hmr_eval_detoxllm_explicit_detox/
│
├── plots/                                  ← training and proxy plots
│   ├── hmr_training_plots_explicit_detox/
│   └── hmr_proxy_training_plots/
│
└── old_results/                            ← legacy (legacy input format); kept for reference
```

---

## Carbon Footprint and Inference Efficiency

This section documents CO2 emissions across the pipeline and the inference-time efficiency difference between the compared systems. Tracking emissions was a design goal from the start: `codecarbon` is installed in the Docker image and every inference script wraps its compute with `EmissionsTracker`. Starting from the current run, training also tracks emissions directly.

### Emission tracking setup

| Pipeline stage | Tracking method |
|---|---|
| Stage 0 — OCR + CLIP filtering | CodeCarbon tracked (`emissions_stage0_<dataset>.csv` per dataset, written by `filter_meme_images.py`) |
| Stage 1 — LLaVA explanations + pseudo-rewrites (shards) | CodeCarbon per shard (`emissions_shard*.csv`) |
| Stage 2 — BART LoRA training (4 conditions) | CodeCarbon tracked from this run onward (`emissions.csv` per condition); estimated a posteriori for the current run from `training_history.json` duration × assumed GPU power |
| Stage 3 — BART inference + evaluation | CodeCarbon per inference run (`emissions.csv` per condition) |
| Proxy inference | CodeCarbon tracked |
| DetoxLLM baseline | CodeCarbon tracked |

All runs execute on `NVIDIA A100-SXM4-40GB` at EPFL RCP (Vaud, Switzerland). The measured carbon intensity for the cluster is approximately **34–35 g CO2/kWh** (Swiss electricity grid, low-carbon hydro+nuclear mix).

### Aggregating pipeline CO2 (a posteriori, no re-run needed)

```bash
bash scripts/runai_pipeline_co2.sh <UID>
# or locally (no GPU needed):
cd hateful_meme_rewriting && python3 analysis/aggregate_pipeline_co2.py \
    --scratch_dir /scratch \
    --output_dir  /scratch/hmr_co2_summary
```

Outputs: `/scratch/hmr_co2_summary/pipeline_co2_summary.{json,tsv}`

The script reads all `emissions*.csv` files written by CodeCarbon across every pipeline stage. All stages are directly tracked.

**Pipeline CO2 summary** (measured 2026-05-08, carbon intensity 34.84 g CO2/kWh — Switzerland, Vaud, EPFL RCP):

| Stage | GPU-time | Energy (kWh) | CO2 |
|---|---:|---:|---:|
| Stage 0 — OCR + CLIP filtering (163 544 images) | 2.8 h | 0.76 | 26.7 g |
| Stage 1 — LLaVA explanations + pseudo-rewrites (train: 8 shards) | 14.1 h | 6.61 | 230.3 g |
| Stage 1 — LLaVA explanations + pseudo-rewrites (val+test: 4 shards) | 3.5 h | 1.65 | 57.5 g |
| Stage 2 — BART LoRA training (4 conditions) | 2.1 h | 0.82 | 28.4 g |
| Stage 3 — BART-base inference (4 conditions) | 52.9 min | 0.18 | 6.3 g |
| Stage 3 — BART-finetuned inference (4 conditions) | 12.3 min | 0.04 | 1.5 g |
| Proxy inference | 3.4 min | 0.01 | 0.5 g |
| DetoxLLM baseline | 7.9 min | 0.04 | 1.3 g |
| **TOTAL** | **~38 h** | **~10.11** | **~352 g** |

GPU-time is the sum of compute time across all parallel shards/jobs. Energy and CO2 are derived from this total GPU-time.

Key observation: **over 80% of the total pipeline CO2 comes from Stage 1 (LLaVA inference)**. Stage 0 filtering across 163 K images and BART fine-tuning each account for ~5–6%, and all BART inference at evaluation time less than 2%. This confirms that the teacher distillation cost is a one-time training overhead, and the deployed student (BART finetuned) is highly efficient.

### Per-model inference efficiency

To report how much faster and greener our student model is compared to the teacher and the text-only baseline:

```bash
bash scripts/runai_benchmark_inference.sh <UID>
```

This runs `analysis/benchmark_single_inference.py` on a single validation example. Each model runs 3 warmup passes (not measured) then 10 timed passes under CodeCarbon. Models are loaded and released one at a time.

**What is timed:**
- `llava_teacher`: `explain()` + `generate_rewrite()` — the full Stage 1 per-meme pipeline (two 7B forward passes)
- `detoxllm`: `detoxify()` — one 7B forward pass on text only
- `bart_finetuned`: `rewrite()` with `condition=full` — one 400M forward pass on text

**Benchmark results** (measured 2026-05-08, A100-SXM4-40GB, 10 timed passes per model):

| System | Params | Load time | Mean inference | CO2/inference | CO2/280 examples |
|---|---:|---:|---:|---:|---:|
| `llava_teacher` | 7.6B | 16.8 s | 9 376 ms | 34 340 μg | 9 615 mg |
| `detoxllm` | 6.7B | 21.4 s | 1 989 ms | 5 646 μg | 1 581 mg |
| `bart_finetuned` | 406M | 4.0 s | 122 ms | 360 μg | 101 mg |
| **Speedup BART vs LLaVA** | 19× fewer | 4.2× | **76.6×** | **95.4×** | **95.4×** |
| **Speedup BART vs DetoxLLM** | 17× fewer | 5.4× | **16.3×** | **15.7×** | **15.7×** |

BART finetuned is **76.6× faster** than LLaVA and **16.3× faster** than DetoxLLM, with CO2 per inference reduced by 95× and 16× respectively. The large gap vs LLaVA comes from running two 7B causal-LM forward passes per meme (explain + rewrite) vs a single 406M encoder-decoder pass. The gap vs DetoxLLM — both text-only at inference time — reflects the cost difference between autoregressive decoding in a 7B causal LM and a 400M seq2seq model.

---

## Current Experiment Report

This section summarizes the current state of the repository and the results obtained from the latest full pipeline run. The goal of the project is not to outperform LLaVA-Next directly. LLaVA-Next is used as a large teacher model to produce explanations and pseudo-rewrites; the real objective is to fine-tune a much smaller BART-large student so that it can detoxify meme text while preserving the intended meaning as much as possible.

### What Was Done

The implemented pipeline follows an explain-then-rewrite strategy:

1. **Filter meme-like images.** Stage 0 uses EasyOCR to extract overlaid meme text and CLIP to discard images that are not visually meme-like. This is necessary because some raw datasets, especially MMHS150K, contain many plain photos, screenshots, and social-media UI captures.

2. **Build unified splits.** The filtered HarMeme, MAMI, and MMHS150K examples are merged into stratified 80/10/10 train/validation/test splits. The split is stratified by dataset and hateful label so each split contains examples from all sources.

3. **Generate LLaVA explanations and pseudo-rewrites.** Stage 1 runs LLaVA-Next over all three splits (train, val, test) to produce structured hate explanations and pseudo-rewrites. Each explanation contains:
   - `target_group`: the protected or attacked group.
   - `visual_evidence`: image evidence relevant to the hateful meaning.
   - `implicit_meaning`: the implied hateful or harmful interpretation.
   Quality is filtered using text STA (RoBERTa) and BERTScore. All records are written with a `passed_stage1_filters` flag.

4. **Build the Stage 2 dataset from all three splits.** The Stage 2 dataset builder produces three separate files using the corresponding proper split for each:
   - `train.jsonl` from `train_pseudo_rewrites_merged.jsonl`
   - `val.jsonl` from `val_pseudo_rewrites_merged.jsonl` (used for validation during BART training)
   - `test.jsonl` from `test_pseudo_rewrites_merged.jsonl` (used for final evaluation only)

5. **Filter pseudo-rewrites before BART training.** The Stage 2 dataset builder keeps only pseudo-rewrites that pass the Stage 1 quality filters, have no parse error, have positive toxicity reduction, and are not almost identical to the source text. The current dataset statistics are:

   | Quantity | Count |
   |---|---:|
   | Loaded LLaVA rewrite rows (train) | 7,918 |
   | Dropped because Stage 1 filters failed | 3,701 |
   | Dropped because of parse errors | 560 |
   | Kept after parse/toxicity filtering | 3,657 |
   | Kept after rewrite text quality filtering | 3,578 |
   | **Train examples** | **3,578** |
   | **Validation examples** (from `unified_val.csv`) | **269** |
   | **Test examples** (from `unified_test.csv`) | **280** |

   The final training set is therefore intentionally smaller but cleaner. This is a good fit for LoRA fine-tuning because only a small fraction of BART's parameters are updated.

6. **Fine-tune BART-large with LoRA.** Four BART-large students are trained, one for each conditioning setting:

   | Condition | Encoder input format |
   |---|---|
   | `full` | `Task: rewrite the original meme text to be non-toxic while preserving the meme topic and intended meaning. Context: target group = {tg}; visual evidence = {ve}; implicit harmful meaning = {im}. Original meme text to detoxify: {text}` |
   | `target_only` | same template with `visual evidence = null; implicit harmful meaning = null` |
   | `visual_only` | same template with `target group = null; implicit harmful meaning = null` |
   | `none` | same template with `target group = null; visual evidence = null; implicit harmful meaning = null` |

   The current fine-tuning setup is:
   - Base model: `facebook/bart-large`.
   - Training data: meme pseudo-rewrites only; no ParaDetox warm-up in the current run.
   - Validation: `val.jsonl` (proper held-out split from `unified_val.csv`).
   - Epochs: 5.
   - Batch size: 8.
   - Learning rate: `1e-4`.
   - Warm-up: 50 steps.
   - Weight decay: `0.01`.
   - LoRA rank: `r=32`.
   - LoRA alpha: `64`.
   - LoRA dropout: `0.05`.
   - LoRA target modules: `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`.
   - Trainable parameters: about 17M out of about 400M total, roughly 4%.

7. **Train a proxy model.** The repo also contains a proxy stage. The proxy is a small MLP trained from CLIP image/text features to BART encoder-memory soft tokens. Its purpose is to support a future VLM-free deployment path. In the current project, the main reported comparison is still the BART fine-tuning/evaluation pipeline.

   Proxy inference evaluates the intended final sequence directly:
   CLIP image/text embeddings are concatenated, the proxy predicts a short sequence of `full` BART encoder hidden states, and the `full` BART decoder generates the rewrite from that predicted encoder memory. This avoids using LLaVA explanations at inference time while still testing whether the proxy can provide useful context for the fine-tuned BART decoder.

8. **Evaluate all systems on the held-out test set.** The primary comparison is:
   - `llava_teacher`: the LLaVA pseudo-rewrite target from `hmr_stage2_dataset/test.jsonl`. Upper bound — LLaVA is both teacher and label source.
   - `detoxllm`: `UBC-NLP/DetoxLLM-7B`, a 7B causal LM purpose-trained for text detoxification. Text-only; does not use images. This is the main external baseline because it was trained to do the same task, making it a fair point of comparison for our fine-tuned BART.
   - `bart_finetuned_*`: the four LoRA-finetuned BART models. Our contribution.
   - `bart_base_*`: non-finetuned `facebook/bart-large` under the four input conditions. Included as an internal ablation to demonstrate the effect of fine-tuning, **not** as a competitive baseline.

   All systems are evaluated on the same 280 held-out test examples from `unified_test.csv`. The evaluation uses the exact model outputs as generated. The code does not sanitize or post-process rewrites before scoring. The only truncation is internal to CLIPScore because CLIP has a hard 77-token text limit.

### Evaluation Outputs

The main evaluation artifacts are written to:

```
/scratch/hmr_eval_results/
├── evaluate.log
├── evaluation_results.json
├── evaluation_summary.json
└── evaluation_summary.tsv
```

The compact TSV is the easiest file to inspect. Its columns mean:

| Column | Meaning |
|---|---|
| `system` | Model/system being evaluated. |
| `n` | Number of evaluated test examples. |
| `valid_images` | Number of examples with valid image paths for image-based metrics. |
| `text_sta` | Mean non-toxic probability from `s-nlp/roberta_toxicity_classifier`; higher is better. |
| `text_sta_delta` | Rewrite text STA minus original text STA; positive means the rewrite is less toxic than the original. |
| `sim` | BERTScore F1 between original text and rewrite; higher means better meaning preservation. |
| `clip` | CLIPScore between image and rewrite; higher means better image-text alignment. |
| `visualbert_hate_prob` | Mean multimodal hate probability from VisualBERT. |
| `visualbert_sta` | Fraction of examples VisualBERT predicts as non-hateful. |
| `visualbert_hate_drop` | Original VisualBERT hate probability minus rewrite hate probability; positive means lower predicted hate after rewriting. |

### Current Test Results

Latest `evaluation_summary.tsv` — all systems on the same 280 held-out test examples from `unified_test.csv`. BART models trained with the `explicit_detox` input format.

| System | n | Text STA | Text STA Delta | SIM | CLIP | VisualBERT Hate Prob | VisualBERT STA | VisualBERT Hate Drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `llava_teacher` | 280 | 0.9903 | 0.2342 | 0.4722 | 0.6357 | 0.6511 | 0.0000 | -0.0053 |
| `detoxllm` | 280 | 0.9400 | 0.1839 | 0.4433 | 0.6337 | 0.6489 | 0.0000 | -0.0031 |
| `bart_finetuned_full` | 280 | 0.9588 | 0.2027 | 0.3395 | 0.6280 | 0.6505 | 0.0000 | -0.0047 |
| `bart_finetuned_target_only` | 280 | 0.9643 | 0.2082 | 0.3422 | 0.6282 | 0.6518 | 0.0000 | -0.0060 |
| `bart_finetuned_visual_only` | 280 | 0.9502 | 0.1941 | 0.3301 | 0.6288 | 0.6521 | 0.0000 | -0.0063 |
| `bart_finetuned_none` | 280 | 0.9508 | 0.1947 | 0.3308 | 0.6287 | 0.6512 | 0.0000 | -0.0054 |
| `clip_proxy_bart_full` | 280 | 0.8787 | 0.1225 | 0.6004 | 0.6395 | 0.6488 | 0.0000 | -0.0030 |

### Multi-Seed Variance Analysis

The multi-seed ablation aggregate is written to:

```
/scratch/eval_results/hmr_multiseed_explicit_detox/aggregate/metric_variance_summary.tsv
```

It summarizes 5 complete seeds (`1 2 3 4 5`) for each evaluated system on the same 280 held-out test examples. All rows have `n = 5` and `expected_seeds = 5`, so the aggregate is complete. Values below are reported as mean +/- sample standard deviation across seeds.

| System | Text STA | Text STA Delta | SIM | CLIP |
|---|---:|---:|---:|---:|
| `llava_teacher` | 0.9903 +/- 0.0000 | 0.2342 +/- 0.0000 | 0.4722 +/- 0.0000 | 0.6357 +/- 0.0000 |
| `bart_finetuned_full` | 0.9634 +/- 0.0058 | 0.2073 +/- 0.0058 | 0.3304 +/- 0.0129 | 0.6284 +/- 0.0009 |
| `bart_finetuned_target_only` | 0.9597 +/- 0.0053 | 0.2036 +/- 0.0053 | 0.3350 +/- 0.0132 | 0.6285 +/- 0.0016 |
| `bart_finetuned_visual_only` | 0.9604 +/- 0.0040 | 0.2043 +/- 0.0040 | 0.2841 +/- 0.0799 | 0.6244 +/- 0.0071 |
| `bart_finetuned_none` | 0.9567 +/- 0.0132 | 0.2006 +/- 0.0132 | 0.3250 +/- 0.0302 | 0.6277 +/- 0.0015 |
| `clip_proxy_bart_full_explicit_detox` | 0.8844 +/- 0.0145 | 0.1282 +/- 0.0145 | 0.5806 +/- 0.0200 | 0.6380 +/- 0.0015 |

The main conclusion is that the finetuned BART models are stable for textual detoxification. Across the four BART ablations, mean `text_sta` stays between 0.9567 and 0.9634, and mean `text_sta_delta` stays between 0.2006 and 0.2073. The `full` condition has the best average detoxification among BART runs, while `target_only` is very close and has the best average semantic similarity among the BART-only ablations.

The conditioning ablation is clearest in the variance, not just the mean. `full` and `target_only` are the most reliable BART settings: their SIM standard deviations are low (`0.0129` and `0.0132`) and their CLIPScore is nearly unchanged across seeds. `none` is weaker because it has larger seed sensitivity in both text detoxification and semantic preservation. `visual_only` is the least reliable: seed 3 drops to SIM `0.1437` and CLIP `0.6119`, which pulls the five-seed mean down and produces the largest variance. Visual evidence alone therefore appears insufficient as a stable conditioning signal for the text rewrite task.

The proxy model shows a different trade-off. It has the highest mean SIM (`0.5806`) and highest mean CLIPScore (`0.6380`), even above the repeated LLaVA teacher target on those metrics, but it has much lower detoxification strength (`text_sta = 0.8844`, `text_sta_delta = 0.1282`). This supports the interpretation that the proxy preserves the original text and image alignment well, but is less aggressive at removing toxicity.

The LLaVA teacher has zero variance because the same teacher pseudo-rewrites are reused as the reference target for every seed. It should be read as a fixed upper-bound target, not as a trained multi-seed model. VisualBERT remains uninformative in this aggregate: `visualbert_sta` is `0.0000` for every system and every seed, and the tiny hate-probability changes do not alter the earlier caveat that this metric is not a reliable pass/fail detoxification signal here.

### Main Findings

#### Fine-tuning ablation (BART base → BART finetuned)

BART base is included as an internal ablation, not as a competitive baseline. Its results confirm that LoRA fine-tuning is necessary: without it, BART produces semantically disconnected outputs even when the text is superficially non-toxic.

Compared with non-finetuned BART, the finetuned BART models move from negative SIM scores to clearly positive SIM scores:

| Condition | Base BART SIM | Finetuned BART SIM | Change |
|---|---:|---:|---:|
| `full` | -0.0609 | 0.3395 | +0.4004 |
| `target_only` | -0.0609 | 0.3422 | +0.4031 |
| `visual_only` | -0.0609 | 0.3301 | +0.3910 |
| `none` | -0.0609 | 0.3308 | +0.3917 |

CLIPScore also improves in every condition:

| Condition | Base BART CLIP | Finetuned BART CLIP | Change |
|---|---:|---:|---:|
| `full` | 0.6218 | 0.6280 | +0.0062 |
| `target_only` | 0.6218 | 0.6282 | +0.0064 |
| `visual_only` | 0.6218 | 0.6288 | +0.0070 |
| `none` | 0.6218 | 0.6287 | +0.0069 |

Textual detoxification remains high after fine-tuning:

| Condition | Base BART Text STA | Finetuned BART Text STA |
|---|---:|---:|
| `full` | 0.9730 | 0.9588 |
| `target_only` | 0.9730 | 0.9643 |
| `visual_only` | 0.9730 | 0.9502 |
| `none` | 0.9730 | 0.9508 |

The `full` condition improves both text STA and faithfulness over base BART. The other finetuned conditions maintain strong detoxification while substantially improving semantic similarity. For the project goal this trade-off is acceptable: a model that keeps `text_sta` around 0.95–0.96 while greatly improving semantic similarity is more useful than a model that produces generic safe text unrelated to the intended rewrite.

#### External comparison: finetuned BART vs DetoxLLM

DetoxLLM (`UBC-NLP/DetoxLLM-7B`) is the primary external baseline. It was explicitly trained for text detoxification on the ParaDetox parallel corpus, making it a fair competitor for our fine-tuned BART models. Neither system uses the meme image at inference time.

Results (280 test examples):

| Metric | DetoxLLM | BART finetuned `full` | BART finetuned `target_only` |
|---|---:|---:|---:|
| Text STA | 0.9400 | **0.9588** | **0.9643** |
| Text STA Delta | 0.1839 | **0.2027** | **0.2082** |
| SIM | **0.4433** | 0.3395 | 0.3422 |
| CLIPScore | **0.6337** | 0.6280 | 0.6282 |

Key findings:

- **Text STA**: all finetuned BART conditions (0.950–0.964) outperform DetoxLLM (0.940). DetoxLLM was trained on broad web-text detoxification and struggles with meme-style OCR captions — it produces exact copies for a fraction of inputs, which drags down its average toxicity reduction.
- **SIM**: DetoxLLM scores higher (0.443 vs 0.330–0.342). This is partly a consequence of copy behaviour — verbatim copies produce artificially high semantic similarity — rather than genuine meaning preservation in rewrites.
- **CLIPScore**: DetoxLLM edges out finetuned BART slightly (0.634 vs 0.628–0.629). Neither system uses the image at inference time; both inherit image-text alignment from the training distribution.

The LLaVA teacher remains the reference upper bound for text STA and SIM. This is expected: LLaVA is a much larger vision-language model and also produced the pseudo-labels. The relevant question for the student is not "does BART beat LLaVA?" but "does our multimodally-grounded fine-tuning produce better rewrites than a text-only detoxification system of similar deployment cost?"

#### Proxy network: VLM-free inference

The proxy network (CLIP image/text → BART encoder soft tokens) allows running the pipeline without LLaVA at inference time. Evaluated on the same 280 held-out test examples:

| Metric | LLaVA teacher | Proxy + BART full |
|---|---:|---:|
| Text STA | 0.9903 | 0.8787 |
| Text STA Delta | 0.2342 | 0.1225 |
| SIM | 0.4722 | **0.6004** |
| CLIPScore | 0.6357 | **0.6395** |

The proxy achieves significantly higher SIM (0.600 vs 0.472) and marginally better CLIPScore than the LLaVA teacher, at the cost of lower text STA (0.879 vs 0.990). This suggests the proxy learns to produce rewrites that are semantically close to the original text and well image-aligned, but is less aggressive at detoxification than the full LLaVA pipeline. The proxy is a useful deployment alternative when a 7B VLM is unavailable, trading some detoxification strength for faster, lighter inference.

### Important Caveat About VisualBERT

The VisualBERT multimodal metric should be interpreted cautiously. `visualbert_sta` is `0.0000` for every system, including the LLaVA teacher. This likely means the VisualBERT classifier is too strict, miscalibrated for rewritten text, or not well matched to this evaluation setup. The hate probability changes are directionally interesting, but the absolute non-hateful prediction rate is not reliable enough to be a main conclusion.

The safer conclusion is:

- Text toxicity metrics show that rewrites are less toxic.
- SIM and CLIPScore show that fine-tuning improves meaning preservation and image-text alignment.
- VisualBERT does not currently provide a useful absolute pass/fail multimodal detox score for this test set.

### Training Behavior

Training curves can be recovered or plotted with:

```bash
bash scripts/runai_plot_curves.sh <UID>
```

The current plots are stored in:

```
/scratch/hmr_training_plots/
├── phase2_loss_curves.png
├── phase2_rouge_curves.png
├── phase2_sta_curves.png
├── phase2_text_toxicity_drop_curves.png
├── phase2_multimodal_hate_prob_curves.png
├── phase2_multimodal_sta_curves.png
├── phase2_copy_rate_high_curves.png
├── phase2_detox_quality_curves.png
└── all_phases_summary.png
```

The plotting script also writes `.pdf` and `.svg` versions. Use the `.pdf`
files in the LaTeX poster to keep text and curves vector-sharp.

Proxy training curves can be plotted separately with:

```bash
bash scripts/runai_plot_proxy_curves.sh <UID>
```

The proxy plots are stored in:

```
/scratch/hmr_proxy_training_plots/
├── proxy_loss_curves.png
├── proxy_generalization_gap.png
└── proxy_training_summary.png
```

The training logs show the expected behavior:

- Training loss decreases strongly for every condition.
- Validation text STA increases during training.
- Text toxicity drop improves during training.
- ROUGE stays relatively stable instead of collapsing.
- Copy rate is monitored to detect outputs that are too close to the original.
- VisualBERT STA stays flat at 0, reinforcing the caveat above.

For the `full` condition, validation loss progression across checkpoints:

| Step | Val Loss |
|---:|---:|
| 200 | 1.4025 |
| 800 | 1.2254 |
| 1600 | 1.1890 (best) |
| 2240 | 1.1948 |

This confirms that the model continues improving throughout training without diverging, and the best checkpoint is saved automatically.

### Final Interpretation

The current finetuning is useful for the intended scope. It distills part of LLaVA's rewriting behavior into a much smaller BART-large model. The student does not reach the teacher's quality, but it substantially improves over non-finetuned BART in the metrics that matter most for a small detoxification model:

- It keeps textual toxicity low.
- It greatly improves semantic faithfulness.
- It improves image-text alignment.
- It avoids obvious collapse.
- It can be evaluated under multiple conditioning ablations.

The comparison with DetoxLLM is the main external validity check. Both are inference-time text-only models; the difference is that ours was fine-tuned on meme-specific rewrites grounded by visual explanations from LLaVA, while DetoxLLM was trained on general-purpose text detoxification data. Our model outperforms DetoxLLM on text STA across all conditioning conditions, confirming the benefit of meme-specific training even without image access at inference time. DetoxLLM scores higher on SIM, partly due to copy behavior inflating its semantic similarity score.

The most interesting next steps are:

1. Analyse the SIM vs text STA trade-off more carefully, particularly whether DetoxLLM's higher SIM reflects genuine meaning preservation or copy behavior.
2. Improve or replace the multimodal toxicity metric, because the current VisualBERT score is not sufficiently informative.
3. Inspect qualitative outputs per condition to understand why `none`, `visual_only`, and `full` are close in final performance.
4. Consider stronger conditioning usage, for example by training a single multi-condition model with explicit condition dropout rather than four fully separate models.

---

## Citations

- Pramanick et al. (2021). HarMeme: A Dataset for Hate Speech Detection in Memes. *EMNLP*.
- Fersini et al. (2022). SemEval-2022 Task 5: Multimedia Automatic Misogyny Identification. *SemEval*.
- Gomez et al. (2020). Exploring Hate Speech Detection in Multimodal Publications. *CVPRW*.
- Logacheva et al. (2022). ParaDetox: Detoxification with Parallel Data. *ACL*.
- Liu et al. (2024). LLaVA-1.6: Improved Baselines with Visual Instruction Tuning. *arXiv:2401.00774*.
- Hallinan et al. (2023). DetoxLLM: An LLM-based text detoxification system. UBC-NLP.
