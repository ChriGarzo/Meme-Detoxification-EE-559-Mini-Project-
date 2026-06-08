# From Censorship to Detoxification: Rewriting Harmful Memes

**EE-559 Deep Learning — EPFL, Group 31**  
Alex Dell'Orto · Christian Garzone · Dario Liuzzo

Instead of detecting and removing hateful memes, we study **text detoxification**: given a meme image and caption, produce a safer caption that preserves the topic and visual coherence. A LLaVA-NeXT teacher generates structured hate explanations and pseudo-rewrites; a LoRA-adapted BART-large student is distilled from these labels. A CLIP Proxy + BART path is introduced for VLM-free deployment.

---

## Pipeline

```
Stage 0 ── OCR + CLIP filtering
             │  EasyOCR keeps images with 10–300 chars of overlaid text.
             │  CLIP removes non-meme images: binary rule for HarMeme/MAMI,
             │  5-prompt multi-class rule (threshold 0.45) for MMHS150K.
             │  Output: manifest.csv per dataset
             ▼
          Build unified splits (80/10/10, stratified by dataset × hateful label)
             │  3 datasets, all splits: unified_train / unified_val / unified_test
             ▼
Stage 1 ── LLaVA-NeXT teacher (sharded; run on all 3 splits)
             │  Per meme: structured explanation (target group, visual evidence,
             │  implicit meaning) + pseudo-detoxified rewrite.
             │  Quality filter: text STA, BERTScore, toxicity-drop check.
             ▼
          Build Stage 2 dataset
             │  Keeps rows with valid parse, positive toxicity reduction,
             │  and source/rewrite similarity ≤ 0.95.
             │  3,578 train · 269 val · 280 test examples
             ▼
Stage 2 ── BART-large LoRA fine-tuning (4 conditions in parallel)
             │  Conditions: full | target_only | visual_only | none
             │  Encoder input: explicit detoxification instruction + original text
             │  + optional teacher fields (target group, visual evidence, meaning).
             │  LoRA: r=32, α=64, dropout=0.05 on attention + FFN projections.
             │  ~17M / ~400M trainable parameters (~4.3%).
             │  5 epochs, lr=1e-4, batch size=8, warm-up=50 steps.
             ▼
Stage 3 ── CLIP Proxy training (VLM-free deployment path)
             │  3-layer MLP: CLIP image + text embeddings (1536-dim)
             │  → K=16 BART encoder soft tokens (1024-dim each).
             │  Supervision: pooled encoder memory of BART-FT full.
             │  At inference: soft tokens are prepended to a null-context
             │  BART encoder input and decoded by the fine-tuned BART decoder.
             ▼
Evaluation — all systems on 280 held-out test memes
             Text STA · STA Δ · BERTScore F1 · CLIPScore
```

**Models:**

| Role | Model |
|---|---|
| Stage 1 teacher | `llava-hf/llava-v1.6-mistral-7b-hf` |
| Stage 2 student | `facebook/bart-large` (LoRA fine-tuned) |
| External baseline | `UBC-NLP/DetoxLLM-7B` |
| Stage 0 filter + Stage 3 proxy | `openai/clip-vit-large-patch14` |
| Text STA metric | `s-nlp/roberta_toxicity_classifier` |

---

## Project Structure

```
hateful_meme_rewriting/
├── data/
│   └── preprocess/
│       ├── filter_meme_images.py        ← Stage 0: OCR + CLIP filter per dataset
│       ├── build_unified_splits.py      ← 80/10/10 stratified splits across datasets
│       ├── build_stage2_dataset.py      ← Stage 1 outputs → train/val/test JSONL
│       └── sample_filter_examples.py    ← visual QC: sample kept/discarded images
│
├── models/
│   ├── explainer.py                     ← LLaVA-NeXT wrapper (explanation + rewrite prompts)
│   ├── rewriter.py                      ← BART wrapper (generate_from_formatted)
│   └── proxy.py                         ← CLIP → BART soft-token proxy (MLP + inference path)
│
├── inference/
│   ├── run_stage1_sharded.py            ← Stage 1: LLaVA explanations + rewrites (sharded)
│   ├── merge_stage1_explanations_shards.py ← merge Stage 1 explanation shards
│   ├── merge_stage1_rewrites_shards.py  ← merge Stage 1 pseudo-rewrite shards
│   ├── run_stage2.py                    ← BART conditioned inference over the test set
│   └── run_proxy_pipeline.py            ← CLIP Proxy + BART inference (no LLaVA at runtime)
│
├── training/
│   ├── train_stage2.py                  ← BART LoRA fine-tuning (4 conditions)
│   └── train_proxy.py                   ← CLIP proxy MLP training (MSE on encoder memory)
│
├── evaluation/
│   ├── evaluate.py                      ← unified evaluation runner (all systems)
│   └── metrics.py                       ← Text STA, BERTScore, CLIPScore helpers
│
├── baselines/
│   ├── run_llava_baseline.py            ← direct LLaVA structured-prompt rewriting baseline
│   └── run_detoxllm_baseline.py         ← DetoxLLM-7B inference wrapper
│
├── analysis/
│   ├── recover_training_metrics.py      ← Stage 2 + Stage 3 training curve plots
│   ├── compare_stage2_outputs.py        ← side-by-side output comparison across conditions
│   ├── aggregate_pipeline_co2.py        ← aggregate CodeCarbon CSVs across all stages
│   └── benchmark_single_inference.py   ← per-model latency + CO2 (warmup + timed passes)
│
├── docker/
│   └── Dockerfile                       ← image used for all RunAI jobs
│
└── scripts/
    ├── runai_stage0_filter.sh           ← Stage 0: OCR + CLIP filter (GPU, per dataset)
    ├── runai_build_unified_splits.sh    ← build unified splits after Stage 0
    ├── runai_stage1_sharded.sh          ← Stage 1: LLaVA teacher, all splits (GPU)
    ├── runai_build_stage2_dataset.sh    ← build train/val/test JSONL after Stage 1
    ├── runai_stage2.sh                  ← Stage 2: BART LoRA fine-tuning, 4 conditions (GPU)
    ├── runai_train_proxy.sh             ← Stage 3: proxy MLP training (GPU)
    ├── runai_evaluate_all.sh            ← evaluation: all systems on test set (GPU)
    ├── runai_plot_curves.sh             ← Stage 2 plots → training_plots/stage_2_training_plots/
    ├── runai_plot_proxy_curves.sh       ← Stage 3 plots → training_plots/stage_3_training_plots/
    ├── runai_pipeline_co2.sh            ← aggregate pipeline CO2 from existing CSVs (CPU)
    └── runai_benchmark_inference.sh     ← per-model inference latency + CO2 (GPU)
```

---

## Dataset Setup

| Dataset | Access | Scope |
|---|---|---|
| HarMeme | `bash data/download_datasets.sh` (automatic) | ~3,500 COVID-19 / US-politics memes |
| MAMI | Manual request: https://forms.gle/AGWMiGicBHiQx4q98 | 10,000 misogynist memes |
| MMHS150K | https://gombru.github.io/2019/10/09/MMHS/ | ~150,000 Twitter image–text posts |

---

## Cluster Workflow (EPFL RCP — Group 31)

**Infrastructure**

| Resource | Value |
|---|---|
| Group scratch PVC | `course-ee-559-scratch-g31` → `/scratch/` |
| Docker image | `registry.rcp.epfl.ch/ee-559-garzone/hmr:v0.1` |
| RunAI project | `course-ee-559-<username>` |

All datasets, checkpoints, and HuggingFace caches live on `/scratch/` (shared). Code lives in each member's `/home/${USER}/hateful_meme_rewriting`.

**One-time setup**

```bash
ssh <username>@jumphost.rcp.epfl.ch
runai login && runai config project course-ee-559-<username>
cd /home/${USER} && git clone <repo> hateful_meme_rewriting

# Download HarMeme; upload MAMI and MMHS150K manually, then move to scratch:
bash scripts/runai_download_datasets.sh
bash scripts/runai_move_datasets.sh
```

**Running the pipeline**

```bash
# Stage 0 — filter memes (once per dataset)
bash scripts/runai_stage0_filter.sh harmeme
bash scripts/runai_stage0_filter.sh mami
bash scripts/runai_stage0_filter.sh mmhs150k

# Build unified splits (after all Stage 0 jobs finish)
bash scripts/runai_build_unified_splits.sh

# Stage 1 — LLaVA teacher (train: 8 shards; val/test: 2 shards each)
for i in $(seq 0 7); do SPLIT=train SHARD_ID=$i NUM_SHARDS=8 bash scripts/runai_stage1_sharded.sh; done
for i in 0 1; do
  SPLIT=val  SHARD_ID=$i NUM_SHARDS=2 bash scripts/runai_stage1_sharded.sh
  SPLIT=test SHARD_ID=$i NUM_SHARDS=2 bash scripts/runai_stage1_sharded.sh
done

# Stage 1 post-processing — merge shards after all Stage 1 jobs finish.
# Stage 1 writes shards to STAGE1_SHARD_DIR; Stage 2 reads merged files from STAGE1_MERGED_DIR.
STAGE1_SHARD_DIR=/scratch/hmr_stage1_output
STAGE1_MERGED_DIR=/scratch/stages/hmr_stage1_output
mkdir -p "${STAGE1_MERGED_DIR}"
python inference/merge_stage1_explanations_shards.py \
  --dataset train --num_shards 8 \
  --input_dir "${STAGE1_SHARD_DIR}" \
  --output_path "${STAGE1_MERGED_DIR}/train_explanations_merged.jsonl"
python inference/merge_stage1_rewrites_shards.py \
  --dataset train --num_shards 8 \
  --input_dir "${STAGE1_SHARD_DIR}" \
  --output_path "${STAGE1_MERGED_DIR}/train_pseudo_rewrites_merged.jsonl"
for split in val test; do
  python inference/merge_stage1_explanations_shards.py \
    --dataset "$split" --num_shards 2 \
    --input_dir "${STAGE1_SHARD_DIR}" \
    --output_path "${STAGE1_MERGED_DIR}/${split}_explanations_merged.jsonl"
  python inference/merge_stage1_rewrites_shards.py \
    --dataset "$split" --num_shards 2 \
    --input_dir "${STAGE1_SHARD_DIR}" \
    --output_path "${STAGE1_MERGED_DIR}/${split}_pseudo_rewrites_merged.jsonl"
done

# Build Stage 2 dataset (after all Stage 1 shards and merges finish)
bash scripts/runai_build_stage2_dataset.sh

# Stage 2 — BART LoRA fine-tuning (4 conditions submitted in parallel)
bash scripts/runai_stage2.sh

# Stage 3 — Proxy training (after Stage 2 full checkpoint is ready)
bash scripts/runai_train_proxy.sh

# Evaluation — all systems on the held-out test set
bash scripts/runai_evaluate_all.sh

# Training plots (written to the repo folder)
bash scripts/runai_plot_curves.sh        # Stage 2 → hateful_meme_rewriting/training_plots/stage_2_training_plots/
bash scripts/runai_plot_proxy_curves.sh  # Stage 3 → hateful_meme_rewriting/training_plots/stage_3_training_plots/
```

---

## Results

All systems evaluated on the same 280 held-out test memes. BART-FT results are mean ± sample std across 5 random seeds. LLaVA teacher (fixed pseudo-rewrite labels) and DetoxLLM are single reported runs.

| System | Text STA | STA Δ | SIM | CLIP |
|---|---:|---:|---:|---:|
| LLaVA-NeXT teacher *(upper bound)* | 0.990 | 0.234 | 0.472 | 0.636 |
| DetoxLLM-7B *(external baseline)* | 0.940 | 0.184 | 0.443 | 0.634 |
| BART-large *(no fine-tuning)* | 0.973 | 0.217 | −0.061 | 0.622 |
| **BART-FT full** *(ours)* | **0.963 ± 0.006** | **0.207 ± 0.006** | 0.330 ± 0.013 | 0.628 ± 0.001 |
| BART-FT target\_only *(ours)* | 0.960 ± 0.005 | 0.204 ± 0.005 | 0.335 ± 0.013 | 0.629 ± 0.002 |
| BART-FT visual\_only *(ours)* | 0.960 ± 0.004 | 0.204 ± 0.004 | 0.284 ± 0.080 | 0.624 ± 0.007 |
| BART-FT none *(ours)* | 0.957 ± 0.013 | 0.201 ± 0.013 | 0.325 ± 0.030 | 0.628 ± 0.002 |
| **CLIP Proxy + BART** *(ours)* | 0.884 ± 0.015 | 0.128 ± 0.015 | **0.581 ± 0.020** | **0.638 ± 0.002** |

**Text STA**: mean non-toxic probability from `s-nlp/roberta_toxicity_classifier` (↑ better). **STA Δ**: rewrite STA minus original meme text STA (↑ better). **SIM**: BERTScore F1 (↑ better). **CLIP**: normalized image–text alignment (↑ better).

**Key findings.** All BART-FT conditions outperform DetoxLLM on Text STA and STA Δ, confirming the value of meme-specific LoRA fine-tuning over general-purpose text detoxification. Base BART achieves high non-toxicity (0.973) but negative SIM (−0.061), producing generic safe text unrelated to the source. DetoxLLM's relatively high SIM partly reflects copy behaviour — verbatim retention of toxic content inflates BERTScore without genuine rewriting.

The CLIP Proxy exposes a fidelity–detoxification trade-off: it achieves the best SIM (0.581) and CLIPScore (0.638), driven by stronger source retention and image alignment, but its Text STA drops to 0.884. Its main advantage is architectural — CLIP + small MLP + BART decoder, with no 7B VLM at inference.

---

## Compute and Emissions

All training and inference ran on `NVIDIA A100-SXM4-40GB` at EPFL RCP (Swiss electricity grid, ~35 g CO2/kWh). Emissions tracked per stage with CodeCarbon.

| Stage | GPU-time | Energy (kWh) | CO2 |
|---|---:|---:|---:|
| Stage 0 — OCR + CLIP (163,544 images) | 2.8 h | 0.76 | 26.7 g |
| Stage 1 — LLaVA train shards (8×) | 14.1 h | 6.61 | 230.3 g |
| Stage 1 — LLaVA val + test shards (4×) | 3.5 h | 1.65 | 57.5 g |
| Stage 2 — BART LoRA training (4 conditions) | 2.1 h | 0.82 | 28.4 g |
| Stage 2/3 — BART + Proxy inference + evaluation | 68.6 min | 0.23 | 8.3 g |
| DetoxLLM baseline | 7.9 min | 0.04 | 1.3 g |
| **Total** | **~38 h** | **~10.11** | **~352 g** |

Over 80% of total emissions come from Stage 1 (LLaVA teacher). At deployment, BART-FT requires **122 ms/meme** (76.6× faster than LLaVA, 16.3× faster than DetoxLLM). The CLIP Proxy + BART path runs at ~730 ms/meme — 12.9× faster than LLaVA and 2.7× faster than DetoxLLM — with no large VLM at inference.

To aggregate pipeline emissions a posteriori (no GPU needed):
```bash
bash scripts/runai_pipeline_co2.sh
```

