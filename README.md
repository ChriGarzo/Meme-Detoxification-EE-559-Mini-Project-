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
Stage 1A ─── LLaVA-Next explanations (sharded)  [GPU]
               │  Run on unified_train split only (8 shards in parallel)
               │  Output: train_explanations_shardXXof08.jsonl
               ▼
  Merge explanations shards
               │  Output: train_explanations_merged.jsonl
               ▼
Stage 1B ─── LLaVA-Next pseudo-rewrites (sharded)  [GPU]
               │  Reads explanations from merged/sharded JSONL
               │  Skips IDs that already have a kept rewrite
               │  Output: train_pseudo_rewrites_shardXXof08.jsonl
               ▼
  Merge pseudo-rewrite shards
               │  Output: train_pseudo_rewrites_merged.jsonl
               ▼
   Build Stage 2 Dataset
               │  Merges all Stage 1 JSONL outputs into train.jsonl / val.jsonl
               │  Input format: [T: {target}] [V: {visual_evidence}] [M: {meaning}] | {text}
               ▼
Stage 2 ───── BART LoRA meme fine-tuning (×4 conditions in parallel)  [GPU]
               │  Conditions: full | target_only | visual_only | none
               │  LoRA: r=32, alpha=64, dropout=0.05 on q/k/v/out_proj + fc1/fc2
               │  ~17M trainable / 400M total parameters (~4.3%)
               │  5 epochs, lr=1e-4, starts directly from facebook/bart-large
               │  Evaluation metrics per checkpoint:
               │    ROUGE-1/2/L, collapse rate, text STA (RoBERTa),
               │    multimodal STA (VisualBERT image+text, eval-only)
               │  5 qualitative (original → generated → reference) examples logged
               │  Output: merged BART checkpoint + lora_adapter/ subdirectory
               ▼
Stage 4 ──── Proxy network training  [GPU]
               │  3-layer MLP: concat(CLIP image + CLIP text) [1536-dim]
               │    → BART encoder hidden state [1024-dim]
               │  Enables VLM-free inference at deployment
               ▼
Stage 3 ──── Evaluation  [GPU]
               BLEU, ROUGE-L, BERTScore, toxicity reduction
               + LLaVA structured-prompt baseline
```

**Models used:**
- `llava-hf/llava-v1.6-mistral-7b-hf` — Stage 1 teacher
- `facebook/bart-large` — Stage 2 student (LoRA fine-tuned, ~4.3% trainable params)
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
│       └── build_stage2_dataset.py    ← merges Stage 1 outputs → train/val JSONL
│
├── models/
│   ├── explainer.py                   ← LLaVA-Next wrapper
│   ├── rewriter.py                    ← BART wrapper (+ generate_from_formatted)
│   └── proxy.py                       ← ExplanationProxy MLP + CLIP
│
├── inference/
│   ├── run_stage1.py                  ← legacy Stage 1 entrypoint
│   ├── run_stage1_explanations_only_sharded.py ← Stage 1A (explanations-only)
│   ├── run_stage1_rewrites_only_sharded.py     ← Stage 1B (rewrites-only, reuses explanations)
│   ├── merge_stage1_explanations_shards.py     ← merges explanation shards
│   ├── merge_stage1_rewrites_shards.py         ← merges rewrite shards
│   ├── run_stage2.py                  ← BART inference over filtered memes
│   └── run_proxy_pipeline.py          ← VLM-free inference via proxy
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
    ├── runai_stage1_explain.sh        ← legacy Stage 1 launcher
    ├── runai_stage1_explanations_only_sharded.sh
    ├── runai_stage1_rewrites_only_sharded.sh
    ├── runai_build_stage2_dataset.sh
    ├── runai_stage2_phase2.sh
    ├── runai_train_proxy.sh
    └── runai_evaluate.sh
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

**Stage 1A — explanations only (sharded)**  
Run all shards. Example for 8 shards:
```bash
for i in $(seq 0 7); do
  SHARD_ID=$i NUM_SHARDS=8 bash scripts/runai_stage1_explanations_only_sharded.sh <UID>
done
```

**Merge explanation shards**  
Run once after all Stage 1A shards finish:
```bash
python inference/merge_stage1_explanations_shards.py \
  --dataset train \
  --input_dir /scratch/hmr_stage1_output \
  --num_shards 8
```
This creates:
`/scratch/hmr_stage1_output/train_explanations_merged.jsonl`

**Stage 1B — rewrites only (sharded, reuses existing explanations)**  
Run all shards. Example for 8 shards:
```bash
for i in $(seq 0 7); do
  SHARD_ID=$i NUM_SHARDS=8 bash scripts/runai_stage1_rewrites_only_sharded.sh <UID>
done
```
By default, rewrites-only will first look for:
`/scratch/hmr_stage1_output/train_explanations_merged.jsonl`
and will skip examples that already exist in shard rewrite outputs.

**Merge pseudo-rewrite shards**  
Run once after all Stage 1B shards finish:
```bash
python inference/merge_stage1_rewrites_shards.py \
  --dataset train \
  --input_dir /mnt/course-ee-559/rcp-caas-ee-559-g31/scratch-g31/hmr_stage1_output \
  --num_shards 8
```
This creates:
`/scratch/hmr_stage1_output/train_pseudo_rewrites_merged.jsonl`

**Build Stage 2 dataset** (wait for all Stage 1 jobs to complete)
```bash
bash scripts/runai_build_stage2_dataset.sh <UID>
```

**Stage 2 — BART LoRA meme fine-tuning** (4 jobs submitted in parallel, one per condition)
```bash
bash scripts/runai_stage2_phase2.sh <UID>
```
Trains four separate LoRA-adapted checkpoints starting directly from `facebook/bart-large`: `full`, `target_only`, `visual_only`, `none`.

Each job applies LoRA (r=32, alpha=64, dropout=0.05) to all attention projections and FFN layers, giving ~17M trainable parameters out of 400M total. At the end of training, the adapter is merged back into the base model and saved to the checkpoint directory. The raw LoRA adapter weights are preserved in `lora_adapter/` for potential future reuse.

Metrics tracked at every eval checkpoint: ROUGE-1/2/L, collapse rate, text STA (RoBERTa toxicity), and multimodal STA (VisualBERT with CLIP image features — images are used for this metric only and never influence gradients). Five qualitative examples (original → generated → reference) are logged at each eval step.

Note: Stage 2 Phase 1 (ParaDetox warm-up) is kept in `training/train_stage2_phase1.py` for reference but is not run in the current pipeline. Phase 2 starts directly from `facebook/bart-large`.

**Stage 4 — Train proxy network** (wait for Stage 2 Phase 2 `full` to complete)
```bash
bash scripts/runai_train_proxy.sh <UID>
```

**Stage 3 — Evaluation** (wait for all Stage 2 Phase 2 checkpoints)
```bash
bash scripts/runai_evaluate.sh <UID>
```

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
│   │   │   ├── harmeme/
│   │   │   ├── mami/
│   │   │   └── mmhs150k/
│   │   └── discarded/
│   │       ├── harmeme/
│   │       ├── mami/
│   │       └── mmhs150k/
│   └── unified_splits/                     ← built after all Stage 0 jobs complete
│       ├── unified_train.csv
│       ├── unified_val.csv
│       ├── unified_test.csv
│       └── split_stats.json
├── hmr_stage1_output/
│   ├── stage1_explain_only_shardXXof08.log
│   ├── stage1_rewrite_only_shardXXof08.log
│   ├── train_explanations_shard00of08.jsonl
│   ├── ... train_explanations_shard07of08.jsonl
│   ├── train_explanations_merged.jsonl
│   ├── train_pseudo_rewrites_shard00of08.jsonl
│   ├── ... train_pseudo_rewrites_shard07of08.jsonl
│   └── train_pseudo_rewrites_merged.jsonl
├── hmr_stage2_dataset/
│   ├── train.jsonl
│   └── val.jsonl
├── hmr_stage2_phase2_full_checkpoint/
├── hmr_stage2_phase2_target_only_checkpoint/
├── hmr_stage2_phase2_visual_only_checkpoint/
├── hmr_stage2_phase2_none_checkpoint/
├── hmr_proxy_checkpoint/
└── hmr_eval_results/
```

---

## Current Experiment Report

This section summarizes the current state of the repository and the results obtained from the latest full pipeline run. The goal of the project is not to outperform LLaVA-Next directly. LLaVA-Next is used as a large teacher model to produce explanations and pseudo-rewrites; the real objective is to fine-tune a much smaller BART-large student so that it can detoxify meme text while preserving the intended meaning as much as possible.

### What Was Done

The implemented pipeline follows an explain-then-rewrite strategy:

1. **Filter meme-like images.** Stage 0 uses EasyOCR to extract overlaid meme text and CLIP to discard images that are not visually meme-like. This is necessary because some raw datasets, especially MMHS150K, contain many plain photos, screenshots, and social-media UI captures.

2. **Build unified splits.** The filtered HarMeme, MAMI, and MMHS150K examples are merged into stratified train/validation/test splits. The split is stratified by dataset and hateful label so each split contains examples from all sources.

3. **Generate LLaVA explanations.** Stage 1A runs LLaVA-Next over the unified training split to produce structured hate explanations. Each explanation contains:
   - `target_group`: the protected or attacked group.
   - `visual_evidence`: image evidence relevant to the hateful meaning.
   - `implicit_meaning`: the implied hateful or harmful interpretation.

4. **Generate LLaVA pseudo-rewrites.** Stage 1B asks LLaVA-Next to rewrite the meme text into a less toxic version. These pseudo-rewrites become the supervision target for BART.

5. **Filter pseudo-rewrites before BART training.** The Stage 2 dataset builder keeps only pseudo-rewrites that pass the Stage 1 quality filters, have no parse error, have positive toxicity reduction, and are not almost identical to the source text. The current dataset statistics are:

   | Quantity | Count |
   |---|---:|
   | Loaded LLaVA rewrite rows | 7,918 |
   | Dropped because Stage 1 filters failed | 3,701 |
   | Dropped because of parse errors | 560 |
   | Kept after parse/toxicity filtering | 3,657 |
   | Kept after rewrite text quality filtering | 3,578 |
   | Train examples | 3,220 |
   | Validation examples | 358 |

   The final training set is therefore intentionally smaller but cleaner. This is a good fit for LoRA fine-tuning because only a small fraction of BART's parameters are updated.

6. **Fine-tune BART-large with LoRA.** Four BART-large students are trained, one for each conditioning setting:

   | Condition | Encoder input format |
   |---|---|
   | `full` | `[T: target_group] [V: visual_evidence] [M: implicit_meaning] \| original_text` |
   | `target_only` | `[T: target_group] [V: null] [M: null] \| original_text` |
   | `visual_only` | `[T: null] [V: visual_evidence] [M: null] \| original_text` |
   | `none` | `[T: null] [V: null] [M: null] \| original_text` |

   The current fine-tuning setup is:
   - Base model: `facebook/bart-large`.
   - Training data: meme pseudo-rewrites only; no ParaDetox warm-up in the current run.
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

7. **Train a proxy model.** The repo also contains a proxy stage. The proxy is a small MLP trained from CLIP image/text features to BART encoder hidden states. Its purpose is to support a future VLM-free deployment path. In the current project, the main reported comparison is still the BART fine-tuning/evaluation pipeline.

8. **Evaluate all systems on the held-out validation set.** The evaluation script compares:
   - `llava_teacher`: the LLaVA pseudo-rewrite target from `hmr_stage2_dataset/val.jsonl`.
   - `bart_base_*`: non-finetuned `facebook/bart-large` under the four input conditions.
   - `bart_finetuned_*`: the four LoRA-finetuned BART models.

   All systems are evaluated on the same 358 validation examples. The evaluation uses the exact BART outputs as generated. The code does not sanitize or post-process BART rewrites before scoring. The only truncation is internal to CLIPScore because CLIP has a hard 77-token text limit.

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
| `n` | Number of evaluated validation examples. |
| `valid_images` | Number of examples with valid image paths for image-based metrics. |
| `text_sta` | Mean non-toxic probability from `s-nlp/roberta_toxicity_classifier`; higher is better. |
| `text_sta_delta` | Rewrite text STA minus original text STA; positive means the rewrite is less toxic than the original. |
| `sim` | BERTScore F1 between original text and rewrite; higher means better meaning preservation. |
| `clip` | CLIPScore between image and rewrite; higher means better image-text alignment. |
| `visualbert_hate_prob` | Mean multimodal hate probability from VisualBERT. |
| `visualbert_sta` | Fraction of examples VisualBERT predicts as non-hateful. |
| `visualbert_hate_drop` | Original VisualBERT hate probability minus rewrite hate probability; positive means lower predicted hate after rewriting. |

### Current Validation Results

Latest `evaluation_summary.tsv`:

| System | n | Text STA | Text STA Delta | SIM | CLIP | VisualBERT Hate Prob | VisualBERT STA | VisualBERT Hate Drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `llava_teacher` | 358 | 0.9517 | 0.2804 | 0.3938 | 0.6253 | 0.6531 | 0.0000 | -0.0072 |
| `bart_base_full` | 358 | 0.9088 | 0.2376 | -0.2235 | 0.6073 | 0.6582 | 0.0000 | -0.0122 |
| `bart_base_target_only` | 358 | 0.9410 | 0.2698 | -0.3046 | 0.5948 | 0.6601 | 0.0000 | -0.0142 |
| `bart_base_visual_only` | 358 | 0.9242 | 0.2530 | -0.2320 | 0.5995 | 0.6624 | 0.0000 | -0.0165 |
| `bart_base_none` | 358 | 0.9355 | 0.2643 | -0.2954 | 0.5945 | 0.6640 | 0.0000 | -0.0181 |
| `bart_finetuned_full` | 358 | 0.9261 | 0.2548 | 0.2991 | 0.6236 | 0.6539 | 0.0000 | -0.0080 |
| `bart_finetuned_target_only` | 358 | 0.9131 | 0.2419 | 0.2815 | 0.6219 | 0.6523 | 0.0000 | -0.0063 |
| `bart_finetuned_visual_only` | 358 | 0.9169 | 0.2456 | 0.3252 | 0.6261 | 0.6522 | 0.0000 | -0.0063 |
| `bart_finetuned_none` | 358 | 0.9178 | 0.2465 | 0.3261 | 0.6256 | 0.6517 | 0.0000 | -0.0058 |

### Main Findings

The most important result is that LoRA fine-tuning makes BART much more faithful to the rewriting task.

Compared with non-finetuned BART, the finetuned BART models move from negative SIM scores to clearly positive SIM scores:

| Condition | Base BART SIM | Finetuned BART SIM | Change |
|---|---:|---:|---:|
| `full` | -0.2235 | 0.2991 | +0.5226 |
| `target_only` | -0.3046 | 0.2815 | +0.5861 |
| `visual_only` | -0.2320 | 0.3252 | +0.5572 |
| `none` | -0.2954 | 0.3261 | +0.6215 |

This means that non-finetuned BART can often produce superficially safe text, but those outputs are not semantically close to the desired rewrite. Fine-tuning teaches BART the actual meme rewriting distribution: the output remains much closer to the original/teacher meaning while still reducing textual toxicity.

CLIPScore also improves in every condition:

| Condition | Base BART CLIP | Finetuned BART CLIP | Change |
|---|---:|---:|---:|
| `full` | 0.6073 | 0.6236 | +0.0163 |
| `target_only` | 0.5948 | 0.6219 | +0.0271 |
| `visual_only` | 0.5995 | 0.6261 | +0.0266 |
| `none` | 0.5945 | 0.6256 | +0.0311 |

This suggests that the finetuned outputs are not only textually closer to the target rewrite, but also more compatible with the meme image.

Textual detoxification remains high after fine-tuning:

| Condition | Base BART Text STA | Finetuned BART Text STA |
|---|---:|---:|
| `full` | 0.9088 | 0.9261 |
| `target_only` | 0.9410 | 0.9131 |
| `visual_only` | 0.9242 | 0.9169 |
| `none` | 0.9355 | 0.9178 |

The `full` condition improves both text STA and faithfulness over base BART. The other finetuned conditions have slightly lower text STA than their base-BART counterparts, but they are far more faithful. For the project goal, this trade-off is acceptable: a model that keeps `text_sta` around 0.91-0.93 while greatly improving semantic similarity is more useful than a model that produces generic safe text unrelated to the intended rewrite.

The LLaVA teacher remains the best system overall in text STA and SIM. This is expected because LLaVA is a much larger vision-language model and also produced the pseudo-labels. The relevant comparison for the student model is therefore not "does BART beat LLaVA?", but "does fine-tuning make BART a useful small detoxification model?". The answer from the current results is yes.

### Important Caveat About VisualBERT

The VisualBERT multimodal metric should be interpreted cautiously. `visualbert_sta` is `0.0000` for every system, including the LLaVA teacher. This likely means the VisualBERT classifier is too strict, miscalibrated for rewritten text, or not well matched to this evaluation setup. The hate probability changes are directionally interesting, but the absolute non-hateful prediction rate is not reliable enough to be a main conclusion.

The safer conclusion is:

- Text toxicity metrics show that rewrites are less toxic.
- SIM and CLIPScore show that fine-tuning improves meaning preservation and image-text alignment.
- VisualBERT does not currently provide a useful absolute pass/fail multimodal detox score for this validation set.

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

The training logs show the expected behavior:

- Training loss decreases strongly for every condition.
- Validation text STA increases during training.
- Text toxicity drop improves during training.
- ROUGE stays relatively stable instead of collapsing.
- Copy rate is monitored to detect outputs that are too close to the original.
- VisualBERT STA stays flat at 0, reinforcing the caveat above.

For example, in the `full` condition:

| Metric | Early eval | Final eval |
|---|---:|---:|
| Eval loss | 1.5503 | 1.4423 |
| Text STA | 0.8631 | 0.9274 |
| Text toxicity drop | 0.1914 | 0.2547 |
| Detox quality | 0.2091 | 0.2788 |
| High-copy rate | 0.1061 | 0.0754 |

This confirms that the model is not merely memorizing copies of the input. It learns to produce outputs that are more non-toxic while keeping a controlled level of similarity to the source.

### Final Interpretation

The current finetuning is useful for the intended scope. It distills part of LLaVA's rewriting behavior into a much smaller BART-large model. The student does not reach the teacher's quality, but it substantially improves over non-finetuned BART in the metrics that matter most for a small detoxification model:

- It keeps textual toxicity low.
- It greatly improves semantic faithfulness.
- It improves image-text alignment.
- It avoids obvious collapse.
- It can be evaluated under multiple conditioning ablations.

The most interesting next steps are:

1. Improve or replace the multimodal toxicity metric, because the current VisualBERT score is not sufficiently informative.
2. Inspect qualitative outputs per condition to understand why `none`, `visual_only`, and `full` are close in final performance.
3. Consider stronger conditioning usage, for example by training a single multi-condition model with explicit condition dropout rather than four fully separate models.
4. Report the fine-tuned BART result primarily as a compact student model that trades a small loss in teacher-level quality for much lower inference cost.

---

## Citations

- Pramanick et al. (2021). HarMeme: A Dataset for Hate Speech Detection in Memes. *EMNLP*.
- Fersini et al. (2022). SemEval-2022 Task 5: Multimedia Automatic Misogyny Identification. *SemEval*.
- Gomez et al. (2020). Exploring Hate Speech Detection in Multimodal Publications. *CVPRW*.
- Logacheva et al. (2022). ParaDetox: Detoxification with Parallel Data. *ACL*.
- Liu et al. (2024). LLaVA-1.6: Improved Baselines with Visual Instruction Tuning. *arXiv:2401.00774*.
