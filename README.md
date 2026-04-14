# Explain-then-Rewrite: Leveraging Hate Explanation Generation for Targeted Meme Text Detoxification

**EE-559 Deep Learning — EPFL, Group 31**

A pipeline that uses LLaVA-Next (7B) as a teacher model to generate structured hate explanations and pseudo-rewrites, then trains BART-large (400M) as a lightweight student for meme text detoxification. Four conditioning ablations are evaluated: `full`, `target_only`, `attack_only`, `none`.

---

## Pipeline Overview

```
Stage 0 ──── OCR + CLIP filtering  [per dataset, GPU]
               │  EasyOCR extracts meme text (10–300 chars)
               │  CLIP score: meme_score > screenshot_score
               │  Outputs: /scratch/hmr_data/<dataset>/manifest.csv
               ▼
  Build Unified Splits
               │  Reads Stage 0 manifests (kept=True only)
               │  Stratified 80/10/10 split by (dataset, hateful)
               │  All 3 datasets represented in every split
               │  Outputs: unified_train.csv / unified_val.csv / unified_test.csv
               ▼
Stage 1 ──── LLaVA-Next explanations + pseudo-rewrites  [GPU]
               │  Run on unified_train split only
               │    - target_group, attack_type, implicit_meaning
               │    - pseudo-rewrite (teacher output, BERTScore + STA filtered)
               ▼
   Build Stage 2 Dataset
               │  Merges all Stage 1 JSONL outputs into train.jsonl / val.jsonl
               │  Input format: [T: {target}] [A: {attack}] [M: {meaning}] </s> {text}
               ▼
Stage 2 Ph1 ── BART ParaDetox warm-up  [GPU]
               │  Seq2SeqTrainer on s-nlp/paradetox
               │  2 epochs, lr=5e-5, warmup=500
               ▼
Stage 2 Ph2 ── BART meme fine-tuning (×4 conditions in parallel)  [GPU]
               │  Conditions: full | target_only | attack_only | none
               │  3 epochs, lr=2e-5, starts from Phase 1 checkpoint
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
- `facebook/bart-large` — Stage 2 student (fine-tuned)
- `openai/clip-vit-large-patch14` — Stage 0 filter + Stage 4 proxy (768-dim embeddings)

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
│       ├── build_unified_splits.py    ← builds 80/10/10 splits from Stage 0 manifests
│       ├── build_stage2_dataset.py    ← merges Stage 1 outputs → train/val JSONL
│       └── rule_based_labels.py
│
├── models/
│   ├── explainer.py                   ← LLaVA-Next wrapper
│   ├── rewriter.py                    ← BART wrapper (+ generate_from_formatted)
│   └── proxy.py                       ← ExplanationProxy MLP + CLIP
│
├── inference/
│   ├── run_stage1.py                  ← LLaVA inference: explanations + pseudo-rewrites
│   ├── run_stage2.py                  ← BART inference over filtered memes
│   └── run_proxy_pipeline.py          ← VLM-free inference via proxy
│
├── training/
│   ├── train_stage2_phase1.py         ← ParaDetox warm-up (Seq2SeqTrainer)
│   ├── train_stage2_phase2.py         ← Meme fine-tuning (×4 conditions)
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
    ├── runai_build_unified_splits.sh  ← builds unified splits after Stage 0
    ├── runai_stage1_explain.sh
    ├── runai_build_stage2_dataset.sh
    ├── runai_stage2_phase1.sh
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
Downloaded automatically from HuggingFace by `train_stage2_phase1.py` at training time (`s-nlp/paradetox`). No manual step needed.

---

## Cluster Workflow (EPFL RCP — Group 31)

### Infrastructure

| Resource | Value |
|---|---|
| Group scratch PVC | `course-ee-559-scratch-g31` mounted at `/scratch/` |
| Personal home PVC | `home` mounted at `/home/${USER}/` |
| Docker registry | `registry.rcp.epfl.ch/ee-559-<username>/hmr:v0.1` |
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
git clone <repo_url>
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
Each job outputs a `manifest.csv` with OCR/CLIP scores and a `kept` flag, plus visual grids of kept/discarded images at `/scratch/hmr_data/<dataset>/filter_examples/` for QC.

**Build unified splits** (run after ALL three Stage 0 jobs complete)
```bash
bash scripts/runai_build_unified_splits.sh <UID>
runai logs hmr-build-unified-splits -p course-ee-559-<username> --follow
```
Reads the three Stage 0 manifests, filters to `kept=True` images only, then creates stratified 80/10/10 splits ensuring all three datasets are represented in every split. Outputs `unified_train.csv`, `unified_val.csv`, `unified_test.csv` to `/scratch/hmr_data/unified_splits/`.

**Stage 1 — LLaVA explanations + pseudo-rewrites** (runs on unified training split)
```bash
bash scripts/runai_stage1_explain.sh <UID>
```

**Build Stage 2 dataset** (wait for all Stage 1 jobs to complete)
```bash
bash scripts/runai_build_stage2_dataset.sh <UID>
```

**Stage 2 Phase 1 — BART ParaDetox warm-up**
```bash
bash scripts/runai_stage2_phase1.sh <UID>
```

**Stage 2 Phase 2 — BART meme fine-tuning** (4 jobs submitted in parallel, one per condition)
```bash
bash scripts/runai_stage2_phase2.sh <UID>
```
Trains four separate checkpoints: `full`, `target_only`, `attack_only`, `none`.

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
│   │   ├── manifest.csv                    ← Stage 0 output (OCR/CLIP scores + kept flag)
│   │   └── filter_examples/
│   │       ├── kept_examples.png           ← visual QC grid
│   │       └── discarded_examples.png
│   ├── mami/   (same structure)
│   ├── mmhs150k/  (same structure)
│   └── unified_splits/                     ← built after all Stage 0 jobs complete
│       ├── unified_train.csv
│       ├── unified_val.csv
│       ├── unified_test.csv
│       └── split_stats.json
├── hmr_stage1_output/
│   ├── harmeme/
│   │   ├── harmeme_explanations.jsonl
│   │   └── harmeme_pseudo_rewrites.jsonl
│   ├── mami/  (same structure)
│   └── mmhs150k/  (same structure)
├── hmr_stage2_dataset/
│   ├── train.jsonl
│   └── val.jsonl
├── hmr_stage2_phase1_checkpoint/
├── hmr_stage2_phase2_full_checkpoint/
├── hmr_stage2_phase2_target_only_checkpoint/
├── hmr_stage2_phase2_attack_only_checkpoint/
├── hmr_stage2_phase2_none_checkpoint/
├── hmr_proxy_checkpoint/
└── hmr_eval_results/
```

---

## Evaluation Results

| Condition | BLEU | ROUGE-L | BERTScore | Toxicity ↓ |
|-----------|------|---------|-----------|------------|
| full      |      |         |           |            |
| target_only |   |         |           |            |
| attack_only |   |         |           |            |
| none      |      |         |           |            |
| LLaVA baseline |  |      |           |            |

---

## Citations

- Pramanick et al. (2021). HarMeme: A Dataset for Hate Speech Detection in Memes. *EMNLP*.
- Fersini et al. (2022). SemEval-2022 Task 5: Multimedia Automatic Misogyny Identification. *SemEval*.
- Gomez et al. (2020). Exploring Hate Speech Detection in Multimodal Publications. *CVPRW*.
- Logacheva et al. (2022). ParaDetox: Detoxification with Parallel Data. *ACL*.
- Liu et al. (2024). LLaVA-1.6: Improved Baselines with Visual Instruction Tuning. *arXiv:2401.00774*.
