# Explain-then-Rewrite: Leveraging Hate Explanation Generation for Targeted Meme Text Detoxification

**Abstract:** A pipeline that uses LLaVA (7B) as teacher to generate structured hate explanations, then trains BART (400M) as a lightweight student for meme text detoxification, achieving competitive quality at ~20x fewer parameters.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     EXPLAIN-THEN-REWRITE PIPELINE                   │
└─────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────┐
    │   Stage 0: Filter    │
    │   OCR + CLIP         │
    │   Extract text/check │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────────────────┐
    │  Stage 1: LLaVA Explanation      │
    │  ├─ Generate hate explanations   │
    │  └─ Pseudo-rewrites (teacher)    │
    └──────────┬───────────────────────┘
               │
               ▼
    ┌──────────────────────────────────┐
    │  Stage 2: BART Student Training  │
    │  ├─ Phase 1: ParaDetox pretraining
    │  └─ Phase 2: Meme fine-tuning    │
    └──────────┬───────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Stage 3: Evaluation │
    │  Metrics & Analysis  │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────────────────┐
    │ Stage 4: Proxy Network           │
    │ VLM-free deployment (ONNX)       │
    └──────────────────────────────────┘
```

## Local Development (No GPU)

To run a debug version locally without GPU access:

```bash
bash scripts/run_debug_local.sh
```

**What DEBUG does:**
- Loads minimal dataset samples (debug subset)
- Runs Stage 0 filtering with OCR/CLIP on small batch
- Executes simplified Stage 1 explanations with LLaVA-7B in CPU mode
- Trains BART for 1 epoch with reduced batch size
- Computes evaluation metrics on filtered data
- Outputs results to `outputs/debug_run/`

**Expected final output line:**
```
[DEBUG] Pipeline complete. Results saved to outputs/debug_run/final_results.json
```

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Setup

### HarMeme
Automatically downloaded via:
```bash
bash scripts/download_datasets.sh
```
**Note:** You will be prompted for dataset usage permission confirmation.

### MAMI
Manual request required at: https://forms.gle/AGWMiGicBHiQx4q98

### MMHS150K
Download from: https://gombru.github.io/2019/10/09/MMHS/

### ParaDetox
Automatically downloaded from HuggingFace:
```bash
python scripts/download_paradetox.py
```

## Cluster Workflow

### Setup Information
- **Group:** 31 (shared scratch PVC: `course-ee-559-scratch-g31`)
- All group members use the **same `/scratch/` storage** — datasets only need to be
  downloaded and preprocessed **once** by any one group member.
- Code lives in each member's personal `/home/<username>/` directory.

### Step 1: Docker Build & Registry Push (each member once)

Each member must push their own Docker image to the EPFL registry under their username:
```bash
# Run locally on your machine
docker build -t hmr:v0.1 docker/
docker tag hmr:v0.1 registry.rcp.epfl.ch/ee-559-${USER}/hmr:v0.1
docker push registry.rcp.epfl.ch/ee-559-${USER}/hmr:v0.1
```

### Step 2: SSH & RunAI Setup

```bash
ssh <your_username>@jumphost.rcp.epfl.ch
runai login
runai config project course-ee-559-${USER}
```

### Step 3: Get your UID

All scripts need your numeric Unix UID:
```bash
id -u    # prints e.g. 123456
```

### Step 4: Clone Repository (each member)

```bash
cd /home/${USER}/
git clone <repo_url>
cd hateful_meme_rewriting
```

### Step 5: Submit Jobs (Sequential)

All scripts accept your UID as the **first argument**. Your Unix username (`$USER`)
is picked up automatically — you never need to edit the scripts.

**Stage 0: Filter Memes** (run once per dataset; shared output on /scratch/)
```bash
bash scripts/runai_stage0_filter.sh <UID> harmeme
bash scripts/runai_stage0_filter.sh <UID> mami
bash scripts/runai_stage0_filter.sh <UID> mmhs150k
```
After each run, visual grids of **kept** and **discarded** images are saved to
`/scratch/hmr_data/<dataset>/filter_examples/` so you can verify the filter.

**Stage 1: Generate LLaVA Explanations + Pseudo-rewrites** (all datasets in parallel)
```bash
bash scripts/runai_stage1_explain.sh <UID>
```

**Build Stage 2 Dataset** (wait for all Stage 1 jobs to finish)
```bash
bash scripts/runai_build_stage2_dataset.sh <UID>
```

**Stage 2 Phase 1: ParaDetox Pretraining**
```bash
bash scripts/runai_stage2_phase1.sh <UID>
```

**Stage 2 Phase 2: Meme Fine-tuning** (4 conditions in parallel)
```bash
bash scripts/runai_stage2_phase2.sh <UID>
```
Conditions: `full` | `target_only` | `attack_only` | `none`

**Stage 4: Train Proxy Network**
```bash
bash scripts/runai_train_proxy.sh <UID>
```

**Stage 3: Evaluation**
```bash
bash scripts/runai_evaluate.sh <UID>
```

### Storage Layout on `/scratch/` (shared by all group members)

```
/scratch/
├── hf_cache/                          ← HuggingFace model cache (downloaded once)
├── hmr_data/
│   ├── harmeme/images/                ← raw images
│   ├── harmeme/manifest.csv           ← Stage 0 output
│   ├── harmeme/filter_examples/       ← kept_examples.png + discarded_examples.png
│   ├── mami/ ...
│   └── mmhs150k/ ...
├── hmr_stage1_output/
│   ├── harmeme/{harmeme}_explanations.jsonl
│   ├── harmeme/{harmeme}_pseudo_rewrites.jsonl
│   ├── mami/ ...
│   └── mmhs150k/ ...
├── hmr_stage2_dataset/
│   ├── train.jsonl
│   └── val.jsonl
├── hmr_stage2_phase1_checkpoint/
├── hmr_stage2_phase2_{full,target_only,attack_only,none}_checkpoint/
├── hmr_proxy_checkpoint/
└── hmr_eval_results/
```

## Evaluation Results

| Dataset | BLEU | ROUGE-L | Meaning Preservation | Toxicity Removal |
|---------|------|---------|----------------------|------------------|
| HarMeme |      |         |                      |                  |
| MAMI    |      |         |                      |                  |
| MMHS150K |     |         |                      |                  |

## Project Structure

```
hateful_meme_rewriting/
├── README.md
├── requirements.txt
├── setup.py
├── Dockerfile
│
├── data/
│   ├── harmeme/
│   ├── mami/
│   ├── mmhs150k/
│   └── paradetox/
│
├── scripts/
│   ├── download_datasets.sh
│   ├── download_paradetox.py
│   ├── run_debug_local.sh
│   ├── runai_stage0_filter.sh
│   ├── runai_stage1_explain.sh
│   ├── runai_stage2_phase1.sh
│   ├── runai_stage2_phase2.sh
│   ├── runai_train_proxy.sh
│   └── runai_evaluate.sh
│
├── src/
│   ├── __init__.py
│   ├── stage0_filter.py
│   ├── stage1_explain.py
│   ├── stage2_train.py
│   ├── stage3_evaluate.py
│   ├── stage4_proxy.py
│   └── models/
│       ├── llava_teacher.py
│       ├── bart_student.py
│       └── proxy_network.py
│
├── configs/
│   ├── stage0.yaml
│   ├── stage1.yaml
│   ├── stage2_phase1.yaml
│   ├── stage2_phase2.yaml
│   ├── stage3.yaml
│   └── stage4.yaml
│
└── outputs/
    ├── debug_run/
    ├── stage0_filtered/
    ├── stage1_explanations/
    ├── stage2_checkpoints/
    ├── stage3_results/
    └── stage4_proxy/
```

## Citations

- Pramanick et al. (2021). "HarMeme: A Dataset for Hate Speech Detection in Memes." *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*.
- Pramanick et al. (2021). "MOMENTA: A Multimodal Framework for Detecting Harmful Memes and Their Targets." *Proceedings of the 2021 IEEE/CVF International Conference on Computer Vision*.
- Fersini et al. (2022). "SemEval-2022 Task 5: Multimedia Automatic Misogyny Identification." *Proceedings of the 16th International Workshop on Semantic Evaluation*.
- Gomez et al. (2020). "Exploring Hate Speech Detection in Multimodal Publications." *Proceedings of the 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops*.
- Logacheva et al. (2022). "ParaDetox: Detoxification with Parallel Data." *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics*.
- Liu et al. (2024). "LLaVA-1.6: Improved Baselines with Visual Instruction Tuning." *arXiv preprint arXiv:2401.00774*.
- Khondaker et al. (2024). "DetoxLLM: Towards Detoxified Language Models." *arXiv preprint arXiv:2404.xxxx*.
