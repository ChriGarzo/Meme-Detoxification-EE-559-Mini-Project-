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
- **User:** garzone
- **Group:** 31

### Step 1: Docker Build & Registry Push

Build Docker image locally:
```bash
docker build -t hmr:v0.1 .
docker tag hmr:v0.1 registry.rcp.epfl.ch/ee-559-garzone/hmr:v0.1
docker push registry.rcp.epfl.ch/ee-559-garzone/hmr:v0.1
```

### Step 2: SSH & RunAI Setup

Connect to jumphost:
```bash
ssh jumphost.rcp.epfl.ch
```

Login and configure RunAI:
```bash
runai login
runai config project course-ee-559-${USER}
```

### Step 3: Clone Repository

```bash
cd /home/garzone/
git clone <repo_url>
cd hateful_meme_rewriting
```

### Step 4: Submit Jobs (Sequential)

Submit the following jobs in order, waiting for completion of each stage:

**Stage 0: Filter Memes** (run per dataset)
```bash
bash scripts/runai_stage0_filter.sh
```

**Stage 1: Generate Explanations**
```bash
bash scripts/runai_stage1_explain.sh
```

**Stage 2 Phase 1: ParaDetox Pretraining**
```bash
bash scripts/runai_stage2_phase1.sh
```

**Stage 2 Phase 2: Meme Fine-tuning**
```bash
bash scripts/runai_stage2_phase2.sh
```

**Stage 4: Train Proxy Network**
```bash
bash scripts/runai_train_proxy.sh
```

**Stage 3: Evaluation**
```bash
bash scripts/runai_evaluate.sh
```

**Note:** Large files are stored on `/scratch/`, while code resides on `/home/`.

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
