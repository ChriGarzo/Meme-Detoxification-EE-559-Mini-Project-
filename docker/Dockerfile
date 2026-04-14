FROM registry.rcp.epfl.ch/rcp-courses/2026-ee-559:latest

# Install Python 3.12 from deadsnakes PPA
RUN apt update && apt install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && apt install -y python3.12 python3.12-venv python3.12-dev && \
    rm -r /var/lib/apt/lists/*

# Create a virtualenv using Python 3.12 and activate it
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip

# --- Layer 1: PyTorch (largest package, isolated so a retry only re-uploads this) ---
RUN pip install --no-cache-dir \
    torch>=2.2.0 \
    torchvision>=0.17.0

# --- Layer 2: HuggingFace core ---
RUN pip install --no-cache-dir \
    transformers>=4.40.0 \
    datasets>=2.18.0 \
    accelerate>=0.28.0 \
    peft>=0.10.0 \
    trl>=0.8.0

# --- Layer 3: Training & quantisation ---
RUN pip install --no-cache-dir \
    bitsandbytes>=0.43.0 \
    sentencepiece>=0.1.99 \
    protobuf>=4.25.0 \
    einops>=0.7.0

# --- Layer 4: NLP evaluation ---
RUN pip install --no-cache-dir \
    bert-score>=0.3.13 \
    evaluate>=0.4.0 \
    rouge_score>=0.1.2

# --- Layer 5: Vision & OCR ---
RUN pip install --no-cache-dir \
    Pillow>=10.0.0 \
    opencv-python>=4.9.0 \
    easyocr>=1.7.0

# --- Layer 6: Utilities ---
RUN pip install --no-cache-dir \
    numpy>=1.26.0 \
    pandas>=2.0.0 \
    scikit-learn>=1.4.0 \
    tqdm>=4.66.0 \
    wandb>=0.16.0 \
    pyyaml>=6.0 \
    codecarbon>=2.3.0

ENV TOKENIZERS_PARALLELISM=false
ENV HF_HOME=/scratch/hf_cache
