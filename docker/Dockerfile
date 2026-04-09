FROM registry.rcp.epfl.ch/rcp-courses/2026-ee-559:latest

# Install Python 3.12 from deadsnakes PPA (standard source for Pythons on Ubuntu)
RUN apt update && apt install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && apt install -y python3.12 python3.12-venv python3.12-dev && \
    rm -r /var/lib/apt/lists/*

# Create a virtualenv using Python 3.12
RUN python3.12 -m venv /opt/venv

# Activate the venv by default
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip inside the venv
RUN pip install --upgrade pip

# Copy and install requirements
COPY requirements_docker.txt .
RUN pip install -r requirements_docker.txt

ENV TOKENIZERS_PARALLELISM=false
ENV HF_HOME=/scratch/hf_cache
