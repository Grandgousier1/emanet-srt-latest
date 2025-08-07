# Image CUDA runtime (Ubuntu 22.04) — OK RunPod B200
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/hf_cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TOKENIZERS_PARALLELISM=false \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git ffmpeg ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# PyTorch CUDA 12.1
RUN pip3 install --upgrade pip && \
    pip3 install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# Libs Python
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Script
WORKDIR /app
COPY emanet_srt.py /app/emanet_srt.py

# Dossiers cache/IO
RUN mkdir -p /app/output /app/workdir /app/hf_cache
VOLUME ["/app/output", "/app/workdir", "/app/hf_cache"]

ENTRYPOINT ["python3", "/app/emanet_srt.py"]
