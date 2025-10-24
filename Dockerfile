# Dockerfile
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y build-essential libsndfile1 git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Set env vars for caching (optional)
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface

# Build‐time model download attempt
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
ENV MODEL_NAME=sesame/csm-1b

RUN python3 - <<'PYCODE'
import os, sys, traceback
from transformers import AutoProcessor
try:
    print("[build] Attempting to download model:", os.getenv("MODEL_NAME"))
    # Download processor
    AutoProcessor.from_pretrained(os.getenv("MODEL_NAME"), use_auth_token=os.getenv("HF_TOKEN"))
    # Attempt model load
    from transformers import CsmForConditionalGeneration
    CsmForConditionalGeneration.from_pretrained(os.getenv("MODEL_NAME"), use_auth_token=os.getenv("HF_TOKEN"))
    print("[build] Model download and load successful at build time.")
except Exception as e:
    print("[build] Build-time model download/load failed:", e)
    traceback.print_exc()
    print("[build] Proceeding, will load model at runtime instead.")
PYCODE

# Copy handler
COPY handler.py .

# Expose port if needed (most serverless images don't need manual exposure)
CMD ["python", "handler.py"]
