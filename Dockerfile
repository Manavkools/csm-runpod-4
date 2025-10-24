# ===============================
# 🚀 CSM RunPod Serverless Dockerfile
# ===============================
FROM python:3.10-slim

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install essential build and audio libs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential git ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency list and install
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Setup cache locations for Hugging Face
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface

# Arguments (passed during build)
ARG MODEL_NAME=sesame/csm-1b

# Pass through to environment so Python sees them
# WARNING: Hardcoding secrets like HF_TOKEN is a security risk.
# See "Recommended Method" below.
ENV HF_TOKEN=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
ENV MODEL_NAME=${MODEL_NAME}

# Attempt to pre-download model at build time (safe fallback)
RUN python3 - <<'PYCODE'
import os, traceback
print(f"[build] Attempting to pre-download: {os.getenv('MODEL_NAME')}")
try:
    from transformers import AutoProcessor
    # Note: Using 'token' is preferred for newer HF versions
    # The handler.py logic correctly handles both 'token' and 'use_auth_token'
    processor = AutoProcessor.from_pretrained(os.getenv("MODEL_NAME"), token=os.getenv("HF_TOKEN"))
    from transformers import CsmForConditionalGeneration
    model = CsmForConditionalGeneration.from_pretrained(os.getenv("MODEL_NAME"), token=os.getenv("HF_TOKEN"))
    print("[build] ✅ Model pre-downloaded successfully.")
except Exception as e:
    print("\n[build] ⚠️ Model download failed, will fallback to runtime load.")
    traceback.print_exc()
PYCODE

# Copy source code
COPY handler.py .

# Environment defaults
ENV PYTHONUNBUFFERED=1
ENV RUNPOD_DEBUG_LEVEL=info

# Entrypoint for RunPod serverless worker
CMD ["python3", "handler.py"]
