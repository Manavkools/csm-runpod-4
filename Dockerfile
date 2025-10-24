# Dockerfile - csm-runpod
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps needed for audio libs and building some wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      libsndfile1 \
      git \
      curl \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# copy requirements (adjust as needed)
COPY requirements.txt .

# ensure modern pip/wheel
RUN python -m pip install --upgrade pip setuptools wheel

# Install Python deps (no-cache to reduce image size)
RUN pip install --no-cache-dir -r requirements.txt

# Optional cache locations for huggingface
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONUNBUFFERED=1

# Pass HF_TOKEN at build time to pre-download model into image (optional)
ARG HF_TOKEN
# Set runtime env var (RunPod will set HF_TOKEN at runtime, this just sets HUGGING_FACE_HUB_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
ENV HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}

# Attempt to download and load the model at build-time if HF_TOKEN provided.
# This caches model files into the image so runtime doesn't have to download them.
# If token not provided or model access denied, continue (runtime will load).
RUN python3 - <<'PYCODE' || true
import os, traceback
from huggingface_hub import login
from transformers import AutoProcessor
MODEL = os.getenv("MODEL_NAME", "sesame/csm-1b")
hf_token = os.getenv("HF_TOKEN")
try:
    if hf_token:
        # login writes token to cache and ensures API access
        login(token=hf_token)
        print("[build] HF login succeeded (token provided).")
    else:
        print("[build] No HF_TOKEN provided at build-time; skipping model predownload.")
        raise SystemExit(0)

    print("[build] Attempting to download processor/config for", MODEL)
    AutoProcessor.from_pretrained(MODEL, use_auth_token=hf_token)
    # attempt to import and instantiate CSM model class if available
    try:
        from transformers import CsmForConditionalGeneration
        CsmForConditionalGeneration.from_pretrained(MODEL, use_auth_token=hf_token)
        print("[build] Model downloaded and loaded successfully at build time.")
    except Exception as e:
        print("[build] Could not load CSM model class at build time (okay).", e)
except Exception as e:
    print("[build] Build-time model download failed or skipped:", e)
    traceback.print_exc()
PYCODE

# Copy handler last (so install steps cached earlier)
COPY handler.py .

# Default runtime command - RunPod runs container and hits /run endpoint via runpod.serverless.start
CMD ["python", "handler.py"]
