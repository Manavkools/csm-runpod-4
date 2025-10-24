# ===========================
# Dockerfile for csm-runpod
# ===========================

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libsndfile1 \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Optional cache locations
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONUNBUFFERED=1

# Copy handler script
COPY handler.py .

# Add lightweight runtime debug for token validation
RUN echo 'echo "[startup check] HF whoami:" && \
python3 - <<PY \
import os, requests; t=os.getenv("HF_TOKEN"); \
r=requests.get("https://huggingface.co/api/whoami-v2", headers={"Authorization":f"Bearer {t}"}); \
print(r.status_code, r.text[:200]) \
PY' >> /usr/local/bin/check_token && chmod +x /usr/local/bin/check_token

# Final start command
CMD ["python", "handler.py"]
