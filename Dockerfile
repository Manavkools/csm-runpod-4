# ===========================
# Dockerfile for csm-runpod (with model preloading)
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

# Cache directories for model files
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONUNBUFFERED=1

# Copy handler script
COPY handler.py .

# --- 🧠 Build-time model pre-download ---
# The model will be downloaded if HF_TOKEN is set in the environment.
RUN echo '\
import os, traceback; \
model_name = os.getenv("MODEL_NAME", "sesame/csm-1b"); \
hf_token = os.getenv("HF_TOKEN"); \
print(f"[build] Attempting to pre-download model: {model_name}"); \
print(f"[build] HF_TOKEN present? {bool(hf_token)}"); \
try: \
    from transformers import AutoProcessor; \
    processor = AutoProcessor.from_pretrained(model_name, use_auth_token=hf_token); \
    from transformers import AutoModelForConditionalGeneration; \
    print("[build] AutoProcessor loaded successfully."); \
    try: \
        from transformers import CsmForConditionalGeneration as ModelClass; \
        print("[build] Using CsmForConditionalGeneration."); \
    except Exception: \
        from transformers import AutoModelForConditionalGeneration as ModelClass; \
        print("[build] CSM not found, falling back to AutoModelForConditionalGeneration."); \
    model = ModelClass.from_pretrained(model_name, use_auth_token=hf_token); \
    print("[build] ✅ Model preloaded successfully and cached to:", os.getenv("HF_HOME")); \
except Exception as e: \
    print("[build] ⚠️ Model pre-download failed, will load at runtime instead."); \
    traceback.print_exc(); \
' | python3

# --- Add lightweight runtime token check utility ---
RUN echo 'echo "[startup check] HF whoami:" && \
python3 - <<PY \
import os, requests; t=os.getenv("HF_TOKEN"); \
if not t: print("[whoami] No token set."); \
else: \
    r=requests.get("https://huggingface.co/api/whoami-v2", headers={"Authorization":f"Bearer {t}"}); \
    print(r.status_code, r.text[:200]) \
PY' >> /usr/local/bin/check_token && chmod +x /usr/local/bin/check_token

# --- ✅ Default container start command ---
CMD ["python", "handler.py"]
