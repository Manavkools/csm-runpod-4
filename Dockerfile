FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libsndfile1 git curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONUNBUFFERED=1

COPY handler.py .

# --- 🧠 Model pre-download (safe HEREDOC version) ---
RUN --mount=type=cache,target=/workspace/.cache/huggingface \
    python3 <<'PYCODE'
import os, traceback
model_name = os.getenv("MODEL_NAME", "sesame/csm-1b")
hf_token = os.getenv("HF_TOKEN")
print(f"[build] Attempting to pre-download model: {model_name}")
print(f"[build] HF_TOKEN present? {bool(hf_token)}")

try:
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_name, use_auth_token=hf_token)
    from transformers import AutoModelForConditionalGeneration
    print("[build] AutoProcessor loaded successfully.")
    try:
        from transformers import CsmForConditionalGeneration as ModelClass
        print("[build] Using CsmForConditionalGeneration.")
    except Exception:
        from transformers import AutoModelForConditionalGeneration as ModelClass
        print("[build] CSM not found, falling back to AutoModelForConditionalGeneration.")
    model = ModelClass.from_pretrained(model_name, use_auth_token=hf_token)
    print("[build] ✅ Model preloaded successfully and cached to:", os.getenv("HF_HOME"))
except Exception as e:
    print("[build] ⚠️ Model pre-download failed, will load at runtime instead.")
    traceback.print_exc()
PYCODE

CMD ["python", "handler.py"]
