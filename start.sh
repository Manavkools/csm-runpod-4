#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/workspace/csm-1b}"
HF_REPO_ID="${HF_REPO_ID:-sesame/csm-1b}"
# Default inference command (will be rendered with model_dir)
INFERENCE_CMD="${INFERENCE_CMD:-python generate_cli.py --model-dir {model_dir} --text-file {input} --output {output}}"

echo "==== start.sh ===="
echo "MODEL_DIR: $MODEL_DIR"
echo "HF_REPO_ID: $HF_REPO_ID"
echo "WORK_DIR: ${WORK_DIR:-/tmp/csm_api}"
echo "PORT: ${PORT:-8000}"

model_has_weights() {
  if [ -d "$MODEL_DIR" ] && (ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1 || ls "$MODEL_DIR"/*.pt >/dev/null 2>&1 || [ -f "$MODEL_DIR/config.json" ] ); then
    return 0
  fi
  return 1
}

if model_has_weights; then
  echo "Model appears present in ${MODEL_DIR} — skipping download."
else
  echo "Model not found at ${MODEL_DIR}."
  if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN not set. Automatic download from Hugging Face may require authentication."
    echo "Please set HF_TOKEN as an environment variable/secret in your deployment platform."
  fi

  echo "Attempting to download model ${HF_REPO_ID} -> ${MODEL_DIR} ..."
  python - <<PY
import os, sys
from huggingface_hub import snapshot_download
from pathlib import Path
import shutil, tempfile
tmp = tempfile.mkdtemp(prefix="hf_download_")
try:
    path = snapshot_download(repo_id=os.environ.get("HF_REPO_ID", "$HF_REPO_ID"), cache_dir=tmp, repo_type="model", token=os.environ.get("HF_TOKEN", None))
    src = Path(path)
    dst = Path(os.environ.get("MODEL_DIR", "$MODEL_DIR"))
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_dir():
            shutil.copytree(item, dst / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst / item.name)
    print("Model downloaded into:", dst)
except Exception as e:
    print("ERROR: snapshot_download failed:", e, file=sys.stderr)
    sys.exit(2)
finally:
    try:
        shutil.rmtree(tmp)
    except Exception:
        pass
PY

  if model_has_weights; then
    echo "Model successfully downloaded into ${MODEL_DIR}."
  else
    echo "ERROR: model download finished but files not found in ${MODEL_DIR}."
    ls -la $(dirname "$MODEL_DIR") || true
    exit 1
  fi
fi

# Render INFERENCE_CMD: replace {model_dir} placeholder with the real path
RENDERED_CMD="$(python - <<PY
import os
cmd = os.environ.get("INFERENCE_CMD", "")
cmd = cmd.replace("{model_dir}", os.environ.get("MODEL_DIR", "$MODEL_DIR"))
print(cmd)
PY
)"
export INFERENCE_CMD="$RENDERED_CMD"

echo "Final INFERENCE_CMD: $INFERENCE_CMD"
mkdir -p "${WORK_DIR:-/tmp/csm_api}"
exec python api.py
