#!/usr/bin/env python3
import os
import json
import uuid
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

# ----------------- Config -----------------
# Must contain placeholders: {model_dir} (optional), {input}, {output}
# Example:
# INFERENCE_CMD="python generate_cli.py --model-dir {model_dir} --text-file {input} --output {output}"
INFERENCE_CMD = os.getenv(
    "INFERENCE_CMD",
    "python generate_cli.py --model-dir {model_dir} --text-file {input} --output {output}"
)

WORK_DIR = Path(os.getenv("WORK_DIR", "/tmp/csm_api"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# -------------- FastAPI app --------------
app = FastAPI(title="Sesame CSM API", version="1.0.0")

class HealthResp(BaseModel):
    status: str
    inference_cmd: str

@app.get("/health", response_model=HealthResp)
def health():
    return HealthResp(status="ok", inference_cmd=INFERENCE_CMD)

def run_inference_cli(input_path: str, output_path: str, timeout: int = 3600) -> dict:
    """
    Run the CLI defined by INFERENCE_CMD.
    INFERENCE_CMD must contain {input} and {output} placeholders (may also contain {model_dir}).
    """
    if "{input}" not in INFERENCE_CMD or "{output}" not in INFERENCE_CMD:
        raise RuntimeError("INFERENCE_CMD must contain {input} and {output} placeholders")

    cmd = INFERENCE_CMD.format(
        input=input_path,
        output=output_path,
        model_dir=os.getenv("MODEL_DIR", "/workspace/csm-1b")
    )
    logger.info(f"Running inference command: {cmd}")

    try:
        completed = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        logger.info(f"[stdout] {completed.stdout[:2000]}")
        logger.info(f"[stderr] {completed.stderr[:2000]}")

        if completed.returncode != 0:
            raise RuntimeError(
                f"Inference command failed (rc={completed.returncode}): {completed.stderr}"
            )

        # Prefer JSON file written by command
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except Exception:
                    return {"stdout": completed.stdout}

        # Or try to parse stdout as JSON
        try:
            return json.loads(completed.stdout)
        except Exception:
            return {"stdout": completed.stdout}

    except subprocess.TimeoutExpired:
        raise RuntimeError("Inference command timed out")
    except Exception as e:
        raise

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    """
    Upload an audio file (form field name: 'file').
    The server writes it to a temp dir, runs INFERENCE_CMD and returns JSON.
    (Matches your earlier pipeline behavior for transcription.)
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    if file.content_type and not file.content_type.startswith("audio"):
        logger.warning(f"Unexpected content-type: {file.content_type}")

    req_id = uuid.uuid4().hex
    req_dir = WORK_DIR / req_id
    req_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename).suffix or ".wav"
    input_path = str(req_dir / f"input{suffix}")
    output_path = str(req_dir / "out.json")

    try:
        with open(input_path, "wb") as f:
            f.write(await file.read())

        if Path(input_path).stat().st_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        try:
            result = run_inference_cli(input_path=input_path, output_path=output_path)
        except Exception as e:
            logger.exception("Inference failed")
            raise HTTPException(status_code=500, detail=f"Inference error: {e}")

        return JSONResponse(result)

    finally:
        try:
            shutil.rmtree(req_dir)
        except Exception:
            pass

@app.post("/generate")
async def generate(payload: dict = Body(...)):
    """
    POST /generate
    Body: {"text": "Hey my name is manav", ...}
    Writes text to a temp input file, calls INFERENCE_CMD and returns generated WAV binary.
    INFERENCE_CMD should write an audio file to {output}.
    """
    text = payload.get("text")
    if not text or not isinstance(text, str):
        raise HTTPException(status_code=400, detail="Missing 'text' in request body")

    req_id = uuid.uuid4().hex
    req_dir = WORK_DIR / req_id
    req_dir.mkdir(parents=True, exist_ok=True)

    input_path = str(req_dir / "input.txt")
    output_path = str(req_dir / "out.wav")

    try:
        with open(input_path, "w", encoding="utf-8") as f:
            f.write(text)

        try:
            run_inference_cli(input_path=input_path, output_path=output_path)
        except Exception as e:
            logger.exception("Generation failed")
            raise HTTPException(status_code=500, detail=f"Generation error: {e}")

        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Generation finished but output file missing")

        def iterfile(path):
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    yield chunk

        headers = {"Content-Disposition": f'attachment; filename="out.wav"'}
        return StreamingResponse(iterfile(output_path), media_type="audio/wav", headers=headers)

    finally:
        try:
            shutil.rmtree(req_dir)
        except Exception:
            pass

if __name__ == "__main__":
    uvicorn.run(
        app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)), log_level="info"
    )
