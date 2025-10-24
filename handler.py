# handler.py (improved)
import os
import io
import base64
import traceback
import importlib
import torch
import soundfile as sf
from transformers import AutoProcessor
import runpod

MODEL_NAME = os.getenv("MODEL_NAME", "sesame/csm-1b")
HF_TOKEN = os.getenv("HF_TOKEN", None)
DEFAULT_SAMPLING_RATE = int(os.getenv("SAMPLING_RATE", "24000"))
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[handler] STARTUP: device={device}, MODEL_NAME={MODEL_NAME}, HF_TOKEN_set={bool(HF_TOKEN)}")

processor = None
model = None
ModelClass = None

# Prefer top-level CSM import, fallback to modeling module or AutoModel
def _try_import_csm():
    try:
        from transformers import CsmForConditionalGeneration
        print("[handler] Imported CsmForConditionalGeneration (top-level).")
        return CsmForConditionalGeneration
    except Exception:
        print("[handler] CsmForConditionalGeneration not found at top-level.")
    try:
        mod = importlib.import_module("transformers.models.csm.modeling_csm")
        cls = getattr(mod, "CsmForConditionalGeneration", None)
        if cls:
            print("[handler] Imported CsmForConditionalGeneration from transformers.models.csm.modeling_csm.")
            return cls
    except Exception:
        pass
    print("[handler] Falling back to AutoModelForConditionalGeneration.")
    from transformers import AutoModelForConditionalGeneration
    return AutoModelForConditionalGeneration

# Helper to select correct HF token kwarg for different HF versions
def _hf_token_kwargs():
    if not HF_TOKEN:
        return {}
    # newer HF versions accept token=..., older use use_auth_token=...
    return {"token": HF_TOKEN} if "token" in importlib.import_module("inspect").signature(AutoProcessor.from_pretrained).parameters else {"use_auth_token": HF_TOKEN}

def load_model_runtime():
    global processor, model, ModelClass
    if processor is None:
        print(f"[handler] Loading processor for {MODEL_NAME} ...")
        try:
            processor = AutoProcessor.from_pretrained(MODEL_NAME, **_hf_token_kwargs())
            print("[handler] Processor loaded.")
        except Exception as e:
            print("[handler] Processor load failed:", e)
            raise

    if model is None:
        ModelClass = _try_import_csm()
        load_kwargs = _hf_token_kwargs()
        # Try device_map first on GPU
        if device == "cuda":
            # attempt to load with device_map and float16 to reduce memory
            try:
                print("[handler] Attempting to load model with device_map='auto' and torch_dtype=float16 ...")
                model = ModelClass.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16, **load_kwargs)
                print("[handler] Model loaded with device_map='auto'.")
            except Exception as e:
                print("[handler] device_map load failed, trying fallback load:", e)
                try:
                    model = ModelClass.from_pretrained(MODEL_NAME, **load_kwargs)
                    try:
                        model.to(device)
                        print("[handler] Model moved to device:", device)
                    except Exception as me:
                        print("[handler] Warning moving model to device failed:", me)
                except Exception as e2:
                    print("[handler] Full fallback load failed:", e2)
                    raise
        else:
            print("[handler] Loading model on CPU ...")
            model = ModelClass.from_pretrained(MODEL_NAME, **load_kwargs)
            print("[handler] Model loaded on CPU.")

        if hasattr(model, "eval"):
            model.eval()
        print("[handler] Model & processor ready. Model type:", type(model))

# Try one-time load on import (will fallback to runtime)
try:
    print("[handler] Import-time: checking model availability...")
    if model is None:
        load_model_runtime()
except Exception as e:
    print("[handler] Import-time load failed (will attempt runtime). Error:", e)
    traceback.print_exc()

def _audio_to_wav_bytes(audio_array, samplerate=DEFAULT_SAMPLING_RATE):
    import numpy as np
    if hasattr(audio_array, "detach"):
        audio_array = audio_array.detach().cpu().numpy()
    audio = np.array(audio_array)
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)
    audio_out = audio.T
    if audio_out.dtype.kind != 'f':
        try:
            maxval = np.iinfo(audio_out.dtype).max
            audio_out = audio_out.astype('float32') / float(maxval)
        except Exception:
            audio_out = audio_out.astype('float32')
    buf = io.BytesIO()
    sf.write(buf, audio_out, samplerate, format="WAV")
    buf.seek(0)
    return buf.read()

def synthesize(text: str, speaker: str = None) -> bytes:
    global processor, model
    if processor is None or model is None:
        print("[handler] Processor/model not ready — loading at runtime.")
        load_model_runtime()

    inputs = processor(text, add_special_tokens=True, return_tensors="pt")
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    with torch.no_grad():
        try:
            outputs = model.generate(**inputs, output_audio=True)
            print("[handler] generate(..., output_audio=True) OK.")
        except TypeError:
            print("[handler] generate(..., output_audio=True) not supported, retrying without that kwarg.")
            outputs = model.generate(**inputs)
        except Exception as e:
            print("[handler] model.generate() raised:", e)
            raise

    audio = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    if isinstance(audio, dict) and "audio" in audio:
        audio = audio["audio"]
    if isinstance(audio, (list, tuple)) and len(audio) > 0:
        audio = audio[0]
    return _audio_to_wav_bytes(audio, samplerate=DEFAULT_SAMPLING_RATE)

def handler(job):
    job_input = job.get("input", {}) or {}
    text = job_input.get("text")
    speaker = job_input.get("speaker", None)
    if not text or not isinstance(text, str) or not text.strip():
        return {"error": "Missing required 'text' (non-empty string)."}
    text = text.strip()
    if speaker is not None and isinstance(speaker, (str, int)):
        if isinstance(speaker, int) or (isinstance(speaker, str) and speaker.isdigit()):
            if not text.startswith("["):
                text = f"[{speaker}]" + text
    try:
        wav_bytes = synthesize(text, speaker=speaker)
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
        return {"sampling_rate": DEFAULT_SAMPLING_RATE, "audio_base64": audio_b64}
    except Exception as e:
        tb = traceback.format_exc()
        print("[handler] Exception during inference:", e)
        print(tb)
        # Raise so RunPod records a failure and shows the traceback
        raise RuntimeError(f"Inference failed: {e}\n{tb}")

# Start the RunPod serverless loop (RunPod will call handler when /run is hit)
runpod.serverless.start({"handler": handler})
