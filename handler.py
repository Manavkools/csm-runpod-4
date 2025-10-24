# handler.py
import os
import io
import base64
import traceback
import torch
import soundfile as sf
import importlib
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

def _try_import_csm():
    try:
        from transformers import CsmForConditionalGeneration
        print("[handler] Imported CsmForConditionalGeneration.")
        return CsmForConditionalGeneration
    except Exception:
        print("[handler] CsmForConditionalGeneration not found in top-level.")
    try:
        mod = importlib.import_module("transformers.models.csm.modeling_csm")
        cls = getattr(mod, "CsmForConditionalGeneration", None)
        if cls:
            print("[handler] Imported CsmForConditionalGeneration from modeling_csm.")
            return cls
    except Exception:
        pass
    print("[handler] Falling back to AutoModelForConditionalGeneration.")
    from transformers import AutoModelForConditionalGeneration
    return AutoModelForConditionalGeneration

def load_model_runtime():
    global processor, model, ModelClass
    if processor is None:
        print(f"[handler] Loading processor for {MODEL_NAME} ...")
        kwargs = {}
        if HF_TOKEN:
            kwargs["use_auth_token"] = HF_TOKEN
        processor = AutoProcessor.from_pretrained(MODEL_NAME, **kwargs)
        print("[handler] Processor loaded.")
    if model is None:
        ModelClass = _try_import_csm()
        kwargs = {}
        if HF_TOKEN:
            kwargs["use_auth_token"] = HF_TOKEN
        if device == "cuda":
            try:
                print("[handler] Attempting model.load with device_map='auto', torch_dtype=float16 ...")
                model = ModelClass.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16, **kwargs)
                print("[handler] Model loaded with device_map='auto'.")
            except Exception as e:
                print("[handler] device_map load failed:", e)
                print("[handler] Loading model normally and moving to device...")
                model = ModelClass.from_pretrained(MODEL_NAME, **kwargs)
                try:
                    model.to(device)
                    print("[handler] Model moved to device:", device)
                except Exception as me:
                    print("[handler] Warning moving model to device failed:", me)
        else:
            print("[handler] Device is CPU; loading model on CPU.")
            model = ModelClass.from_pretrained(MODEL_NAME, **kwargs)
            print("[handler] Model loaded on CPU.")
        if hasattr(model, "eval"):
            model.eval()
        print("[handler] Model & processor loaded successfully at runtime. Model type:", type(model))

# Try to load model at import time (if built properly)
try:
    print("[handler] Checking if model already loaded in image ...")
    # If we can instantiate the class (empty init) maybe okay; otherwise load
    _ = None
    if model is None:
        load_model_runtime()
except Exception as e:
    print("[handler] Exception in import-time load:", e)
    traceback.print_exc()
    print("[handler] Will continue, but requests may trigger runtime load.")

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
        print("[handler] Model or processor not ready — loading now at runtime.")
        load_model_runtime()
    inputs = processor(text, add_special_tokens=True, return_tensors="pt")
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k,v in inputs.items()}
    with torch.no_grad():
        try:
            outputs = model.generate(**inputs, output_audio=True)
            print("[handler] model.generate(..., output_audio=True) succeeded.")
        except TypeError:
            print("[handler] generate() with output_audio not supported, retrying...")
            outputs = model.generate(**inputs)
        except Exception as e:
            print("[handler] model.generate threw exception:", e)
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
    if speaker is not None and isinstance(speaker,(str,int)):
        if isinstance(speaker,int) or (isinstance(speaker,str) and speaker.isdigit()):
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
        raise RuntimeError(f"Inference failed: {e}\n{tb}")

runpod.serverless.start({"handler": handler})
