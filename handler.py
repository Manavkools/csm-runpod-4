# handler.py
import os
import io
import base64
import traceback
import importlib
import torch
import soundfile as sf

# Make HF_TOKEN available at top-level from env as requested
import os as _os
HF_TOKEN = _os.getenv("HF_TOKEN")

# Explicit HF login to ensure gated repos accessible (will not print token)
try:
    if HF_TOKEN:
        from huggingface_hub import login as _hf_login
        try:
            _hf_login(token=HF_TOKEN)
            print("[startup] huggingface_hub.login() succeeded (HF_TOKEN provided).")
        except Exception as e:
            print("[startup] huggingface_hub.login() failed:", str(e))
    else:
        print("[startup] HF_TOKEN not set. If model is gated/private, runtime load will fail.")
except Exception as e:
    print("[startup] Warning: huggingface_hub login attempt raised:", str(e))

# Import runpod only when available
try:
    import runpod
except Exception:
    runpod = None
    print("[startup] runpod SDK not available; local testing only.")

# Config / env
MODEL_NAME = os.getenv("MODEL_NAME", "sesame/csm-1b")
DEFAULT_SAMPLING_RATE = int(os.getenv("SAMPLING_RATE", "24000"))
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[handler] STARTUP: device={device}, MODEL_NAME={MODEL_NAME}, HF_TOKEN_set={bool(HF_TOKEN)}")

# Lazy-loaded objects
processor = None
model = None
ModelClass = None

def _try_import_csm():
    """
    Try to get the CSM model class. If not present, fallback to AutoModelForConditionalGeneration.
    """
    try:
        from transformers import CsmForConditionalGeneration
        print("[handler] Imported CsmForConditionalGeneration from transformers.")
        return CsmForConditionalGeneration
    except Exception:
        print("[handler] CsmForConditionalGeneration not found in top-level transformers import.")
    try:
        mod = importlib.import_module("transformers.models.csm.modeling_csm")
        cls = getattr(mod, "CsmForConditionalGeneration", None)
        if cls:
            print("[handler] Imported CsmForConditionalGeneration from transformers.models.csm.modeling_csm.")
            return cls
    except Exception:
        pass
    # fallback
    from transformers import AutoModelForConditionalGeneration
    print("[handler] Falling back to AutoModelForConditionalGeneration.")
    return AutoModelForConditionalGeneration

def load_model_runtime():
    """
    Load processor and model at runtime. Uses HF_TOKEN from env if set.
    Attempts device_map='auto' with float16 if CUDA available, otherwise loads on CPU.
    """
    global processor, model, ModelClass
    kwargs = {}
    if HF_TOKEN:
        kwargs["use_auth_token"] = HF_TOKEN

    if processor is None:
        print("[handler] Loading processor...")
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(MODEL_NAME, **kwargs)
        print("[handler] Processor loaded.")

    if model is None:
        ModelClass = _try_import_csm()
        try:
            if device == "cuda":
                # attempt efficient load
                try:
                    print("[handler] Loading model with device_map='auto', torch_dtype=float16 ...")
                    model = ModelClass.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16, **kwargs)
                    print("[handler] Model loaded with device_map='auto'.")
                except Exception as e:
                    print("[handler] device_map load failed:", str(e))
                    print("[handler] Falling back to normal from_pretrained() and model.to(device).")
                    model = ModelClass.from_pretrained(MODEL_NAME, **kwargs)
                    try:
                        model.to(device)
                        print("[handler] Model moved to device:", device)
                    except Exception as me:
                        print("[handler] Warning moving model to device failed:", me)
            else:
                print("[handler] Loading model on CPU.")
                model = ModelClass.from_pretrained(MODEL_NAME, **kwargs)
                print("[handler] Model loaded on CPU.")
        except Exception as e:
            print("[handler] ERROR loading model at runtime:", e)
            traceback.print_exc()
            raise
        if hasattr(model, "eval"):
            model.eval()
        print("[handler] Model & processor ready. Model type:", type(model))

def _audio_to_wav_bytes(audio_array, samplerate=DEFAULT_SAMPLING_RATE):
    """
    Convert numpy/torch array (channels,samples) or (samples,) -> WAV bytes.
    """
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
            audio_out = audio_out.astype("float32") / float(maxval)
        except Exception:
            audio_out = audio_out.astype("float32")
    buf = io.BytesIO()
    sf.write(buf, audio_out, samplerate, format="WAV")
    buf.seek(0)
    return buf.read()

def synthesize(text: str, speaker: str = None) -> bytes:
    """
    Run model.generate and return raw WAV bytes.
    """
    global processor, model
    if processor is None or model is None:
        print("[handler] Processor/model not ready. Loading at runtime...")
        load_model_runtime()

    # Prepare inputs
    inputs = processor(text, add_special_tokens=True, return_tensors="pt")
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    # Generate with robust fallbacks
    with torch.no_grad():
        try:
            outputs = model.generate(**inputs, output_audio=True)
            print("[handler] model.generate(..., output_audio=True) succeeded.")
        except TypeError:
            print("[handler] generate() does not accept output_audio; retrying without it.")
            outputs = model.generate(**inputs)
        except Exception as e:
            print("[handler] model.generate raised exception:", e)
            raise

    # Extract audio: support multiple possible output formats
    audio = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    if isinstance(audio, dict) and "audio" in audio:
        audio = audio["audio"]
    if isinstance(audio, (list, tuple)) and len(audio) > 0:
        audio = audio[0]
    return _audio_to_wav_bytes(audio, samplerate=DEFAULT_SAMPLING_RATE)

def handler(job):
    """
    RunPod handler signature expecting:
      job["input"] = {"text": "...", "speaker": "0" (optional)}
    """
    job_input = (job or {}).get("input", {}) or {}
    text = job_input.get("text")
    speaker = job_input.get("speaker", None)

    if not text or not isinstance(text, str) or not text.strip():
        return {"error": "Missing required 'text' (non-empty string)."}

    text = text.strip()
    # Embed speaker token if provided and numeric
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
        print("[handler] Inference exception:", e)
        print(tb)
        # raise runtime error so RunPod marks job as failed and surfaces logs
        raise RuntimeError(f"Inference failed: {e}\n{tb}")

# If runpod available, start serverless handler
if runpod is not None:
    try:
        runpod.serverless.start({"handler": handler})
    except Exception as e:
        print("[startup] runpod.serverless.start() raised:", e)
        print("If running locally, call handler(...) directly for testing.")
else:
    print("[startup] runpod SDK not installed; module will not start serverless automatically.")
    print("For local testing you can call handler({'input': {'text': 'hi'}}) manually.")
