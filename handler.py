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

print(f"[handler] INIT | device={device}, MODEL={MODEL_NAME}, HF_TOKEN_SET={bool(HF_TOKEN)}")

processor, model, ModelClass = None, None, None


def _try_import_csm():
    """Try importing CSM model from multiple potential paths."""
    try:
        from transformers import CsmForConditionalGeneration
        print("[handler] ✅ Imported CsmForConditionalGeneration from transformers root.")
        return CsmForConditionalGeneration
    except Exception:
        pass
    try:
        mod = importlib.import_module("transformers.models.csm.modeling_csm")
        cls = getattr(mod, "CsmForConditionalGeneration", None)
        if cls:
            print("[handler] ✅ Imported CsmForConditionalGeneration from transformers.models.csm.")
            return cls
    except Exception:
        pass
    print("[handler] ⚠️ Falling back to AutoModelForConditionalGeneration.")
    from transformers import AutoModelForConditionalGeneration
    return AutoModelForConditionalGeneration


def load_model_runtime():
    """Load model and processor dynamically if missing."""
    global processor, model, ModelClass

    if processor is None:
        print(f"[handler] Loading processor for {MODEL_NAME} ...")
        kwargs = {"use_auth_token": HF_TOKEN} if HF_TOKEN else {}
        processor = AutoProcessor.from_pretrained(MODEL_NAME, **kwargs)
        print("[handler] ✅ Processor loaded.")

    if model is None:
        ModelClass = _try_import_csm()
        kwargs = {"use_auth_token": HF_TOKEN} if HF_TOKEN else {}

        try:
            print("[handler] Attempting model.load(device_map='auto') ...")
            model = ModelClass.from_pretrained(
                MODEL_NAME, device_map="auto", torch_dtype=torch.float16, **kwargs
            )
            print("[handler] ✅ Model loaded with device_map='auto'.")
        except Exception as e:
            print("[handler] ⚠️ device_map load failed:", e)
            print("[handler] Loading model on CPU or direct device.")
            model = ModelClass.from_pretrained(MODEL_NAME, **kwargs)
            model.to(device)
            print(f"[handler] ✅ Model moved to {device}.")

        if hasattr(model, "eval"):
            model.eval()
        print("[handler] ✅ Model ready.")


def _audio_to_wav_bytes(audio_array, samplerate=DEFAULT_SAMPLING_RATE):
    """Convert numpy or tensor audio to WAV bytes."""
    import numpy as np
    if hasattr(audio_array, "detach"):
        audio_array = audio_array.detach().cpu().numpy()
    audio = np.array(audio_array)
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)
    buf = io.BytesIO()
    sf.write(buf, audio.T, samplerate, format="WAV")
    buf.seek(0)
    return buf.read()


def synthesize(text: str, speaker: str = None):
    """Generate audio for text input."""
    global processor, model
    if processor is None or model is None:
        load_model_runtime()

    inputs = processor(text, add_special_tokens=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        try:
            outputs = model.generate(**inputs, output_audio=True)
        except TypeError:
            outputs = model.generate(**inputs)

    audio = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    if isinstance(audio, dict) and "audio" in audio:
        audio = audio["audio"]

    return _audio_to_wav_bytes(audio)


def handler(job):
    job_input = job.get("input", {}) or {}
    text = job_input.get("text")
    speaker = job_input.get("speaker", None)

    if not text or not text.strip():
        return {"error": "Missing 'text' input."}

    text = text.strip()
    if speaker and str(speaker).isdigit() and not text.startswith("["):
        text = f"[{speaker}]" + text

    try:
        wav_bytes = synthesize(text, speaker=speaker)
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
        return {"sampling_rate": DEFAULT_SAMPLING_RATE, "audio_base64": audio_b64}
    except Exception as e:
        tb = traceback.format_exc()
        print("[handler] ❌ Inference failed:", e)
        print(tb)
        raise RuntimeError(f"Inference failed: {e}\n{tb}")


runpod.serverless.start({"handler": handler})
