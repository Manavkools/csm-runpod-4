# handler.py
import os
import io
import base64
import traceback
import torch
import soundfile as sf
from transformers import CsmForConditionalGeneration, AutoProcessor
import runpod

# --- Config / env ---
MODEL_NAME = os.getenv("MODEL_NAME", "sesame/csm-1b")
HF_TOKEN = os.getenv("HF_TOKEN", None)
DEFAULT_SAMPLING_RATE = int(os.getenv("SAMPLING_RATE", "24000"))
# Device selection
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load model + processor once at startup ---
print(f"[handler] Loading model '{MODEL_NAME}' on device={device} ...")
try:
    # AutoProcessor or CsmProcessor depending on versions; AutoProcessor should work per README
    # use_auth_token argument is used to allow private / gated HF models; older/newer HF versions may accept token differently
    processor = AutoProcessor.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
    model = CsmForConditionalGeneration.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
    # Move model to device; for large models we rely on device_map or .to()
    try:
        model.to(device)
    except Exception:
        # Some HF models accept device_map; if CPU or simple GPU, fallback to .to() which we already tried
        pass
    if hasattr(model, "eval"):
        model.eval()
    print("[handler] Model & processor loaded successfully.")
except Exception as e:
    print("[handler] ERROR loading model:", e)
    traceback.print_exc()
    # If model didn't load, let RunPod fail early on handler init
    raise

def _audio_to_wav_bytes(audio_array, samplerate=DEFAULT_SAMPLING_RATE):
    """
    Accepts: numpy array or torch tensor, shape: (samples,) or (channels, samples)
    Returns: bytes of WAV file
    """
    # Convert torch -> numpy
    if hasattr(audio_array, "detach"):
        audio_array = audio_array.detach().cpu().numpy()

    import numpy as np

    audio = np.array(audio_array)

    # If shape (n,) -> make (1, n)
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)

    # audio is expected as (channels, samples) -> soundfile wants (samples, channels)
    # transpose to (samples, channels)
    audio_out = audio.T

    # ensure float32 in [-1,1] if required
    if audio_out.dtype.kind != 'f':
        # if integer type, normalize
        try:
            maxval = np.iinfo(audio_out.dtype).max
            audio_out = audio_out.astype('float32') / float(maxval)
        except Exception:
            audio_out = audio_out.astype('float32')

    # write to buffer
    buf = io.BytesIO()
    sf.write(buf, audio_out, samplerate, format="WAV")
    buf.seek(0)
    return buf.read()

def synthesize(text: str, speaker: str = None):
    """
    Wraps the typical usage from the CSM README.
    Returns raw wav bytes (not base64).
    """
    # Prepare inputs using the processor - the README shows processor(text, add_special_tokens=True)
    inputs = processor(text, add_special_tokens=True, return_tensors="pt")

    # Move tensors to device
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    # Generate - many variants from README use output_audio=True
    with torch.no_grad():
        try:
            outputs = model.generate(**inputs, output_audio=True)
        except TypeError:
            # Older/newer versions might require different kwargs; try without explicit flag
            outputs = model.generate(**inputs)

    # outputs could be: tensor, list, tuple, numpy, or dict
    audio = None
    if isinstance(outputs, (list, tuple)):
        audio = outputs[0]
    else:
        audio = outputs

    if isinstance(audio, dict) and "audio" in audio:
        audio = audio["audio"]

    wav_bytes = _audio_to_wav_bytes(audio, samplerate=DEFAULT_SAMPLING_RATE)
    return wav_bytes

def handler(job):
    """
    RunPod standard handler signature:
      job is a dict with key "input" (a dict)
    Expected input example:
      {"input": {"text": "Hey my name is Manav", "speaker": "0"}}
    Returns:
      {"audio_base64": "<base64 WAV>", "sampling_rate": 24000}
    """
    try:
        job_input = job.get("input", {}) or {}
        text = job_input.get("text")
        speaker = job_input.get("speaker", None)

        if not text or not isinstance(text, str) or not text.strip():
            return {"error": "Missing required 'text' (non-empty string)."}

        text = text.strip()
        # Optionally embed speaker token if user provides numeric speaker id like "0"
        if speaker is not None and isinstance(speaker, (str, int)):
            if isinstance(speaker, int) or (isinstance(speaker, str) and speaker.isdigit()):
                # Prepend bracketed speaker token if user didn't already include it
                if not text.startswith("["):
                    text = f"[{speaker}]" + text

        wav_bytes = synthesize(text, speaker=speaker)
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
        return {"sampling_rate": DEFAULT_SAMPLING_RATE, "audio_base64": audio_b64}

    except Exception as e:
        tb = traceback.format_exc()
        print("[handler] Exception during inference:", str(e))
        print(tb)
        # Raise to let RunPod mark job FAILED and surface the traceback
        raise RuntimeError(f"Inference failed: {e}\n{tb}")

# Start RunPod serverless (RunPod will call the handler when /run is hit)
runpod.serverless.start({"handler": handler})
