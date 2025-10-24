# csm-runpod

This repository contains a RunPod serverless handler for Sesame CSM-1B.

## Files created
- handler.py      -> RunPod-compatible serverless handler that loads the model and returns base64 WAV
- requirements.txt
- Dockerfile      -> optional example; adapt CUDA/tensor versions per RunPod image choice
- README.md

## Environment variables required
- HF_TOKEN         : Hugging Face token with read access (for sesame/csm-1b)
- MODEL_NAME       : (optional) default "sesame/csm-1b"
- SAMPLING_RATE    : (optional) default 24000

## Quick usage notes
1. Upload this project to RunPod or build the image in the environment of your choice.
2. Ensure the container/runtime has a CUDA-compatible PyTorch if using GPU.
3. Set HF_TOKEN in RunPod environment so the handler can download the model.
4. RunPod will call handler(job) when POSTing to /run; the handler returns JSON:
   { "sampling_rate": 24000, "audio_base64": "<base64 WAV>" }

## Example job payload
{
  "input": {
    "text": "Hey my name is Manav",
    "speaker": "0"
  }
}

## Decode output to file locally
# Save base64 to file (resp_base64.txt), then:
python - <<PY
import base64
b = base64.b64decode(open('resp_base64.txt','r').read())
open('out.wav','wb').write(b)
PY

## Notes and caveats
- Large model download at cold start; ensure sufficient GPU memory.
- Torch/CUDA version must match the runtime.
- If you prefer a pre-built RunPod image with matching torch + CUDA, use that in their UI to avoid building a heavy image.
