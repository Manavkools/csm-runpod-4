# csm-api-server (Sesame CSM wrapper)

This repository contains a FastAPI wrapper around Sesame CSM (https://github.com/SesameAILabs/csm).

Files created:
- api.py          : FastAPI server with /health, /infer (audio->json), /generate (text->wav)
- generate_cli.py : small wrapper that attempts to call the CSM generator
- start.sh        : downloads model (via huggingface_hub) if missing, renders INFERENCE_CMD and starts api.py
- Dockerfile      : CUDA-friendly Dockerfile (adjust CUDA base or torch wheel for your node)
- requirements.txt
- .dockerignore

IMPORTANT SECURITY NOTES
- Do NOT embed your HF_TOKEN in files. Provide HF_TOKEN as a secret/environment at deploy time.
- Pick a Runpod GPU that matches the CUDA version you use for torch.
- For correct PyTorch + CUDA support, provide TORCH_WHEEL_URL at build time or use a Runpod base image with PyTorch installed.

Quick test (after container is running and model is present):
- Health: GET /health
- Generate: POST /generate with JSON: {"text":"Hey my name is manav"} -> returns WAV

README: adapt to your needs.
