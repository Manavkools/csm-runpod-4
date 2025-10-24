# Dockerfile - CUDA-friendly. Adjust base image to your Runpod GPU/CUDA version if necessary.
FROM nvidia/cuda:12.6.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ARG TORCH_WHEEL_URL=""   # optional build-arg to install a specific torch wheel
ARG PYTHON_VER=3.10

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates curl git wget build-essential ffmpeg libsndfile1 python3 python3-venv python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10

RUN useradd -ms /bin/bash csmuser
USER csmuser
WORKDIR /home/csmuser/app

COPY --chown=csmuser:csmuser api.py generate_cli.py start.sh requirements.txt ./
COPY --chown=csmuser:csmuser csm ./csm

RUN python -m pip install --upgrade pip setuptools wheel

# Optionally install torch wheel provided as build-arg
RUN if [ -n "$TORCH_WHEEL_URL" ]; then \
      python -m pip install "$TORCH_WHEEL_URL"; \
    else \
      echo "No TORCH_WHEEL_URL provided at build time. Ensure correct torch is installed at runtime."; \
    fi

RUN if [ -f requirements.txt ]; then pip install -r requirements.txt ; fi
RUN if [ -f csm/requirements.txt ]; then pip install -r csm/requirements.txt ; fi

ENV PORT=8000
EXPOSE 8000

ENV MODEL_DIR=/workspace/csm-1b
ENV HF_REPO_ID=sesame/csm-1b

ENTRYPOINT ["./start.sh"]
