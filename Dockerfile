# Dockerfile - example (may need adaptation to RunPod's runtime)
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y build-essential libsndfile1 git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

CMD ["python", "handler.py"]
