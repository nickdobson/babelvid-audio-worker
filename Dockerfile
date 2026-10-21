# syntax=docker/dockerfile:1.7
# RunPod Serverless queue worker: dialog/background separation (BS-RoFormer via audio-separator)
# and SenseVoice delivery analysis. Build context: workers/audio
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv ffmpeg libsndfile1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /usr/local/bin/

COPY requirements.txt /app/requirements.txt
RUN uv venv /opt/venv --python /usr/bin/python3 \
    && VIRTUAL_ENV=/opt/venv uv pip install --torch-backend=cu128 torch torchaudio -r /app/requirements.txt \
    && rm -rf /root/.cache

ENV PATH="/opt/venv/bin:${PATH}" MODELS_DIR=/models

# Bake the models in: ~1.5 GB, saves a download on every cold start.
COPY download_models.py /app/download_models.py
RUN python /app/download_models.py && rm -rf /root/.cache

COPY handler.py /app/handler.py
WORKDIR /app
CMD ["python", "-u", "/app/handler.py"]
