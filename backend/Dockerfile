# Backend container for the Semantic Geometry Explorer ML API.
# Targets a CPU host (Hugging Face Spaces "Docker" SDK, Render, Fly.io, etc.).

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    SGE_MODEL=pythia-70m \
    PORT=7860

WORKDIR /app

# Build tools needed to compile ripser (C++) and friends.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch first so we don't drag in the multi-GB CUDA build,
# then the rest of the requirements (torch is already satisfied).
RUN pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts

# Pre-cache the default model into the image so cold starts skip the download.
RUN python -c "from transformer_lens import HookedTransformer; HookedTransformer.from_pretrained('pythia-70m')"

# HF Spaces injects $PORT; default to 7860 locally.
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
