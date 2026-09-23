FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/huggingface \
    TRANSFORMERS_CACHE=/workspace/huggingface \
    TORCH_HOME=/workspace/torch \
    MODEL_ID=BiliSakura/SkySensepp

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

RUN python -m pip install --upgrade pip setuptools wheel

# Install a CUDA-enabled PyTorch build first. SkySense++ is a GPU
# foundation model and must not fall back to CPU execution.
RUN pip install \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY handler.py /app/handler.py

# The model is downloaded on first worker start. Keeping the image free of
# the multi-GB checkpoint makes Cloud Build and image pushes manageable.
ENV PYTHONUNBUFFERED=1

CMD ["python", "/app/handler.py"]
