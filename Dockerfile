FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu126

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install .

COPY src/ src/
