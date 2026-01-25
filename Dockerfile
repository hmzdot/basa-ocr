FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu126

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        build-essential \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install .

ENV PATH="/opt/venv/bin:${PATH}"

COPY src/ src/
