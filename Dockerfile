FROM nvcr.io/nvidia/pytorch:25.10-py3

WORKDIR /workspace

RUN pip install uv

COPY pyproject.toml uv.lock ./

# Exclude torch packages as they are already installed
RUN uv sync --locked \
    --exclude torch \
    --exclude torchvision \
    --exclude torchaudio

COPY src/ src/


