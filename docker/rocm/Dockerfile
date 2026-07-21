# syntax=docker/dockerfile:1

FROM rocm/dev-ubuntu-24.04:6.4.4-complete
ENV DEBIAN_FRONTEND=noninteractive \
    PHONEMIZER_ESPEAK_PATH=/usr/bin \
    PHONEMIZER_ESPEAK_DATA=/usr/share/espeak-ng-data \
    ESPEAK_DATA_PATH=/usr/share/espeak-ng-data

# Install Python and other dependencies
RUN apt-get update && apt upgrade -y && apt-get install -y --no-install-recommends \
    espeak-ng \
    espeak-ng-data \
    rocrand \
    git \
    libsndfile1 \
    curl \
    ffmpeg \
    wget \
    nano \
    g++ \
    zstd \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/share/espeak-ng-data \
    && ln -s /usr/lib/*/espeak-ng-data/* /usr/share/espeak-ng-data/ \

    # Install UV using the installer script
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/ \
    && mv /root/.local/bin/uvx /usr/local/bin/ \

    # Create non-root user and set up directories and permissions
    && useradd -m -u 1001 appuser \
    && mkdir -p /app/api/src/models/v1_0 \
    && chown -R appuser:appuser /app \
    # Models folder
    && mkdir -p /app/api/src/models/v1_0

USER appuser
WORKDIR /app

# Copy dependency files
COPY --chown=appuser:appuser pyproject.toml ./pyproject.toml

ENV PHONEMIZER_ESPEAK_PATH=/usr/bin \
    PHONEMIZER_ESPEAK_DATA=/usr/share/espeak-ng-data \
    ESPEAK_DATA_PATH=/usr/share/espeak-ng-data

# Install dependencies with GPU extras (using cache mounts)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python 3.12 && \
    uv sync --extra rocm

# Japanese support requires the UniDic dictionary (~526MB on disk) for fugashi/MeCab.
# Enabled by default; set --build-arg INCLUDE_JAPANESE=false to skip and shave the image.
# Keep true. Setting false silently breaks ja generation.
ARG INCLUDE_JAPANESE=true
RUN if [ "$INCLUDE_JAPANESE" = "true" ]; then \
        .venv/bin/python -m unidic download; \
    fi

# Run kdb files (shape files for MIOpen)
ENV ROCM_VERSION=6.4.4
COPY --chown=appuser:appuser docker/rocm/kdb_install.sh /tmp/
RUN /tmp/kdb_install.sh

# Support older GFX Arch
ENV ROCBLAS_VERSION=6.4.4-1
RUN cd /tmp && wget https://archive.archlinux.org/packages/r/rocblas/rocblas-${ROCBLAS_VERSION}-x86_64.pkg.tar.zst -O rocblas.tar.zst \
    && pwd && ls -lah ./ \
    && tar --zstd -xvf rocblas.tar.zst && rm rocblas.tar.zst \
    && rm -rf /app/.venv/lib/python3.12/site-packages/torch/lib/rocblas/library/ \
    && mv ./opt/rocm/lib/rocblas/library/ /app/.venv/lib/python3.12/site-packages/torch/lib/rocblas/

# download_model.py re-downloads via entrypoint
COPY --chown=appuser:appuser docker/scripts/download_model.py ./download_model.py
# local model bakes here so the download no-ops;
COPY --chown=appuser:appuser api/src/models ./api/src/models
ARG DOWNLOAD_MODEL=true
RUN if [ "$DOWNLOAD_MODEL" = "true" ]; then \
    .venv/bin/python download_model.py --output api/src/models/v1_0; \
    fi

# model already baked above
COPY --exclude=src/models --chown=appuser:appuser api ./api
COPY --chown=appuser:appuser web ./web
COPY --chown=appuser:appuser VERSION ./VERSION
COPY --chown=appuser:appuser docker/scripts/ ./
COPY --chown=appuser:appuser docker/rocm/warmup_miopen.py /app/docker/rocm/warmup_miopen.py

RUN chmod +x ./entrypoint.sh

# Pre-create MIOpen cache dirs so named-volume mounts inherit appuser
# ownership (Docker would otherwise create the mount target as root:root
# and MIOpen as uid 1001 could not write to it on first run).
RUN mkdir -p /home/appuser/.config/miopen /home/appuser/.cache/miopen

# Set all environment variables in one go.
# MIOPEN_FIND_MODE=2 (FAST): reuse the on-disk find DB when present, fall back
# to immediate-mode heuristics otherwise. Cheap default that avoids the 5-60s
# per-shape kernel search on first hit. See docker/rocm/warmup_miopen.py to
# pre-populate the cache for best-case latency. Override at runtime to redo
# the search (e.g. MIOPEN_FIND_MODE=3 + MIOPEN_FIND_ENFORCE=3).
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/api \
    PATH="/app/.venv/bin:$PATH" \
    UV_LINK_MODE=copy \
    USE_GPU=true \
    DOWNLOAD_MODEL=true \
    DEVICE="gpu" \
    MIOPEN_FIND_MODE=2

# Run FastAPI server through entrypoint.sh
CMD ["./entrypoint.sh"]
