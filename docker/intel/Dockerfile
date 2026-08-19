# Stage 1: Builder - Use devel image for compilation
FROM ubuntu:24.04 AS builder

# Install Python and build dependencies
RUN apt-get update -y && \
    apt-get install -y python3.10 python3-venv python3-dev git curl wget gnupg2 && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/ && \
    mv /root/.local/bin/uvx /usr/local/bin/

WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./pyproject.toml

# Install dependencies with intel extras
ENV UV_HTTP_TIMEOUT=120 UV_HTTP_RETRIES=3 \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python

RUN uv venv --python 3.10 && \
    uv sync --extra intel --no-cache --no-install-project

# Stage 2: Runtime
FROM ubuntu:24.04

# Install runtime dependencies + uv
RUN apt-get update -y && \
    apt-get install -y python3.10 espeak-ng espeak-ng-data libsndfile1 ffmpeg curl wget gnupg2 libnuma1 ocl-icd-libopencl1 libze1 intel-media-va-driver-non-free libmfx1 && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/ && \
    mv /root/.local/bin/uvx /usr/local/bin/ && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Add Intel oneAPI repository for MKL and other tools
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update && \
    apt-get install -y intel-oneapi-mkl intel-oneapi-runtime-mkl intel-oneapi-runtime-compilers intel-oneapi-compiler-dpcpp-cpp-runtime-2025.3 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Intel GPU drivers (Compute Runtime & IGC) using the reference script
COPY docker/intel/install_ubuntu_gpu_drivers.sh /tmp/install_gpu_drivers.sh
ARG INSTALL_DRIVER_VERSION="25.35.35096"
RUN chmod +x /tmp/install_gpu_drivers.sh && \
    /tmp/install_gpu_drivers.sh && \
    rm -f /tmp/install_gpu_drivers.sh

# Setup user and GID
ARG RENDER_GID=992
RUN mkdir -p /usr/share/espeak-ng-data && \
    ln -s /usr/lib/*/espeak-ng-data/* /usr/share/espeak-ng-data/ && \
    (groupmod -g ${RENDER_GID} render || groupadd -g ${RENDER_GID} render) && \
    useradd -m -u 1001 -G video,render appuser && \
    mkdir -p /app/api/src/models/v1_0 && \
    chown -R appuser:appuser /app

WORKDIR /app

# Copy uv-managed Python interpreter (the venv's bin/python symlinks into here)
COPY --from=builder /opt/uv-python /opt/uv-python

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Download model in runtime stage so download_model.py is present for runtime re-downloads via entrypoint
COPY --chown=appuser:appuser docker/scripts/download_model.py ./download_model.py
ARG DOWNLOAD_MODEL=true
RUN if [ "$DOWNLOAD_MODEL" = "true" ]; then \
    /app/.venv/bin/python download_model.py --output api/src/models/v1_0 && \
    chown -R appuser:appuser /app/api/src/models; \
    fi

# Japanese support requires the UniDic dictionary
ARG INCLUDE_JAPANESE=true
RUN if [ "$INCLUDE_JAPANESE" = "true" ]; then \
    /app/.venv/bin/python -m unidic download && \
    chown -R appuser:appuser /app/.venv/lib/python*/site-packages/unidic; \
    fi

# Copy project files
COPY --chown=appuser:appuser pyproject.toml ./pyproject.toml
COPY --chown=appuser:appuser api ./api
COPY --chown=appuser:appuser web ./web
COPY --chown=appuser:appuser VERSION ./VERSION
COPY --chown=appuser:appuser docker/intel/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

USER appuser

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/api \
    UV_LINK_MODE=copy \
    USE_GPU=true \
    PHONEMIZER_ESPEAK_PATH=/usr/bin \
    PHONEMIZER_ESPEAK_DATA=/usr/share/espeak-ng-data \
    ESPEAK_DATA_PATH=/usr/share/espeak-ng-data \
    DEVICE="intel" \
    LD_LIBRARY_PATH="/opt/intel/oneapi/mkl/latest/lib:/opt/intel/oneapi/mkl/latest/lib/intel64:/opt/intel/oneapi/compiler/latest/lib/intel64:$LD_LIBRARY_PATH"

# Run FastAPI server
CMD ["./entrypoint.sh"]
