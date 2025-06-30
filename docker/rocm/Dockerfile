FROM rocm/dev-ubuntu-24.04:6.4.1
ENV DEBIAN_FRONTEND=noninteractive \
    PHONEMIZER_ESPEAK_PATH=/usr/bin \
    PHONEMIZER_ESPEAK_DATA=/usr/share/espeak-ng-data \
    ESPEAK_DATA_PATH=/usr/share/espeak-ng-data

# Install Python and other dependencies
RUN apt-get update && apt upgrade -y && apt-get install -y --no-install-recommends \
    espeak-ng \
    espeak-ng-data \
    git \
    libsndfile1 \
    curl \
    ffmpeg \
    wget \
    nano \
    g++ \
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
WORKDIR /app

USER appuser
WORKDIR /app

# Copy dependency files
COPY --chown=appuser:appuser pyproject.toml ./pyproject.toml

ENV PHONEMIZER_ESPEAK_PATH=/usr/bin \
    PHONEMIZER_ESPEAK_DATA=/usr/share/espeak-ng-data \
    ESPEAK_DATA_PATH=/usr/share/espeak-ng-data

# Install dependencies with GPU extras (using cache mounts)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python 3.10 && \
    uv sync --extra rocm

# Copy project files including models
COPY --chown=appuser:appuser api ./api
COPY --chown=appuser:appuser web ./web
COPY --chown=appuser:appuser docker/scripts/ ./
RUN chmod +x ./entrypoint.sh

# Set all environment variables in one go
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/api \
    PATH="/app/.venv/bin:$PATH" \
    UV_LINK_MODE=copy \
    USE_GPU=true \
    DOWNLOAD_MODEL=true \
    DEVICE="gpu"

# Run FastAPI server through entrypoint.sh
CMD ["./entrypoint.sh"]
