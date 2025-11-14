# ============================================================================
# ZipVoice TTS API - Optimized GPU Dockerfile
# ============================================================================
# Multi-stage build for optimized image size
# Includes: ZipVoice, Whisper, ONNX Runtime GPU, TensorRT support
# ============================================================================

# Stage 1: Builder - Install dependencies and compile
FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

# Install build dependencies
RUN apt-get update -y && \
    apt-get install -y \
        python3.10 \
        python3.10-dev \
        python3-pip \
        python3-venv \
        git \
        curl \
        g++ \
        cmake \
        build-essential \
        libsndfile1-dev \
        ffmpeg \
        espeak-ng \
        espeak-ng-data && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv for faster package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/ && \
    mv /root/.local/bin/uvx /usr/local/bin/

WORKDIR /build

# Copy dependency files
COPY pyproject.toml ./

# Create virtual environment and install dependencies
RUN uv venv --python 3.10 && \
    . .venv/bin/activate && \
    uv pip install --no-cache \
        # Core PyTorch with CUDA
        torch==2.8.0+cu129 --index-url https://download.pytorch.org/whl/cu129 && \
    uv sync --extra gpu --extra zipvoice --no-cache && \
    # Install ZipVoice and k2
    uv pip install --no-cache \
        zipvoice \
        k2==1.24.4.dev20250208+cuda12.1.torch2.5.1 -f https://k2-fsa.github.io/k2/cuda.html && \
    # Install Whisper for auto-transcription
    uv pip install --no-cache openai-whisper && \
    # Install ONNX Runtime GPU for optimization
    uv pip install --no-cache onnxruntime-gpu && \
    # Clean up
    find .venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find .venv -type f -name "*.pyc" -delete && \
    find .venv -type f -name "*.pyo" -delete

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Install only runtime dependencies
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        libsndfile1 \
        ffmpeg \
        espeak-ng \
        espeak-ng-data \
        curl \
        ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Setup espeak symlinks
    mkdir -p /usr/share/espeak-ng-data && \
    ln -sf /usr/lib/x86_64-linux-gnu/espeak-ng-data/* /usr/share/espeak-ng-data/ 2>/dev/null || \
    ln -sf /usr/lib/aarch64-linux-gnu/espeak-ng-data/* /usr/share/espeak-ng-data/ 2>/dev/null || true

# Create non-root user
RUN useradd -m -u 1001 -s /bin/bash zipvoice && \
    mkdir -p /app && \
    chown -R zipvoice:zipvoice /app

USER zipvoice
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=zipvoice:zipvoice /build/.venv /app/.venv

# Copy application code
COPY --chown=zipvoice:zipvoice api ./api
COPY --chown=zipvoice:zipvoice examples ./examples
COPY --chown=zipvoice:zipvoice docs ./docs

# Copy scripts
COPY --chown=zipvoice:zipvoice docker/scripts/entrypoint-zipvoice.sh ./entrypoint.sh
COPY --chown=zipvoice:zipvoice docker/scripts/healthcheck.sh ./healthcheck.sh
RUN chmod +x ./entrypoint.sh ./healthcheck.sh

# Create necessary directories with proper permissions
RUN mkdir -p \
    /app/api/src/models/onnx_cache \
    /app/api/src/models/tensorrt_cache \
    /app/api/src/voices/zipvoice_prompts \
    /app/api/temp_files \
    /app/api/logs && \
    # Download Whisper base model to cache
    python3 -c "import whisper; whisper.load_model('base')" 2>/dev/null || true

# Environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/api \
    PYTHONDONTWRITEBYTECODE=1 \
    # Device settings
    USE_GPU=true \
    DEVICE_TYPE=cuda \
    # Espeak
    PHONEMIZER_ESPEAK_PATH=/usr/bin \
    PHONEMIZER_ESPEAK_DATA=/usr/share/espeak-ng-data \
    ESPEAK_DATA_PATH=/usr/share/espeak-ng-data \
    # ZipVoice settings
    ENABLE_KOKORO=false \
    ENABLE_ZIPVOICE=true \
    DEFAULT_BACKEND=zipvoice \
    # Smart features
    ENABLE_AUTO_TRANSCRIPTION=true \
    WHISPER_MODEL_SIZE=base \
    ENABLE_SMART_TUNING=true \
    ENABLE_QUALITY_DETECTION=true \
    # Optimizations (disabled by default, enable via docker-compose)
    ENABLE_ONNX=false \
    ENABLE_TENSORRT=false \
    # API settings
    API_LOG_LEVEL=INFO \
    HOST=0.0.0.0 \
    PORT=8880

# Expose port
EXPOSE 8880

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["./healthcheck.sh"]

# Run application
CMD ["./entrypoint.sh"]
