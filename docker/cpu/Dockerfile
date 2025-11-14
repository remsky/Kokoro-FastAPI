# ============================================================================
# ZipVoice TTS API - Optimized CPU Dockerfile
# ============================================================================
# Multi-stage build for optimized image size
# Includes: ZipVoice, Whisper, ONNX Runtime CPU
# ============================================================================

# Stage 1: Builder - Install dependencies and compile
FROM python:3.10-slim AS builder

# Install build dependencies including Rust
RUN apt-get update -y && \
    apt-get install -y \
        git \
        curl \
        g++ \
        cmake \
        build-essential \
        libsndfile1-dev \
        ffmpeg \
        espeak-ng \
        espeak-ng-data \
        pkg-config \
        libssl-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Rust (needed for building some Python packages)
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

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
        # Core PyTorch CPU version
        torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu && \
    uv sync --extra cpu --extra zipvoice --no-cache && \
    # Install ZipVoice and k2 (CPU build)
    uv pip install --no-cache \
        zipvoice \
        k2==1.24.4.dev20250208+cpu.torch2.5.1 -f https://k2-fsa.github.io/k2/cpu.html && \
    # Install Whisper for auto-transcription (tiny model for CPU)
    uv pip install --no-cache openai-whisper && \
    # Install ONNX Runtime CPU for optimization
    uv pip install --no-cache onnxruntime && \
    # Clean up
    find .venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find .venv -type f -name "*.pyc" -delete && \
    find .venv -type f -name "*.pyo" -delete

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
FROM python:3.10-slim

# Install only runtime dependencies
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
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
    ln -sf /usr/lib/*/espeak-ng-data/* /usr/share/espeak-ng-data/ 2>/dev/null || true

# Create non-root user
RUN useradd -m -u 1000 zipvoice && \
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
    /app/api/src/voices/zipvoice_prompts \
    /app/api/temp_files \
    /app/api/logs && \
    # Download Whisper tiny model to cache (CPU-efficient)
    python3 -c "import whisper; whisper.load_model('tiny')" 2>/dev/null || true

# Environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/api \
    PYTHONDONTWRITEBYTECODE=1 \
    # Device settings
    USE_GPU=false \
    DEVICE_TYPE=cpu \
    # Espeak
    PHONEMIZER_ESPEAK_PATH=/usr/bin \
    PHONEMIZER_ESPEAK_DATA=/usr/share/espeak-ng-data \
    ESPEAK_DATA_PATH=/usr/share/espeak-ng-data \
    # ZipVoice settings
    ENABLE_KOKORO=false \
    ENABLE_ZIPVOICE=true \
    DEFAULT_BACKEND=zipvoice \
    # Smart features (CPU-optimized)
    ENABLE_AUTO_TRANSCRIPTION=true \
    WHISPER_MODEL_SIZE=tiny \
    ENABLE_SMART_TUNING=true \
    ENABLE_QUALITY_DETECTION=true \
    # Optimizations (ONNX only on CPU)
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
