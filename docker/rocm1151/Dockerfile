FROM --platform=$BUILDPLATFORM python:3.12-slim

# Install Python and other dependencies
RUN apt-get update -y &&  \
    apt-get install -y --no-install-recommends espeak-ng espeak-ng-data git libsndfile1 curl ffmpeg g++ cmake \
    make wget gnupg2 ca-certificates libnuma1 libstdc++6 build-essential pigz &&  \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    mkdir -p /usr/share/espeak-ng-data &&  \
    ln -s /usr/lib/*/espeak-ng-data/* /usr/share/espeak-ng-data/ && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/ && \
    mv /root/.local/bin/uvx /usr/local/bin/ && \
    useradd -m -u 1001 appuser && \
    mkdir -p /app/api/src/models/v1_0 && \
    chown -R appuser:appuser /app

USER appuser
WORKDIR /app
RUN wget --progress=dot:giga https://repo.amd.com/rocm/tarball/therock-dist-linux-gfx1151-7.10.0.tar.gz && \
    mkdir rocm_install && \
    tar -I pigz -xf *.tar.gz -C rocm_install && \
    rm *.tar.gz

ENV ROCM_PATH=/app/rocm_install
ENV PATH=$PATH:$ROCM_PATH/bin
ENV LD_LIBRARY_PATH=$ROCM_PATH/lib

# Copy dependency files
COPY --chown=appuser:appuser pyproject.toml ./pyproject.toml

# Install dependencies with ROCM 1151 extras
RUN uv venv --python 3.12 && \
    uv sync --extra rocm1151 --no-cache

# Copy project files including models
COPY --chown=appuser:appuser api ./api
COPY --chown=appuser:appuser web ./web
COPY --chown=appuser:appuser docker/scripts/ ./
RUN chmod +x ./entrypoint.sh


# Set all environment variables in one go
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/api \
    UV_LINK_MODE=copy \
    USE_GPU=true \
    PHONEMIZER_ESPEAK_PATH=/usr/bin \
    PHONEMIZER_ESPEAK_DATA=/usr/share/espeak-ng-data \
    ESPEAK_DATA_PATH=/usr/share/espeak-ng-data \
    DEVICE="rocm1151"

ENV DOWNLOAD_MODEL=true
# Download model if enabled
RUN if [ "$DOWNLOAD_MODEL" = "true" ]; then \
    python download_model.py --output api/src/models/v1_0; \
    fi

# Run FastAPI server through entrypoint.sh
CMD ["./entrypoint.sh"]
