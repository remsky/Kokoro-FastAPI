FROM --platform=$BUILDPLATFORM nvidia/cuda:12.9.0-base-ubuntu24.04
# Set non-interactive frontend
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and other dependencies
RUN apt-get update -y && \
    apt-get install -y python3 python3-venv libsndfile1 curl ffmpeg g++ && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/ && \
    mv /root/.local/bin/uvx /usr/local/bin/ && \
    useradd -m -u 1001 appuser && \
    mkdir -p /app/api/src/models/v1_0 && \
    chown -R appuser:appuser /app
    
USER appuser
WORKDIR /app

ENV PHONEMIZER_ESPEAK_PATH=/usr/bin \
    PHONEMIZER_ESPEAK_DATA=/usr/share/espeak-ng-data \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/api \
    PATH="/app/.venv/bin:$PATH" \
    UV_LINK_MODE=copy \
    USE_GPU=true \
    DEVICE="gpu"

COPY --chown=appuser:appuser pyproject.toml ./pyproject.toml
COPY --chown=appuser:appuser api ./api
COPY --chown=appuser:appuser web ./web
COPY --chown=appuser:appuser docker/scripts/ ./
RUN chmod +x ./entrypoint.sh
    
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python 3.10 && \
    uv sync --extra gpu && \
    uv cache clean && \
    python download_model.py --output api/src/models/v1_0

# Run FastAPI server through entrypoint.sh
CMD ["./entrypoint.sh"]



# If you want to test the docker image locally, run this from the project root:
#   docker build -f docker\gpu\Dockerfile -t kokoro .
# Run it with
#   docker run -p 8880:8880 --name kokoro kokoro --gpus all
#
# You can log into the container with
#   docker exec -it kokoro /bin/bash
#
# Other commands:
# 1. Stop and remove container
# docker stop kokoro
# docker container remove kokoro
# 2. List and remove images
# docker images
# docker image remove kokoro