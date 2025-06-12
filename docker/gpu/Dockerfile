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

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/api \
    PATH="/app/.venv/bin:$PATH" \
    UV_LINK_MODE=copy \
    USE_GPU=true \
    DEVICE="gpu"

COPY --chown=appuser:appuser pyproject.toml ./pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python 3.10 && \
    uv sync --extra gpu && \
    uv cache clean

COPY --chown=appuser:appuser api ./api
COPY --chown=appuser:appuser web ./web
COPY --chown=appuser:appuser docker/scripts/ ./
RUN chmod +x ./entrypoint.sh && \
    python download_model.py --output api/src/models/v1_0

# Run FastAPI server through entrypoint.sh
CMD ["./entrypoint.sh"]



# If you want to test the docker image locally, with GPU support, use Podman.
# See ionstructions fro enabling CUDA on Podman: https://podman-desktop.io/docs/podman/gpu
# Use these commands from the project root:
#   podman build -f docker\gpu\Dockerfile -t evie/kokorotts .
# Run it with:
#   podman run --gpus all -p 8880:8880 --name kokorotts evie/kokorotts
#
# You can log into the container with
#   docker exec -it kokorotts /bin/bash
#
# Other commands:
# 1. Stop and remove container
#    podman container rm kokorotts
#
# 2. List or remove images
#    podman images
#    podman image remove evie/kokorotts
#
# See docs for pushing to ECR
#   https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html
# This is my private repo:
#   https://eu-west-1.console.aws.amazon.com/ecr/private-registry/repositories?region=eu-west-1
# I can get the dodcker push / pull commands from there.
#
# SSH keys for logging into EC2 containers:
#   https://eu-west-1.console.aws.amazon.com/ec2/home?region=eu-west-1#KeyPairs:
#
# docker build -f docker\gpu\Dockerfile -t evie/kokorotts .
# aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin 678811077621.dkr.ecr.eu-west-1.amazonaws.com
# docker tag evie/kokorotts:latest 678811077621.dkr.ecr.eu-west-1.amazonaws.com/evie/kokorotts:latest
# docker push 678811077621.dkr.ecr.eu-west-1.amazonaws.com/evie/kokorotts:latest
# docker run --gpus all -p 8880:8880 --name kokorotts  678811077621.dkr.ecr.eu-west-1.amazonaws.com/evie/kokorotts 
# oe use swarm as explained here: https://blog.container-solutions.com/rolling-updates-with-docker-swarm