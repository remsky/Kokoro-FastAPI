#!/usr/bin/env bash

# Get project root directory
PROJECT_ROOT=$(pwd)

# Set environment variables
export USE_GPU=true
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/api
export MODEL_DIR=src/models
export MODEL_VERSION=${MODEL_VERSION:-v1_0}
export VOICES_DIR=src/voices/$MODEL_VERSION
export WEB_PLAYER_PATH=$PROJECT_ROOT/web

# Run FastAPI with GPU extras using uv run
# Note: espeak may still require manual installation,
uv pip install -e ".[gpu]"
uv run --no-sync python docker/scripts/download_model.py --version $MODEL_VERSION --output api/src/models/$MODEL_VERSION --voices-output api/src/voices/$MODEL_VERSION
uv run --no-sync uvicorn api.src.main:app --host 0.0.0.0 --port 8880
