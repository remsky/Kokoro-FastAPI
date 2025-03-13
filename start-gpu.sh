#!/bin/bash

# Get project root directory
PROJECT_ROOT=$(pwd)

# Set environment variables
export USE_GPU=true
export USE_ONNX=false
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/api
export MODEL_DIR=src/models
export VOICES_DIR=src/voices/v1_0
export WEB_PLAYER_PATH=$PROJECT_ROOT/web

# Set about the Chinese environment variable
# export DEFAULT_VOICE=zf_xiaobei
# export REPO_ID=hexgrad/Kokoro-82M-v1.1-zh
# export KOKORO_V1_FILE=v1_1-zh/kokoro-v1_1-zh.pth

# Run FastAPI with GPU extras using uv run
uv pip install -e ".[gpu]"
uv run --no-sync uvicorn api.src.main:app --host 0.0.0.0 --port 8880
