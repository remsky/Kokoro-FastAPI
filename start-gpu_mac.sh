#!/bin/bash

# Get project root directory
PROJECT_ROOT=$(pwd)

# Create mps-specific venv directory
VENV_DIR="$PROJECT_ROOT/.venv-mps"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating MPS-specific virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Set other environment variables
export USE_GPU=true
export USE_ONNX=false
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/api
export MODEL_DIR=src/models
export VOICES_DIR=src/voices/v1_0
export WEB_PLAYER_PATH=$PROJECT_ROOT/web

# Set environment variables
export USE_GPU=true
export USE_ONNX=false
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/api
export MODEL_DIR=src/models
export VOICES_DIR=src/voices/v1_0
export WEB_PLAYER_PATH=$PROJECT_ROOT/web

export DEVICE_TYPE=mps
# Enable MPS fallback for unsupported operations
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Run FastAPI with GPU extras using uv run
uv pip install -e .
uv run --no-sync uvicorn api.src.main:app --host 0.0.0.0 --port 8880
