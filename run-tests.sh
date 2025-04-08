#!/bin/bash

# Get project root directory
PROJECT_ROOT=$(pwd)

# Set environment variables
export USE_GPU=false
export USE_ONNX=false
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/api
export MODEL_DIR=src/models
export VOICES_DIR=src/voices/v1_0
export WEB_PLAYER_PATH=$PROJECT_ROOT/web
# Set the espeak-ng data path to your location
export ESPEAK_DATA_PATH=/usr/lib/x86_64-linux-gnu/espeak-ng-data

# Run FastAPI with CPU extras using uv run
# Note: espeak may still require manual installation,
uv pip install -e ".[test,cpu]"
uv run --no-sync python docker/scripts/download_model.py --output api/src/models/v1_0

uv run pytest api/tests/ --asyncio-mode=auto --cov=api --cov-report=term-missing
