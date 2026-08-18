#!/bin/bash
set -e

MODEL_VERSION=${MODEL_VERSION:-v1_0}

if [ "$DOWNLOAD_MODEL" = "true" ]; then
    python download_model.py --version "$MODEL_VERSION" --output "api/src/models/$MODEL_VERSION" --voices-output "api/src/voices/$MODEL_VERSION"
fi

exec python -m uvicorn api.src.main:app --host 0.0.0.0 --port 8880 --log-level debug
