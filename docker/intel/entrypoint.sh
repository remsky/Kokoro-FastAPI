#!/bin/bash
set -e

# In this simplified version, we assume permissions are handled via Dockerfile/Compose
# and the user is 'appuser'.

if [ "$DOWNLOAD_MODEL" = "true" ]; then
    echo "Downloading model..."
    python download_model.py --output api/src/models/v1_0
fi

echo "Starting Application..."
# We use 'exec' so the process handles signals
exec uv run --extra $DEVICE --no-sync python -m uvicorn api.src.main:app --host 0.0.0.0 --port 8880 --log-level debug
