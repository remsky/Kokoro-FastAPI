#!/bin/bash
set -e

exec uv run --extra $DEVICE --no-sync python -m uvicorn api.src.main:app --host 0.0.0.0 --port 8880 --log-level $UVICORN_LOG_LEVEL --workers $NUM_WORKERS