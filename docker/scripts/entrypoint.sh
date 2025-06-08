#!/bin/bash
set -e

exec python -m uvicorn api.src.main:app --host 0.0.0.0 --port 8880 --log-level info