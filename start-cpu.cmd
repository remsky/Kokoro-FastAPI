@echo off
set PHONEMIZER_ESPEAK_LIBRARY=C:\Program Files\eSpeak NG\libespeak-ng.dll
set PYTHONUTF8=1
set PROJECT_ROOT=%cd%
set USE_GPU=false
set USE_ONNX=false
set PYTHONPATH=%PROJECT_ROOT%;%PROJECT_ROOT%/api
set MODEL_DIR=src/models
set VOICES_DIR=src/voices/v1_0
set WEB_PLAYER_PATH=%PROJECT_ROOT%/web

call setup_visual_studio_env.bat

uv pip install -e ".[cpu]"
uv run --no-sync python docker/scripts/download_model.py --output api/src/models/v1_0
uv run --no-sync uvicorn api.src.main:app --host 0.0.0.0 --port 8880