$env:PHONEMIZER_ESPEAK_LIBRARY = "C:\Program Files\eSpeak NG\libespeak-ng.dll"
$env:PYTHONUTF8 = "1"
$env:PROJECT_ROOT = $PWD
$env:USE_GPU = "true"
$env:USE_ONNX = "false"
$env:PYTHONPATH = "$env:PROJECT_ROOT;$env:PROJECT_ROOT/api"
$env:MODEL_DIR = "src/models"
$env:VOICES_DIR = "src/voices/v1_0"
$env:WEB_PLAYER_PATH = "$env:PROJECT_ROOT/web"
$env:UVICORN_LOG_LEVEL = "warning"
$env:LOGURU_LEVEL = "WARNING"

& ./setup_visual_studio_env.bat

uv pip install -e ".[gpu]"
uv run --no-sync python docker/scripts/download_model.py --output api/src/models/v1_0
uv run --no-sync uvicorn api.src.main:app --host 0.0.0.0 --port 8880 --log-level $env:UVICORN_LOG_LEVEL --workers 2