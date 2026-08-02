$env:PHONEMIZER_ESPEAK_LIBRARY="C:\Program Files\eSpeak NG\libespeak-ng.dll"
$env:PYTHONUTF8=1
$Env:PROJECT_ROOT="$pwd"
$Env:USE_GPU="false"
$Env:USE_ONNX="false"
$Env:PYTHONPATH="$Env:PROJECT_ROOT;$Env:PROJECT_ROOT/api"
$Env:MODEL_DIR="src/models"
if (-not $Env:MODEL_VERSION) { $Env:MODEL_VERSION = "v1_0" }
$Env:VOICES_DIR="src/voices/$Env:MODEL_VERSION"
$Env:WEB_PLAYER_PATH="$Env:PROJECT_ROOT/web"

uv pip install -e ".[cpu]"
uv run --no-sync python docker/scripts/download_model.py --version $Env:MODEL_VERSION --output "api/src/models/$Env:MODEL_VERSION" --voices-output "api/src/voices/$Env:MODEL_VERSION"
uv run --no-sync uvicorn api.src.main:app --host 0.0.0.0 --port 8880