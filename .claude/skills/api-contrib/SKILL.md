---
name: api-contrib
description: "Contributing to the Kokoro-FastAPI Python API: module layout, endpoint gating pattern, test expectations. Use when adding or changing endpoints, services, or inference code."
---

# API contributions

## Where things live

- `api/src/routers/openai_compatible.py` - main API surface (`/v1/audio/speech`, captioned speech). Changes here affect every OpenAI-client consumer, so keep request/response shapes backward compatible.
- `api/src/routers/development.py` and `debug.py` - dev/introspection endpoints, opt-in only.
- `api/src/services/tts_service.py` - orchestration between text processing, inference, and audio encoding.
- `api/src/services/streaming_audio_writer.py` - per-format encode/finalize. Historically fragile at chunk boundaries and finalize time (WAV trailer click, OGG final-page loss), so test full stream + finalize output, not just single chunks.
- `api/src/inference/` - `kokoro_v1.py` (backend), `model_manager.py`, `voice_manager.py`. Voice tensors are cached and loaded with `weights_only=True`; keep both properties intact.
- `api/src/core/config.py` - pydantic-settings, all env-overridable.

## Adding an endpoint

1. Add the route in the right router (or a new one, registered in `api/src/main.py`).
2. If it surfaces host, process, or model internals (unload, introspection, etc), gate it behind a `False`-default setting in `config.py` to avoid unintentional exposure on shared deployments. Follow `enable_debug_endpoints` / `allow_dev_unload`: return 403 when disabled, document the env var in README.
3. Add `api/tests/test_<feature>.py` covering enabled, disabled, and error paths.

## Testing

- `uv run pytest` runs `api/tests/` with coverage; integration tests are excluded (`-m integration` to opt in, needs a running server).
- Tests mock the model; no GPU needed. If a change is only verifiable on CUDA/ROCm hardware you don't have, say so in the PR.
- `ruff format .` and `ruff check . --fix` before staging.
