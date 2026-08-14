---
name: api-contrib
description: "Contributing to the Kokoro-FastAPI Python API: module layout, endpoint gating pattern, test expectations. Use when adding or changing endpoints, services, or inference code."
---

# API contributions

## Where things live

- `api/src/routers/openai_compatible.py` - main API surface (`/v1/audio/speech`, captioned speech). Changes here affect every OpenAI-client consumer, so keep request/response shapes backward compatible.
- `api/src/routers/development.py` and `debug.py` - dev/introspection endpoints, opt-in only.
- `api/src/routers/ssml.py` - `/dev/ssml`, behind `enable_ssml`. `web_player.py` serves `/web`.
- `api/src/services/tts_service.py` - orchestration between text processing, inference, and audio encoding.
- `api/src/services/streaming_audio_writer.py` - per-format encode/finalize. Historically fragile at chunk boundaries and finalize time (WAV trailer click, OGG final-page loss), so test full stream + finalize output, not just single chunks.
- `api/src/inference/` - `kokoro_v1.py` (backend), `model_manager.py`, `voice_manager.py`. Voice tensors are cached and loaded with `weights_only=True`; keep both properties intact.
- `api/src/core/config.py` - pydantic-settings, all env-overridable.

## Adding an endpoint

1. Add the route in the right router (or a new one, registered in `api/src/main.py`).
2. Two gating patterns, both 403 when off and both documented in the README config table:
   - Surfaces host, process, or model internals (unload, introspection, etc): `False`-default setting, opt-in, to avoid unintentional exposure on shared deployments. Follow `enable_debug_endpoints` / `allow_dev_unload`.
   - Parses richer user text (tags, markup): `True`-default kill switch so operators proxying untrusted input can turn it off. Follow `enable_voice_tags` / `enable_ssml`.
3. Add `api/tests/test_<feature>.py` covering enabled, disabled, and error paths.

## Testing

- `uv run pytest` runs `api/tests/` with coverage; integration tests are excluded (`-m integration` to opt in, needs a running server).
- Tests mock the model; no GPU needed. If a change is only verifiable on CUDA/ROCm hardware you don't have, say so in the PR.
- `ruff format .` and `ruff check . --fix` before staging.
