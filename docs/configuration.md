# Configuration

*Last updated: 2026-08-11*

## Docker Compose

Compose files are in `docker/cpu`, `docker/gpu`, `docker/rocm`. Run `docker compose up` from inside one.

### image vs build

`build:` is active, `image:` is commented out:

```yaml
services:
  kokoro-tts:
    # image: ghcr.io/remsky/kokoro-fastapi-gpu:${VERSION:-latest}
    build:
      context: ../..
      dockerfile: docker/gpu/Dockerfile.optimized
```

`build:` compiles from your checkout. Use it if you're changing anything under `api/`.

For the published image, uncomment `image:` and comment out `build:`. `VERSION` defaults to `latest`, or set the full tag (`v` included, same as `docker-bake.hcl`):

```bash
VERSION=v0.8.0 docker compose up
```

That `VERSION` comes from `docker/gpu/.env`, not the repo root (compose reads it next to the compose file). Separate from the app `.env` below.

No checkout needed? Skip compose, use `docker run` (see README).

### Volume mounts shadow the image

```yaml
volumes:
  - ../../api:/app/api
  - ../../web:/app/web
```

Gets you edits without a rebuild. Also means `/app/api/src/models` is your host dir, not the baked one, hence `DOWNLOAD_MODEL=true` in the compose env. `docker run` needs neither, nothing shadows the baked model.

Drop the mounts for the image as published. Keeping them with `image:` runs published deps against local source.

GPU file sets `user: "1001:1001"` to match `appuser`. On Linux, write failures to the mounts are usually host ownership.

## Setting variables

Compose `environment:` block:

```yaml
environment:
  - API_LOG_LEVEL=WARNING
  - DEFAULT_VOICE=bf_emma
```

Or the command line:

```bash
docker run --env API_LOG_LEVEL=WARNING -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:latest
```

A `.env` at the repo root works, picked up by:

```yaml
env_file:
  - path: ../../.env
    required: false
```

`required: false` so a fresh clone still starts without one.

`environment:` beats `env_file`. `PYTHONPATH`, `DOWNLOAD_MODEL`, `API_LOG_LEVEL`, and `USE_GPU` (GPU/ROCm) are pinned there, so setting those in `.env` does nothing. Edit the compose file instead.

Names are the field names from `api/src/core/config.py`, uppercased. Unrecognized keys in `.env` are ignored. Two rows below are process-only: read outside the settings object, so they work as env vars but not from `.env`.

## Reference

**API**

| Variable | Default | |
|---|---|---|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8880` | Bind port |
| `API_TITLE` | `Kokoro TTS API` | OpenAPI title |
| `API_DESCRIPTION` | `API for text-to-speech generation using Kokoro` | OpenAPI description |
| `API_VERSION` | from `VERSION` | OpenAPI version string |
| `API_LOG_LEVEL` | `DEBUG` | loguru level, see Logging below (process-only) |

**Model & device**

| Variable | Default | |
|---|---|---|
| `USE_GPU` | `true` | Use GPU if one is available |
| `DEVICE_TYPE` | auto | Force `cuda`, `mps`, or `cpu` |
| `MODEL_DIR` | `/app/api/src/models` | Where model weights live, container path |
| `VOICES_DIR` | `/app/api/src/voices/v1_0` | Where voice packs live, container path |
| `MODEL_REPO_ID` | `hexgrad/Kokoro-82M` | Fallback download source when `MODEL_DIR` is empty |
| `DOWNLOAD_MODEL` | `true` | Fetch weights at image build and container start (process-only) |

**Voices**

| Variable | Default | |
|---|---|---|
| `DEFAULT_VOICE` | `af_heart` | Voice used when a request omits one |
| `DEFAULT_VOICE_CODE` | unset | Override the language code normally taken from the voice name's first letter |
| `VOICE_WEIGHT_NORMALIZATION` | `true` | Rescale combined voice weights to sum to 1 |
| `ALLOW_LOCAL_VOICE_SAVING` | `false` | Let combined voices be written to disk |
| `ENABLE_VOICE_TAGS` | `true` | Kill switch for `[voice:]` parsing and `/dev/dialogue` |

**Text processing**

| Variable | Default | |
|---|---|---|
| `TARGET_MIN_TOKENS` | `175` | Chunker aims for at least this many tokens |
| `TARGET_MAX_TOKENS` | `250` | Chunker aims for at most this many |
| `ABSOLUTE_MAX_TOKENS` | `450` | Hard ceiling per chunk, model limit is 510 |
| `ENABLE_SSML` | `true` | Kill switch for SSML translation, the `/dev/ssml` routes and `ssml: true` on the speech endpoints will 403 when off |
| `SSML_MAX_DEPTH` | `10` | Deepest SSML nesting translated, past it is a 400 |
| `MAX_PAUSE_DURATION_S` | `60.0` | Ceiling for a single `[pause:Ns]` tag or SSML `<break>`, longer values are clamped |
| `ADVANCED_TEXT_NORMALIZATION` | `true` | Master switch for number/URL/email expansion before phonemizing; English only, opt out per request with `normalization_options` |

**Audio**

| Variable | Default | |
|---|---|---|
| `DEFAULT_VOLUME_MULTIPLIER` | `1.0` | Global gain applied to generated audio |
| `GAP_TRIM_MS` | `1` | Base trim from each streaming chunk end |
| `DYNAMIC_GAP_TRIM_PADDING_MS` | `410` | Padding added back for dynamic gap trim |
| `DYNAMIC_GAP_TRIM_PADDING_CHAR_MULTIPLIER` | `{".": 1, "!": 0.9, "?": 1, ",": 0.8}` | Per-punctuation scaling of that padding, dict-valued so set it in `.env` rather than a shell |

**Web player & CORS**

| Variable | Default | |
|---|---|---|
| `ENABLE_WEB_PLAYER` | `true` | Serve the browser UI |
| `WEB_PLAYER_PATH` | `web` | Static file root for it |
| `CORS_ENABLED` | `true` | Send CORS headers |
| `CORS_ORIGINS` | `["*"]` | Allowed origins, narrow this if the port is reachable beyond localhost |

**Temp files**

| Variable | Default | |
|---|---|---|
| `TEMP_FILE_DIR` | `api/temp_files` | Where `return_download_link` files are written |
| `MAX_TEMP_DIR_SIZE_MB` | `2048` | Prune temp files past this total |
| `MAX_TEMP_DIR_AGE_HOURS` | `1` | Prune temp files older than this |
| `MAX_TEMP_DIR_COUNT` | `3` | Keep at most this many temp files |

**Operational routes**

| Variable | Default | |
|---|---|---|
| `ENABLE_DEBUG_ENDPOINTS` | `false` | Expose `/debug/*` host and process introspection |
| `ALLOW_DEV_UNLOAD` | `false` | Expose `/dev/model`, `POST /dev/unload`, and `POST /dev/reload` |
| `MODEL_AUTO_UNLOAD_TIMEOUT_SECONDS` | `0.0` | Idle seconds before auto-unload; `0` disables auto-unload |
| `MODEL_UNLOAD_STRATEGY` | `destroy` | `destroy` releases model objects; `cpu_cache` keeps model weights in system RAM for faster reload |

## Logging

Global API [loguru logging level](https://loguru.readthedocs.io/en/stable/api/logger.html#levels) can be set using the `API_LOG_LEVEL` environment variable. Defaults to `DEBUG`.

**Docker**

Modify the appropriate compose `yml` or append to command line.
```bash
docker run --env 'API_LOG_LEVEL=WARNING' ...
```

**Direct via UV**

Linux and macOS
```bash
export API_LOG_LEVEL=WARNING
./start-cpu.sh OR
./start-gpu.sh
```

Windows
```powershell
$env:API_LOG_LEVEL = 'WARNING'
.\start-cpu.ps1 OR
.\start-gpu.ps1
```
