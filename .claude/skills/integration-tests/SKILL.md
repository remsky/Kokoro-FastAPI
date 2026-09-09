---
name: integration-tests
description: "Running the full e2e integration test suite (including Whisper roundtrip) against a live Kokoro server."
---

Read [AGENTS.md](../../../AGENTS.md) and [CONTRIBUTING.md](../../../CONTRIBUTING.md) first, house rules live there.

# Integration tests

## Prerequisites

A running Kokoro server (GPU or CPU container) on port 8880.

## Running

Unit tests (no server needed):

    uv run pytest

Integration tests use the prebuilt test-client image with Whisper baked in. Mount the test directory and point at the running server:

    docker run --rm --network host       -v "$PWD/api/tests/integration:/tests/integration:ro"       -e KOKORO_BASE_URL=http://localhost:8880       -e WHISPER_MODEL=/opt/whisper/small       ghcr.io/remsky/tts-api-test-client:latest

`--network host` is Linux only. On Docker Desktop drop it, use `-e KOKORO_BASE_URL=http://host.docker.internal:8880`, and mount with an absolute `C:/...` path. Restart a hot-mounted server container first, the Python process only loads code at start.

Or the full compose stack (builds a CPU server + test-client, self-contained):

    docker compose -f docker/docker-compose.test.yml up --build ^
      --abort-on-container-exit --exit-code-from test-client

## What runs

- `test_rate_durations.py` - speed/rate tag scaling against real audio durations
- `test_tts_roundtrip.py` - synth + Whisper transcribe, 10 voice/language cases across 8 languages, WER/CER thresholds, plus an unpunctuated long run
- `test_voices_endpoint.py` - voice listing shape, legacy compat, nova mapping

19 tests total. The roundtrip tests need ~30s (Whisper inference is the bottleneck).
