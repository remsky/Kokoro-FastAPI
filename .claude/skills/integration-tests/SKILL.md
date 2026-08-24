---
name: integration-tests
description: "Running the full e2e integration test suite (including Whisper roundtrip) against a live Kokoro server."
---

# Integration tests

## Prerequisites

A running Kokoro server (GPU or CPU container) on port 8880.

## Running

Unit tests (no server needed):

    uv run pytest

Integration tests use the prebuilt test-client image with Whisper baked in. Mount the test directory and point at the running server:

    docker run --rm --network host ^
      -v "%cd%/api/tests/integration:/tests/integration:ro" ^
      -e KOKORO_BASE_URL=http://localhost:8880 ^
      -e WHISPER_MODEL=/opt/whisper/small ^
      ghcr.io/remsky/tts-api-test-client:latest

Or the full compose stack (builds a CPU server + test-client, self-contained):

    docker compose -f docker/docker-compose.test.yml up --build ^
      --abort-on-container-exit --exit-code-from test-client

## What runs

- `test_rate_durations.py` - speed/rate tag scaling against real audio durations
- `test_tts_roundtrip.py` - synth + Whisper transcribe across 9 languages, WER/CER thresholds
- `test_voices_endpoint.py` - voice listing shape, legacy compat, nova mapping

17 tests total. The roundtrip tests need ~30s (Whisper inference is the bottleneck).
