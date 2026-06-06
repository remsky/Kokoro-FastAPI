"""Integration test fixtures.

Designed to run inside the `tts-api-test-client` image against a networked
Kokoro server. The fixtures only set up clients; they do NOT manage server
lifecycle.

Required env:
    KOKORO_BASE_URL  Base URL of a running Kokoro server. Inside compose
                     this is `http://server:8880`; for local non-Docker
                     runs, point at whatever you've got listening.
    WHISPER_MODEL    Filesystem path (or model name) for faster-whisper.
                     The test client image pre-bakes weights at
                     /opt/whisper/small and sets this for you.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator

import httpx
import pytest

SERVER_READY_TIMEOUT = float(os.environ.get("KOKORO_SERVER_READY_TIMEOUT", "120"))
HEALTH_POLL_INTERVAL = 1.0


def _wait_for_health(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{url}/health", timeout=2.0)
            if r.status_code == 200:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(HEALTH_POLL_INTERVAL)
    raise RuntimeError(
        f"Server at {url} did not become healthy within {timeout}s "
        f"(last error: {last_error!r})"
    )


@pytest.fixture(scope="session")
def server_url() -> Iterator[str]:
    raw = os.environ.get("KOKORO_BASE_URL")
    if not raw:
        pytest.skip(
            "KOKORO_BASE_URL not set. Run via `docker compose -f "
            "docker-compose.test.yml up` or set the env var to a "
            "running Kokoro server's base URL."
        )
    base = raw.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    _wait_for_health(base, SERVER_READY_TIMEOUT)
    yield base


@pytest.fixture(scope="session")
def openai_client(server_url: str):
    import openai

    # Tight client-side timeout: healthy synth on a CPU runner is 1-3s.
    # Anything past 30s means the server is wedged; fail fast rather than
    # stall the whole sweep on one hung case.
    return openai.OpenAI(
        base_url=f"{server_url}/v1",
        api_key="not-needed",
        timeout=30,
    )


@pytest.fixture(scope="session")
def whisper_model():
    """Load faster-whisper once per session.

    Inside the test-client image, WHISPER_MODEL points at a pre-baked
    directory so this never touches the network. For source-tree runs
    you can set it to a model name (e.g. `small`) and faster-whisper
    will fetch from Hugging Face on first use.

    WHISPER_DEVICE / WHISPER_COMPUTE override the defaults below. Default
    is cpu/int8 inside the image; bumping to cuda/float16 is supported if
    you mount in a GPU and a CUDA-enabled ctranslate2.
    """
    from faster_whisper import WhisperModel

    model = os.environ.get("WHISPER_MODEL", "small")
    device = os.environ.get("WHISPER_DEVICE", "cpu")
    compute = os.environ.get(
        "WHISPER_COMPUTE", "int8" if device == "cpu" else "float16"
    )
    return WhisperModel(model, device=device, compute_type=compute)
