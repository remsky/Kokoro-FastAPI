"""A live server on a free port, so the real client libraries connect over a socket."""

import threading
import time
from unittest.mock import AsyncMock, patch

import pytest
import uvicorn

from api.src.main import app
from api.tests.test_websocket_tts import FakeService


@pytest.fixture
def service():
    fake = FakeService()
    voices = AsyncMock(side_effect=lambda voice, _service: voice)
    with (
        patch(
            "api.src.routers.openai_realtime.get_tts_service",
            AsyncMock(return_value=fake),
        ),
        patch("api.src.routers.openai_realtime.process_and_validate_voices", voices),
    ):
        yield fake


@pytest.fixture(scope="module")
def server_url():
    config = uvicorn.Config(
        app, port=0, ws="wsproto", lifespan="off", log_level="warning"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("Test server did not start")
        time.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}/v1"
    server.should_exit = True
    thread.join(5)
