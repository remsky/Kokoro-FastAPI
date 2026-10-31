"""API_KEYS gates the API routes behind a bearer token and leaves health and the web player open."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketDenialResponse

from api.src.core.config import settings
from api.src.main import app

client = TestClient(app)


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setattr(settings, "api_keys", ["key-one", "key-two"])


def test_no_keys_configured_leaves_api_open():
    assert client.get("/v1/models").status_code == 200


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic key-one"}],
)
def test_missing_or_wrong_key_is_401(keys, headers):
    for path in ["/v1/models", "/dev/ssml"]:
        response = client.get(path, headers=headers)
        assert response.status_code == 401, path
        assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize(
    "headers",
    [
        {"Authorization": "Bearer key-one"},
        {"Authorization": "bearer key-two"},
        {"xi-api-key": "key-one"},
        {"X-API-Key": "key-two"},
    ],
)
def test_any_configured_key_in_any_client_header_is_accepted(keys, headers):
    assert client.get("/v1/models", headers=headers).status_code == 200


def test_empty_configured_key_never_matches_a_missing_header(monkeypatch):
    monkeypatch.setattr(settings, "api_keys", ["", "key-one"])
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer "}).status_code == 401
    assert client.get("/v1/models", headers={"xi-api-key": "key-one"}).status_code == 200


WS_URLS = ["/v1/text-to-speech/af_heart/multi-stream-input", "/tts/websocket"]


@pytest.mark.parametrize("url", WS_URLS)
def test_websocket_without_key_is_refused(keys, url):
    with pytest.raises(WebSocketDenialResponse) as refused:
        with client.websocket_connect(url, headers={"Authorization": "Bearer wrong"}):
            pass
    assert refused.value.status_code == 401
    assert refused.value.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize(
    "url, headers",
    [(WS_URLS[0], {"xi-api-key": "key-one"}), (WS_URLS[1], {"X-API-Key": "key-two"})],
)
def test_websocket_with_key_connects(keys, url, headers):
    with (
        patch("api.src.routers.elevenlabs_compatible.get_tts_service", AsyncMock()),
        patch("api.src.routers.cartesia_compatible.get_tts_service", AsyncMock()),
    ):
        with client.websocket_connect(url, headers=headers):
            pass


def test_health_and_web_player_stay_open(keys):
    assert client.get("/health").status_code == 200
    assert client.get("/v1/test").status_code == 200
    assert client.get("/web/").status_code != 401
