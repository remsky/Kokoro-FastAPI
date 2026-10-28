"""API_KEYS gates the API routes behind a bearer token and leaves health and the web player open."""

import pytest
from fastapi.testclient import TestClient

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


@pytest.mark.parametrize("key", ["key-one", "key-two"])
def test_any_configured_key_is_accepted(keys, key):
    response = client.get("/v1/models", headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 200


def test_health_and_web_player_stay_open(keys):
    assert client.get("/health").status_code == 200
    assert client.get("/web/").status_code != 401
