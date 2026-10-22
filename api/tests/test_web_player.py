"""Tests for the /web static file routes."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.src.core.config import settings
from api.src.inference import inno_tuner
from api.src.main import app

client = TestClient(app)


def test_web_config_reports_root_path_and_version():
    with patch.dict("os.environ", {"UVICORN_ROOT_PATH": "/tts"}):
        response = client.get("/web/config")

    assert response.status_code == 200
    assert response.json() == {
        "root_path": "/tts",
        "version": settings.api_version,
        "tuner": inno_tuner.available(),
        "voice_saving": settings.allow_local_voice_saving,
    }


def test_web_root_serves_index():
    response = client.get("/web/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-cache"
    assert b"<html" in response.content


def test_web_missing_file_is_404():
    assert client.get("/web/nope.js").status_code == 404


def test_web_read_error_is_500():
    with patch("api.src.routers.web_player.read_bytes", side_effect=OSError("boom")):
        assert client.get("/web/index.html").status_code == 500


@pytest.mark.parametrize("path", ["/web/config", "/web/index.html"])
def test_web_disabled_is_404(path):
    with patch.object(settings, "enable_web_player", False):
        response = client.get(path)

    assert response.status_code == 404
    assert response.json()["detail"] == "Web player is disabled"
