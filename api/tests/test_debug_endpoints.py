"""Tests for the /debug/* opt-in gate."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from api.src.core.config import settings
from api.src.main import app

client = TestClient(app)


def test_debug_endpoints_403_when_disabled():
    """Disabled by default: every /debug/* route returns 403."""
    for path in ["/debug/threads", "/debug/storage", "/debug/system"]:
        response = client.get(path)
        assert response.status_code == 403
        assert response.json()["detail"]["error"] == "Debug endpoints are disabled"


def test_debug_endpoints_200_when_enabled():
    with patch.object(settings, "enable_debug_endpoints", True):
        response = client.get("/debug/threads")

    assert response.status_code == 200
    assert "total_threads" in response.json()
