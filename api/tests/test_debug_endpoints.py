"""Tests for the /debug/* opt-in gate."""

from types import SimpleNamespace
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


def test_debug_storage_reports_gigabytes():
    partition = SimpleNamespace(device="/dev/sda1", mountpoint="/", fstype="ext4")
    usage = SimpleNamespace(
        total=8 * 1024**3, used=2 * 1024**3, free=6 * 1024**3, percent=25.0
    )

    with (
        patch.object(settings, "enable_debug_endpoints", True),
        patch("api.src.routers.debug.psutil.disk_partitions", return_value=[partition]),
        patch("api.src.routers.debug.psutil.disk_usage", return_value=usage),
    ):
        response = client.get("/debug/storage")

    assert response.status_code == 200
    assert response.json()["storage_info"] == [
        {
            "device": "/dev/sda1",
            "mountpoint": "/",
            "fstype": "ext4",
            "total_gb": 8.0,
            "used_gb": 2.0,
            "free_gb": 6.0,
            "percent_used": 25.0,
        }
    ]


def test_debug_storage_skips_unreadable_partitions():
    readable = SimpleNamespace(device="/dev/sda1", mountpoint="/", fstype="ext4")
    locked = SimpleNamespace(
        device="/dev/sdb1", mountpoint="/mnt/locked", fstype="ext4"
    )
    usage = SimpleNamespace(
        total=8 * 1024**3, used=2 * 1024**3, free=6 * 1024**3, percent=25.0
    )

    with (
        patch.object(settings, "enable_debug_endpoints", True),
        patch(
            "api.src.routers.debug.psutil.disk_partitions",
            return_value=[locked, readable],
        ),
        patch(
            "api.src.routers.debug.psutil.disk_usage",
            side_effect=[PermissionError, usage],
        ),
    ):
        response = client.get("/debug/storage")

    assert response.status_code == 200
    assert [e["device"] for e in response.json()["storage_info"]] == ["/dev/sda1"]


def test_debug_system_reports_cpu_memory_process():
    with (
        patch.object(settings, "enable_debug_endpoints", True),
        patch("api.src.routers.debug.psutil.cpu_percent", return_value=1.0),
    ):
        response = client.get("/debug/system")

    assert response.status_code == 200
    body = response.json()
    assert body["cpu"]["cpu_percent"] == 1.0
    assert body["memory"]["virtual"]["total_gb"] > 0
    assert body["process"]["pid"] > 0
    assert "network_io" in body["network"]
