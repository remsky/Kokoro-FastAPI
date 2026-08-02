"""Tests for the checksum guards in docker/scripts/download_model.py."""

import hashlib
import importlib.util
import os
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "docker" / "scripts" / "download_model.py"
)


@pytest.fixture
def dm():
    """Load download_model.py, which lives outside the api package."""
    spec = importlib.util.spec_from_file_location("download_model", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, body: bytes) -> str:
    path.write_bytes(body)
    return hashlib.sha256(body).hexdigest()


def test_verify_files_missing(dm, tmp_path):
    assert dm.verify_files(str(tmp_path), "v1_0") is False


def test_verify_files_matches_checksums(dm, tmp_path, monkeypatch):
    model_sha = _write(tmp_path / "kokoro-v1_0.pth", b"weights")
    config_sha = _write(tmp_path / "config.json", b"{}")
    monkeypatch.setitem(
        dm.MODEL_FILES,
        "v1_0",
        {
            "kokoro-v1_0.pth": ("kokoro-v1_0.pth", model_sha),
            "config.json": ("config.json", config_sha),
        },
    )

    assert dm.verify_files(str(tmp_path), "v1_0") is True

    # a truncated or replaced file no longer verifies
    (tmp_path / "kokoro-v1_0.pth").write_bytes(b"<html>404</html>")
    assert dm.verify_files(str(tmp_path), "v1_0") is False


def test_fetch_verified_rejects_mismatch_and_cleans_up(dm, tmp_path):
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"payload")
    dm.BASE_URL = tmp_path.as_uri()
    dest = tmp_path / "out.bin"

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        dm._fetch_verified("asset.bin", "0" * 64, str(dest))

    assert not dest.exists()
    assert not any(f.endswith(".download") for f in os.listdir(tmp_path))


def test_fetch_verified_temp_path_is_pid_scoped(dm, tmp_path, monkeypatch):
    """Concurrent starts on a shared volume must not share a temp path."""
    seen = []
    monkeypatch.setattr(dm, "urlretrieve", lambda url, tmp: seen.append(tmp))
    monkeypatch.setattr(dm, "_sha256", lambda path: "abc")
    monkeypatch.setattr(dm.os, "replace", lambda src, dst: None)

    dm._fetch_verified("asset.bin", "abc", str(tmp_path / "out.bin"))

    assert seen == [f"{tmp_path / 'out.bin'}.{os.getpid()}.download"]
