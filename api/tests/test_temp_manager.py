import os

import pytest

from api.src.core.config import settings
from api.src.services.temp_manager import cleanup_temp_files


@pytest.mark.asyncio
async def test_cleanup_removes_timing_sidecar_with_its_audio(tmp_path, monkeypatch):
    """An expired audio file takes its timing sidecar with it, and fresh pairs stay."""
    monkeypatch.setattr(settings, "temp_file_dir", str(tmp_path))
    monkeypatch.setattr(settings, "max_temp_dir_age_hours", 1)
    monkeypatch.setattr(settings, "max_temp_dir_count", 100)
    monkeypatch.setattr(settings, "max_temp_dir_size_mb", 1000)

    old_audio = tmp_path / "tmpold.mp3"
    old_sidecar = tmp_path / "tmpold.mp3.json"
    fresh_audio = tmp_path / "tmpnew.mp3"
    fresh_sidecar = tmp_path / "tmpnew.mp3.json"
    for file in (old_audio, old_sidecar, fresh_audio, fresh_sidecar):
        file.write_bytes(b"x")

    now = os.stat(tmp_path).st_mtime
    stale = now - 2 * 3600
    os.utime(old_audio, (stale, stale))
    os.utime(old_sidecar, (now, now))

    await cleanup_temp_files()

    assert not old_audio.exists()
    assert not old_sidecar.exists()
    assert fresh_audio.exists()
    assert fresh_sidecar.exists()
