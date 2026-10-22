import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from api.src.core.config import settings
from api.src.core.paths import (
    _find_file,
    _scan_directories,
    cleanup_temp_files,
    get_content_type,
    get_model_path,
    get_temp_dir_size,
    get_temp_file_path,
    get_voice_path,
    list_temp_files,
    list_voices,
    load_json,
    load_model_weights,
    load_voice_tensor,
    read_bytes,
    read_file,
    save_voice_tensor,
    verify_model_path,
)


@pytest.mark.asyncio
async def test_find_file_exists():
    """Test finding existing file."""
    with patch("aiofiles.os.path.isfile") as mock_isfile:
        mock_isfile.return_value = True
        path = await _find_file("test.txt", ["/test/path"])
        assert Path(path) == Path(os.path.realpath("/test/path/test.txt"))


@pytest.mark.asyncio
async def test_find_file_not_exists():
    """Test finding non-existent file."""
    with patch("aiofiles.os.path.isfile") as mock_isfile:
        mock_isfile.return_value = False
        with pytest.raises(FileNotFoundError, match="File not found"):
            await _find_file("test.txt", ["/test/path"])


@pytest.mark.parametrize(
    "filename",
    [
        "../secret.txt",
        "../../../../etc/passwd",
        "sub/../../secret.txt",
        os.path.join(os.path.abspath(os.sep), "etc", "passwd"),
    ],
)
@pytest.mark.asyncio
async def test_find_file_rejects_paths_outside_search_root(filename):
    """Paths resolving outside the search root are refused even if they exist."""
    with patch("aiofiles.os.path.isfile") as mock_isfile:
        mock_isfile.return_value = True
        with pytest.raises(FileNotFoundError):
            await _find_file(filename, ["/test/path"])


@pytest.mark.asyncio
async def test_find_file_treats_dot_runs_as_literal_names():
    """`....//` is a directory named '....', not a traversal, so it stays in root."""
    with patch("aiofiles.os.path.isfile") as mock_isfile:
        mock_isfile.return_value = True
        path = await _find_file("....//secret.txt", ["/test/path"])
        assert Path(path) == Path(os.path.realpath("/test/path/..../secret.txt"))


@pytest.mark.asyncio
async def test_find_file_allows_subdirectories():
    """Nested paths inside the search root stay valid (model files live in v1_0/)."""
    with patch("aiofiles.os.path.isfile") as mock_isfile:
        mock_isfile.return_value = True
        path = await _find_file("v1_0/kokoro-v1_0.pth", ["/test/path"])
        assert Path(path) == Path(os.path.realpath("/test/path/v1_0/kokoro-v1_0.pth"))


@pytest.mark.asyncio
async def test_find_file_skips_directories(tmp_path):
    """A name resolving to a directory is not a hit (FileResponse would 500 on it)."""
    (tmp_path / "subdir").mkdir()
    with pytest.raises(FileNotFoundError):
        await _find_file("subdir", [str(tmp_path)])


@pytest.mark.asyncio
async def test_find_file_follows_symlinked_files(tmp_path):
    """A file symlinked into a search root resolves to its target (K8s/NAS voice setups)."""
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "voice.pt"
    target.write_bytes(b"tensor")
    root = tmp_path / "voices"
    root.mkdir()
    try:
        (root / "af_custom.pt").symlink_to(target)
    except OSError:
        pytest.skip("symlink creation not permitted on this host")  # ty: ignore[too-many-positional-arguments]
    path = await _find_file("af_custom.pt", [str(root)])
    assert Path(path) == Path(os.path.realpath(target))


@pytest.mark.asyncio
async def test_find_file_rejects_traversal_through_symlinked_dir(tmp_path):
    """../ collapses lexically before symlinks are followed, so a linked dir can't widen the root."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "secret.txt").write_text("secret")
    root = tmp_path / "voices"
    root.mkdir()
    try:
        (root / "linkdir").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation not permitted on this host")  # ty: ignore[too-many-positional-arguments]
    with pytest.raises(FileNotFoundError):
        await _find_file("linkdir/../../secret.txt", [str(root)])


@pytest.mark.asyncio
async def test_find_file_handles_separator_terminated_root():
    """A root like '/' already ends in os.sep, so the prefix must not double it."""
    with patch("aiofiles.os.path.isfile") as mock_isfile:
        mock_isfile.return_value = True
        path = await _find_file("test.txt", [os.sep])
        assert Path(path) == Path(os.path.realpath(os.sep + "test.txt"))


@pytest.mark.asyncio
async def test_find_file_with_filter():
    """Test finding file with filter function."""
    with patch("aiofiles.os.path.isfile") as mock_isfile:
        mock_isfile.return_value = True
        filter_fn = lambda p: p.endswith(".txt")
        path = await _find_file("test.txt", ["/test/path"], filter_fn)
        assert Path(path) == Path(os.path.realpath("/test/path/test.txt"))


@pytest.mark.asyncio
async def test_scan_directories():
    """Test scanning directories."""
    mock_entry = type("MockEntry", (), {"name": "test.txt"})()

    with (
        patch("aiofiles.os.path.exists") as mock_exists,
        patch("aiofiles.os.scandir") as mock_scandir,
    ):
        mock_exists.return_value = True
        mock_scandir.return_value = [mock_entry]

        files = await _scan_directories(["/test/path"])
        assert "test.txt" in files


@pytest.mark.asyncio
async def test_get_content_type():
    """Test content type detection."""
    test_cases = [
        ("test.html", "text/html"),
        ("test.js", "application/javascript"),
        ("test.css", "text/css"),
        ("test.png", "image/png"),
        ("test.unknown", "application/octet-stream"),
        ("test.mp3", "audio/mpeg"),
        ("test.wav", "audio/wav"),
        ("test.opus", "audio/opus"),
        ("test.flac", "audio/flac"),
        ("test.aac", "audio/aac"),
        ("test.ogg", "audio/ogg"),
        ("test.m4a", "audio/mp4"),
        ("test.pcm", "audio/pcm"),
    ]

    for filename, expected in test_cases:
        content_type = await get_content_type(filename)
        assert content_type == expected


@pytest.mark.asyncio
async def test_get_temp_file_path():
    """Test temp file path generation."""
    with (
        patch("aiofiles.os.path.exists") as mock_exists,
        patch("aiofiles.os.makedirs") as mock_makedirs,
    ):
        mock_exists.return_value = False

        path = await get_temp_file_path("test.wav")
        assert "test.wav" in path
        mock_makedirs.assert_called_once()


@pytest.mark.asyncio
async def test_list_temp_files():
    """Test listing temp files."""

    class MockEntry:
        def __init__(self, name):
            self.name = name

        def is_file(self):
            return True

    mock_entry = MockEntry("test.wav")

    with (
        patch("aiofiles.os.path.exists") as mock_exists,
        patch("aiofiles.os.scandir") as mock_scandir,
    ):
        mock_exists.return_value = True
        mock_scandir.return_value = [mock_entry]

        files = await list_temp_files()
        assert "test.wav" in files


@pytest.mark.asyncio
async def test_get_temp_dir_size():
    """Test getting temp directory size."""

    class MockEntry:
        def __init__(self, path):
            self.path = path

        def is_file(self):
            return True

    mock_entry = MockEntry("/tmp/test.wav")
    mock_stat = type("MockStat", (), {"st_size": 1024})()

    with (
        patch("aiofiles.os.path.exists") as mock_exists,
        patch("aiofiles.os.scandir") as mock_scandir,
        patch("aiofiles.os.stat") as mock_stat_fn,
    ):
        mock_exists.return_value = True
        mock_scandir.return_value = [mock_entry]
        mock_stat_fn.return_value = mock_stat

        size = await get_temp_dir_size()
        assert size == 1024


@pytest.mark.asyncio
async def test_voice_lookup_in_voices_dir(tmp_path):
    (tmp_path / "af_bella.pt").write_bytes(b"")
    (tmp_path / "am_adam.pt").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("")

    with patch.object(settings, "voices_dir", str(tmp_path)):
        assert await list_voices() == ["af_bella", "am_adam"]
        assert await get_voice_path("af_bella") == str(tmp_path / "af_bella.pt")
        with pytest.raises(FileNotFoundError):
            await get_voice_path("af_nope")


@pytest.mark.asyncio
async def test_model_lookup_in_model_dir(tmp_path):
    (tmp_path / "kokoro.pth").write_bytes(b"")

    with patch.object(settings, "model_dir", str(tmp_path)):
        assert await get_model_path("kokoro.pth") == str(tmp_path / "kokoro.pth")
        with pytest.raises(FileNotFoundError):
            await get_model_path("other.pth")


@pytest.mark.asyncio
async def test_voice_tensor_roundtrip(tmp_path):
    path = str(tmp_path / "voice.pt")
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    await save_voice_tensor(tensor, path)
    assert torch.equal(await load_voice_tensor(path), tensor)

    with pytest.raises(RuntimeError, match="Failed to load voice tensor"):
        await load_voice_tensor(str(tmp_path / "missing.pt"))
    with pytest.raises(RuntimeError, match="Failed to save voice tensor"):
        await save_voice_tensor(tensor, str(tmp_path / "no_dir" / "voice.pt"))


@pytest.mark.asyncio
async def test_model_weights_load(tmp_path):
    path = str(tmp_path / "model.pth")
    torch.save({"w": torch.zeros(2)}, path)

    weights = await load_model_weights(path)
    assert list(weights) == ["w"]
    with pytest.raises(RuntimeError, match="Failed to load model weights"):
        await load_model_weights(str(tmp_path / "missing.pth"))


@pytest.mark.asyncio
async def test_text_json_and_bytes_readers(tmp_path):
    text = tmp_path / "a.json"
    text.write_text('{"k": 1}', encoding="utf-8")

    assert await read_file(str(text)) == '{"k": 1}'
    assert await read_bytes(str(text)) == b'{"k": 1}'
    assert await load_json(str(text)) == {"k": 1}
    assert await verify_model_path(str(text)) is True
    assert await verify_model_path(str(tmp_path / "missing")) is False

    missing = str(tmp_path / "missing")
    for reader in (read_file, read_bytes, load_json):
        with pytest.raises(RuntimeError, match="Failed to"):
            await reader(missing)


@pytest.mark.asyncio
async def test_cleanup_temp_files_removes_only_old_files(tmp_path):
    old = tmp_path / "old.wav"
    new = tmp_path / "new.wav"
    old.write_bytes(b"x")
    new.write_bytes(b"x")
    stale = time.time() - 2 * settings.max_temp_dir_age_hours * 3600
    os.utime(old, (stale, stale))

    with patch.object(settings, "temp_file_dir", str(tmp_path)):
        await cleanup_temp_files()

    assert not old.exists()
    assert new.exists()


@pytest.mark.asyncio
async def test_cleanup_temp_files_creates_missing_dir(tmp_path):
    target = tmp_path / "temp"

    with patch.object(settings, "temp_file_dir", str(target)):
        await cleanup_temp_files()

    assert target.is_dir()
