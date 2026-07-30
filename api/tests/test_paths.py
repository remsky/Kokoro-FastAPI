import os
from pathlib import Path
from unittest.mock import patch

import pytest

from api.src.core.paths import (
    _find_file,
    _scan_directories,
    get_content_type,
    get_temp_dir_size,
    get_temp_file_path,
    list_temp_files,
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
