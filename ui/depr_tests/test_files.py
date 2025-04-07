import os
from unittest.mock import patch

import pytest

from ui.lib import files
from ui.lib.config import AUDIO_FORMATS


@pytest.fixture
def mock_dirs(tmp_path):
    """Create temporary input and output directories"""
    inputs_dir = tmp_path / "inputs"
    outputs_dir = tmp_path / "outputs"
    inputs_dir.mkdir()
    outputs_dir.mkdir()

    with (
        patch("ui.lib.files.INPUTS_DIR", str(inputs_dir)),
        patch("ui.lib.files.OUTPUTS_DIR", str(outputs_dir)),
    ):
        yield inputs_dir, outputs_dir


def test_list_input_files_empty(mock_dirs):
    """Test listing input files from empty directory"""
    assert files.list_input_files() == []


def test_list_input_files(mock_dirs):
    """Test listing input files with various files"""
    inputs_dir, _ = mock_dirs

    # Create test files
    (inputs_dir / "test1.txt").write_text("content1")
    (inputs_dir / "test2.txt").write_text("content2")
    (inputs_dir / "nottext.pdf").write_text("should not be listed")

    result = files.list_input_files()
    assert len(result) == 2
    assert "test1.txt" in result
    assert "test2.txt" in result
    assert "nottext.pdf" not in result


def test_list_output_files_empty(mock_dirs):
    """Test listing output files from empty directory"""
    assert files.list_output_files() == []


def test_list_output_files(mock_dirs):
    """Test listing output files with various formats"""
    _, outputs_dir = mock_dirs

    # Create test files for each format
    for fmt in AUDIO_FORMATS:
        (outputs_dir / f"test.{fmt}").write_text("dummy content")
    (outputs_dir / "test.txt").write_text("should not be listed")

    result = files.list_output_files()
    assert len(result) == len(AUDIO_FORMATS)
    for fmt in AUDIO_FORMATS:
        assert any(f".{fmt}" in file for file in result)


def test_read_text_file_empty_filename(mock_dirs):
    """Test reading with empty filename"""
    assert files.read_text_file("") == ""


def test_read_text_file_nonexistent(mock_dirs):
    """Test reading nonexistent file"""
    assert files.read_text_file("nonexistent.txt") == ""


def test_read_text_file_success(mock_dirs):
    """Test successful file reading"""
    inputs_dir, _ = mock_dirs
    content = "Test content\nMultiple lines"
    (inputs_dir / "test.txt").write_text(content)

    assert files.read_text_file("test.txt") == content


def test_save_text_empty(mock_dirs):
    """Test saving empty text"""
    assert files.save_text("") is None
    assert files.save_text("   ") is None


def test_save_text_auto_filename(mock_dirs):
    """Test saving text with auto-generated filename"""
    inputs_dir, _ = mock_dirs

    # First save
    filename1 = files.save_text("content1")
    assert filename1 == "input_1.txt"
    assert (inputs_dir / filename1).read_text() == "content1"

    # Second save
    filename2 = files.save_text("content2")
    assert filename2 == "input_2.txt"
    assert (inputs_dir / filename2).read_text() == "content2"


def test_save_text_custom_filename(mock_dirs):
    """Test saving text with custom filename"""
    inputs_dir, _ = mock_dirs

    filename = files.save_text("content", "custom.txt")
    assert filename == "custom.txt"
    assert (inputs_dir / filename).read_text() == "content"


def test_save_text_duplicate_filename(mock_dirs):
    """Test saving text with duplicate filename"""
    inputs_dir, _ = mock_dirs

    # First save
    filename1 = files.save_text("content1", "test.txt")
    assert filename1 == "test.txt"

    # Save with same filename
    filename2 = files.save_text("content2", "test.txt")
    assert filename2 == "test_1.txt"

    assert (inputs_dir / "test.txt").read_text() == "content1"
    assert (inputs_dir / "test_1.txt").read_text() == "content2"


def test_delete_all_input_files(mock_dirs):
    """Test deleting all input files"""
    inputs_dir, _ = mock_dirs

    # Create test files
    (inputs_dir / "test1.txt").write_text("content1")
    (inputs_dir / "test2.txt").write_text("content2")
    (inputs_dir / "keep.pdf").write_text("should not be deleted")

    assert files.delete_all_input_files() is True
    remaining_files = list(inputs_dir.iterdir())
    assert len(remaining_files) == 1
    assert remaining_files[0].name == "keep.pdf"


def test_delete_all_output_files(mock_dirs):
    """Test deleting all output files"""
    _, outputs_dir = mock_dirs

    # Create test files
    for fmt in AUDIO_FORMATS:
        (outputs_dir / f"test.{fmt}").write_text("dummy content")
    (outputs_dir / "keep.txt").write_text("should not be deleted")

    assert files.delete_all_output_files() is True
    remaining_files = list(outputs_dir.iterdir())
    assert len(remaining_files) == 1
    assert remaining_files[0].name == "keep.txt"


def test_process_uploaded_file_empty_path(mock_dirs):
    """Test processing empty file path"""
    assert files.process_uploaded_file("") is False


def test_process_uploaded_file_invalid_extension(mock_dirs, tmp_path):
    """Test processing file with invalid extension"""
    test_file = tmp_path / "test.pdf"
    test_file.write_text("content")
    assert files.process_uploaded_file(str(test_file)) is False


def test_process_uploaded_file_success(mock_dirs, tmp_path):
    """Test successful file upload processing"""
    inputs_dir, _ = mock_dirs

    # Create source file
    source_file = tmp_path / "test.txt"
    source_file.write_text("test content")

    assert files.process_uploaded_file(str(source_file)) is True
    assert (inputs_dir / "test.txt").read_text() == "test content"


def test_process_uploaded_file_duplicate(mock_dirs, tmp_path):
    """Test processing file with duplicate name"""
    inputs_dir, _ = mock_dirs

    # Create existing file
    (inputs_dir / "test.txt").write_text("existing content")

    # Create source file
    source_file = tmp_path / "test.txt"
    source_file.write_text("new content")

    assert files.process_uploaded_file(str(source_file)) is True
    assert (inputs_dir / "test.txt").read_text() == "existing content"
    assert (inputs_dir / "test_1.txt").read_text() == "new content"
