"""Tests for model version selection in Settings."""

import pytest
from pydantic import ValidationError

from api.src.core.config import Settings


def test_default_model_version():
    """Stock settings select the baked v1.0 model."""
    s = Settings()
    assert s.model_version == "v1_0"
    assert s.model_repo_id == "hexgrad/Kokoro-82M"
    assert s.model_file == "v1_0/kokoro-v1_0.pth"
    assert "v1_0" in s.voices_dir
    assert s.default_voice == "af_heart"


def test_v1_1_zh_derives_paths_and_default_voice():
    """Selecting v1.1-zh follows through to voices dir and default voice."""
    s = Settings(model_version="v1_1-zh")
    assert s.model_repo_id == "hexgrad/Kokoro-82M-v1.1-zh"
    assert s.model_file == "v1_1-zh/kokoro-v1_1-zh.pth"
    assert s.voices_dir.endswith("v1_1-zh")
    assert s.default_voice == "zf_001"


def test_v1_1_zh_respects_explicit_overrides():
    """Explicitly set voices_dir and default_voice are not rewritten."""
    s = Settings(
        model_version="v1_1-zh",
        voices_dir="/custom/voices",
        default_voice="zf_010",
    )
    assert s.voices_dir == "/custom/voices"
    assert s.default_voice == "zf_010"


def test_unknown_model_version_rejected():
    with pytest.raises(ValidationError, match="Unknown model_version"):
        Settings(model_version="v2_5")
