"""Tests for settings loading and model version selection."""

import pytest
from pydantic import ValidationError

from api.src.core.config import Settings, settings, unrecognized_env_file_keys
from api.src.structures.schemas import CaptionedSpeechRequest, OpenAISpeechRequest

# keys removed from Settings, plus ones that were never fields, that a real .env may still carry
STALE_KEYS = [
    "OUTPUT_DIR",
    "OUTPUT_DIR_SIZE_LIMIT_MB",
    "SAMPLE_RATE",
    "API_LOG_LEVEL",
    "DOWNLOAD_MODEL",
]


def test_unknown_env_file_keys_do_not_block_startup(tmp_path):
    """An upgraded install keeps its old .env, so a dropped key must not stop the server booting."""
    env_file = tmp_path / "dotenv"
    env_file.write_text(
        "\n".join(f"{k}=x" for k in STALE_KEYS) + "\nPORT=9001\n", encoding="utf-8"
    )

    settings = Settings(_env_file=str(env_file))

    assert settings.port == 9001  # real keys still load
    for key in STALE_KEYS:
        assert not hasattr(settings, key.lower())


def test_unrecognized_keys_are_reported(tmp_path, monkeypatch):
    """Ignoring a key silently would turn a typo'd setting into a wrong default with no signal."""
    env_file = tmp_path / "dotenv"
    env_file.write_text("DEFAULT_VOCE=af_bella\nPORT=9001\n", encoding="utf-8")
    monkeypatch.setitem(Settings.model_config, "env_file", str(env_file))

    assert unrecognized_env_file_keys() == ["DEFAULT_VOCE"]


def test_no_env_file_reports_nothing(tmp_path, monkeypatch):
    monkeypatch.setitem(Settings.model_config, "env_file", str(tmp_path / "absent"))
    assert unrecognized_env_file_keys() == []


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


def test_request_voice_default_follows_settings():
    """Request schemas default to the configured voice, not a hardcoded v1.0 one."""
    assert OpenAISpeechRequest.model_fields["voice"].default == settings.default_voice
    assert (
        CaptionedSpeechRequest.model_fields["voice"].default == settings.default_voice
    )


def test_unknown_model_version_rejected():
    with pytest.raises(ValidationError, match="Unknown model_version"):
        Settings(model_version="v2_5")
