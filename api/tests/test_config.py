"""Tests for settings loading."""

import api.src.core.config as config_module
from api.src.core.config import Settings, unrecognized_env_file_keys

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
