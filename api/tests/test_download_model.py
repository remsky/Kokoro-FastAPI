import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import huggingface_hub.utils
import pytest
from huggingface_hub import constants

spec = importlib.util.spec_from_file_location(
    "download_model", Path(__file__).parents[2] / "docker/scripts/download_model.py"
)
download_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_model)


def test_tuner_fetch_failure_is_not_fatal(monkeypatch, tmp_path):
    def boom(_):
        raise ConnectionError("hub unreachable")

    monkeypatch.setattr(sys, "argv", ["download_model.py", "--output", str(tmp_path)])
    monkeypatch.setattr(download_model, "download_model", lambda _: None)
    monkeypatch.setattr(download_model, "download_tuner", boom)
    download_model.main()

    monkeypatch.setattr(download_model, "download_model", boom)
    with pytest.raises(ConnectionError):
        download_model.main()


def test_update_check_honours_offline_and_telemetry_opt_outs(monkeypatch):
    urls = []

    class Session:
        def get(self, url, timeout):
            urls.append(url)
            return MagicMock(json=lambda: {"version": "9.9.9"})

    monkeypatch.setattr(huggingface_hub.utils, "get_session", Session)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_TELEMETRY", False)
    download_model.check_tuner_update()
    assert urls == [
        "https://huggingface.co/remsky/kokoro-inno-clone-tuner/resolve/main/config.json"
    ]

    for flag in ("HF_HUB_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        monkeypatch.setattr(constants, flag, True)
        download_model.check_tuner_update()
        monkeypatch.setattr(constants, flag, False)
    assert len(urls) == 1
