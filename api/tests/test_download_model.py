import hashlib
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from inno_kokoro import enroll

spec = importlib.util.spec_from_file_location(
    "download_model", Path(__file__).parents[2] / "docker/scripts/download_model.py"
)
download_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_model)


def test_hub_check_logs_drift_and_honours_opt_outs(monkeypatch):
    from huggingface_hub import constants

    for flag in ("HF_HUB_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        monkeypatch.setattr(constants, flag, False)
    kokoro_url = "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/config.json"
    tuner_url = f"https://huggingface.co/{enroll.HUB_REPO}/resolve/main/config.json"
    digest = hashlib.sha256(b"{}").hexdigest()
    monkeypatch.setattr(
        download_model, "HUB_CONFIG_SHA256", {kokoro_url: digest, tuner_url: digest}
    )
    get = MagicMock(return_value=b"{}")
    log = MagicMock()
    monkeypatch.setattr(download_model, "_hub_get", get)
    monkeypatch.setattr(download_model, "logger", log)

    download_model.check_hub()
    log.info.assert_not_called()
    assert [c.args[0] for c in get.call_args_list] == [kokoro_url, tuner_url]

    get.return_value = b'{"new": 1}'
    download_model.check_hub()
    assert log.info.call_count == 2

    get.side_effect = OSError("hub unreachable")
    download_model.check_hub()

    get.reset_mock()
    for flag in ("HF_HUB_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        monkeypatch.setattr(constants, flag, True)
        download_model.check_hub()
        monkeypatch.setattr(constants, flag, False)
    get.assert_not_called()


def test_tuner_fetch_failure_is_not_fatal_and_hub_check_still_runs(
    monkeypatch, tmp_path
):
    def boom(_):
        raise ConnectionError("hub unreachable")

    check = MagicMock()
    monkeypatch.setattr(sys, "argv", ["download_model.py", "--output", str(tmp_path)])
    monkeypatch.setattr(download_model, "download_model", lambda _: None)
    monkeypatch.setattr(download_model, "download_tuner", boom)
    monkeypatch.setattr(download_model, "check_hub", check)
    download_model.main()
    check.assert_called_once()

    monkeypatch.setattr(download_model, "download_model", boom)
    with pytest.raises(ConnectionError):
        download_model.main()
    check.assert_called_once()


def test_model_download_falls_back_to_release(monkeypatch, tmp_path):
    names = ["config.json", "kokoro-v1_0.pth"]

    def release_only(url, path):
        if not url.startswith(download_model.FALLBACK_URL):
            raise OSError("hub unreachable")
        Path(path).write_text(url)

    def unreachable(url, path):
        raise OSError("release unreachable")

    monkeypatch.setattr(
        download_model,
        "EXPECTED_SHA256",
        {
            n: hashlib.sha256(f"{download_model.FALLBACK_URL}/{n}".encode()).hexdigest()
            for n in names
        },
    )
    monkeypatch.setattr(download_model, "urlretrieve", release_only)
    download_model.download_model(str(tmp_path))
    assert sorted(p.name for p in tmp_path.iterdir()) == names

    monkeypatch.setattr(download_model, "urlretrieve", unreachable)
    with pytest.raises(OSError, match="release unreachable"):
        download_model.download_model(str(tmp_path))
