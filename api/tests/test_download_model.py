import importlib.util
import sys
from pathlib import Path

import pytest

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
