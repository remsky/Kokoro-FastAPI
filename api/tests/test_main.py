import logging
from unittest.mock import patch

from api.src.main import _WebPlayerStartupHandler


def _record(msg, *args):
    return logging.LogRecord("uvicorn.error", logging.INFO, "", 0, msg, args, None)


def test_web_player_url_uses_bound_port():
    # issue #545
    record = _record(
        "Uvicorn running on %s://%s:%d (Press CTRL+C to quit)", "http", "0.0.0.0", 3000
    )
    with patch("api.src.main.logger") as log:
        _WebPlayerStartupHandler().emit(record)
    lines = [c.args[0] for c in log.info.call_args_list]
    assert lines[0].endswith("http://0.0.0.0:3000/web/")
    assert lines[1].endswith("http://localhost:3000/web/")


def test_web_player_url_ignores_other_records():
    with patch("api.src.main.logger") as log:
        _WebPlayerStartupHandler().emit(_record("Started server process [%d]", 123))
        _WebPlayerStartupHandler().emit(
            _record(
                "Uvicorn running on unix socket %s (Press CTRL+C to quit)",
                "/tmp/k.sock",
            )
        )
    log.info.assert_not_called()
