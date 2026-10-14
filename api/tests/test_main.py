"""Tests for startup helpers in main.py."""

import sys
from unittest.mock import patch

from api.src.main import _get_effective_port


def test_get_effective_port_falls_back_to_settings(monkeypatch):
    """With no --port CLI arg the configured settings value is returned."""
    monkeypatch.setattr(sys, "argv", ["uvicorn", "api.src.main:app"])
    # settings.port default is 8880; just check we get an int and it equals settings.port
    from api.src.core.config import settings

    assert _get_effective_port() == settings.port


def test_get_effective_port_reads_cli_arg(monkeypatch):
    """--port N passed on the CLI overrides the settings value."""
    monkeypatch.setattr(sys, "argv", ["uvicorn", "api.src.main:app", "--port", "3000"])
    assert _get_effective_port() == 3000


def test_get_effective_port_reads_cli_arg_equals_form(monkeypatch):
    """--port=N (equals form) is also recognised."""
    monkeypatch.setattr(
        sys, "argv", ["uvicorn", "api.src.main:app", "--port=9001"]
    )
    assert _get_effective_port() == 9001


def test_get_effective_port_short_flag(monkeypatch):
    """-p N short flag is recognised."""
    monkeypatch.setattr(sys, "argv", ["uvicorn", "api.src.main:app", "-p", "4567"])
    assert _get_effective_port() == 4567


def test_get_effective_port_invalid_value_falls_back(monkeypatch):
    """A non-integer --port value falls back to settings.port gracefully."""
    monkeypatch.setattr(
        sys, "argv", ["uvicorn", "api.src.main:app", "--port", "notanumber"]
    )
    from api.src.core.config import settings

    assert _get_effective_port() == settings.port
