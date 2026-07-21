"""Regression tests for ReDoS / catastrophic-backtracking in the text
normalizer. Each adversarial input previously drove normalize_text() into
super-linear time, blocking the async event loop (a single-request DoS,
since normalize_text runs synchronously inside the request path).
"""

import time

from api.src.services.text_processing.normalizer import normalize_text
from api.src.structures.schemas import NormalizationOptions

OPTS = NormalizationOptions()
BUDGET_S = 2.0  # generous ceiling; pathological inputs previously took 10-65s


def _elapsed(payload: str) -> float:
    start = time.monotonic()
    normalize_text(payload, OPTS)
    return time.monotonic() - start


def test_long_word_run_is_fast():
    """URL_PATTERN backtracked from every position of a dotless word run."""
    assert _elapsed("a" * 100_000) < BUDGET_S


def test_dotted_run_is_fast():
    """'a.a.a...' hit URL/EMAIL domain scans and the dotted-acronym handler."""
    assert _elapsed("a." * 50_000) < BUDGET_S
    assert _elapsed(".a" * 50_000) < BUDGET_S


def test_at_sign_run_is_fast():
    """EMAIL_PATTERN local/domain runs on an '@' flood."""
    assert _elapsed("a@" * 50_000) < BUDGET_S


def test_scaling_is_linear():
    """4x input must cost far less than the ~16x a quadratic scan implies."""
    small = _elapsed("a." * 25_000)
    large = _elapsed("a." * 100_000)
    assert large / max(small, 1e-3) < 9.0


def test_url_normalization_preserved():
    """The hardened patterns must still normalize real URLs and emails."""
    assert "google" in normalize_text("visit https://google.com now", OPTS)
    assert "example" in normalize_text("see www.example.com/x", OPTS)
    out = normalize_text("mail me at a.user@example.org please", OPTS)
    assert "at" in out and "example" in out


def test_dotted_acronym_preserved():
    """The bounded acronym handler must still hyphenate 'U.S.A. b'."""
    assert "U-S-A-" in normalize_text("The U.S.A. beats all", OPTS)
