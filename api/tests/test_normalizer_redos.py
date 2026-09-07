"""Regression tests for ReDoS / catastrophic-backtracking in the text
normalizer. Each adversarial input previously drove normalize_text() into
super-linear time, blocking the async event loop (a single-request DoS,
since normalize_text runs synchronously inside the request path).
"""

import time

from api.src.services.text_processing.normalization import normalize_text
from api.src.structures.schemas import NormalizationOptions

OPTS = NormalizationOptions()
BUDGET_S = 2.0  # generous ceiling; pathological inputs previously took 10-65s


def _elapsed(payload: str, opts: NormalizationOptions = OPTS) -> float:
    start = time.monotonic()
    normalize_text(payload, opts)
    return time.monotonic() - start


def test_long_word_run_is_fast():
    """URL_PATTERN backtracked from every position of a dotless word run (pr #489)."""
    assert _elapsed("a" * 100_000) < BUDGET_S


def test_dotted_run_is_fast():
    """'a.a.a...' hit URL/EMAIL domain scans and the dotted-acronym handler (pr #489)."""
    assert _elapsed("a." * 50_000) < BUDGET_S
    assert _elapsed(".a" * 50_000) < BUDGET_S


def test_at_sign_run_is_fast():
    """EMAIL_PATTERN local/domain runs on an '@' flood (pr #489)."""
    assert _elapsed("a@" * 50_000) < BUDGET_S


def test_scaling_is_linear():
    """4x input must cost far less than the ~16x a quadratic scan implies (pr #489)."""
    small = _elapsed("a." * 25_000)
    large = _elapsed("a." * 100_000)
    assert large / max(small, 1e-3) < 9.0


def test_dash_flood_is_fast():
    """Double-dash rule in the symbols pass (issue #249)."""
    assert _elapsed("-" * 100_000) < BUDGET_S
    assert _elapsed(" --" * 30_000) < BUDGET_S


def test_digit_flood_is_fast():
    """Bare digit runs must not backtrack quadratically in the number passes (pr #492)."""
    assert _elapsed("9" * 80_000) < BUDGET_S


def test_number_floods_scale_linearly():
    """Comma runs and bare digit runs: the version regex went quadratic on the latter."""
    for unit in ("1", "1,", "123,", "123,4", "$1,"):
        small = _elapsed(unit * 5_000)
        large = _elapsed(unit * 20_000)
        assert large / max(small, 0.05) < 9.0, unit


def test_unit_flood_scales_linearly():
    """UNIT_PATTERN restarted its thousands scan after every comma."""
    opts = NormalizationOptions(unit_normalization=True)
    for unit in ("111,", "999,", "111,111,", "1.5", "1 kb"):
        small = _elapsed(unit * 5_000, opts)
        large = _elapsed(unit * 20_000, opts)
        assert large / max(small, 0.05) < 9.0, unit


def test_pattern_floods_scale_linearly():
    """Every remaining pattern: time, phone, money, multipliers, versions, ranges, acronyms, contractions."""
    opts = NormalizationOptions(phone_normalization=True)
    for unit in (
        "12:34 ",
        "555-",
        "$1.5k",
        "1 thousand ",
        "1.2.3.",
        "1-",
        "1.1.1.",
        "A.",
        "there're ",
    ):
        small = _elapsed(unit * 500, opts)
        large = _elapsed(unit * 2_000, opts)
        assert large / max(small, 0.05) < 9.0, unit


def test_huge_digit_run_does_not_raise():
    """float() overflows to inf on 310+ digits, which used to crash the request."""
    assert normalize_text("9" * 400, OPTS) == "9" * 400
    assert normalize_text("$" + "9" * 400, OPTS).startswith(" dollar ")
    assert normalize_text("123," * 400, OPTS)


def test_join_lines_is_fast():
    """Newline and space floods must stay linear through join_lines."""
    from api.src.services.text_processing.text_processor import join_lines

    for text in (
        "\n" + " " * 80_000,
        " " * 80_000 + "\n",
        "\n " * 40_000,
        " \n" * 40_000,
        "\r\n" * 40_000,
        "\n\xa0" * 40_000,
        "\n" + "\xa0" * 80_000,
        "a\n\n" * 30_000,
    ):
        start = time.monotonic()
        join_lines(text)
        assert time.monotonic() - start < BUDGET_S


def test_unbalanced_brackets_are_fast():
    """Stray brackets must not hang the splitter (issue #287)."""
    from api.src.services.text_processing.text_processor import get_sentence_info

    text = "MR. BOWERS: Nothing that I can recall. [Emphasis supplied.).\n" * 100
    start = time.monotonic()
    normalize_text(text, OPTS)
    list(get_sentence_info(text))
    assert time.monotonic() - start < BUDGET_S


def test_decimal_normalization_preserved():
    """the lookbehind must not break normal decimal rendering (pr #492)."""
    out = normalize_text("costs 3.14 dollars", OPTS)
    assert "point" in out


def test_url_normalization_preserved():
    """The hardened patterns must still normalize real URLs and emails (pr #489)."""
    assert "google" in normalize_text("visit https://google.com now", OPTS)
    assert "example" in normalize_text("see www.example.com/x", OPTS)
    out = normalize_text("mail me at a.user@example.org please", OPTS)
    assert "at" in out and "example" in out


def test_dotted_acronym_preserved():
    """The bounded acronym handler must still hyphenate 'U.S.A. b' (pr #489)."""
    assert "U-S-A-" in normalize_text("The U.S.A. beats all", OPTS)
