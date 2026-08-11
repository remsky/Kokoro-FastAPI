"""Tests for LaTeX math normalization."""

import pytest

from api.src.services.text_processing.latex import normalize_latex
from api.src.services.text_processing.normalizer import normalize_text
from api.src.structures.schemas import NormalizationOptions


@pytest.mark.parametrize(
    "latex,expected",
    [
        ("$E = mc^2$", "E = mc squared"),
        ("$\\frac{a}{b}$", "a over b"),
        ("$\\sqrt{x^2 + y^2}$", "the square root of x squared + y squared"),
        ("$\\sqrt[3]{x}$", "the cube root of x"),
        ("$\\binom{n}{k}$", "n choose k"),
        ("$x_1 + x_2$", "x sub 1 + x sub 2"),
        ("$a^{10}$", "a to the power of 10"),
        ("$\\alpha \\le \\beta$", "alpha less than or equal to beta"),
        ("$\\int_0^1 f(x) dx$", "the integral from 0 to 1 f(x) dx"),
        ("$\\sum_{i=1}^{n} i$", "the sum from i=1 to n i"),
        ("\\[\\frac{1}{2} \\times \\frac{3}{4}\\]", "1 over 2 times 3 over 4"),
        ("\\(a + b\\)", "a + b"),
        ("$$E = mc^2$$", "E = mc squared"),
        ("$\\mathbf{v} \\cdot \\mathbf{w}$", "v times w"),
    ],
)
def test_spoken_forms(latex, expected):
    assert normalize_latex(latex).strip() == expected


@pytest.mark.parametrize(
    "text",
    [
        "It cost $5 and then $6 total.",
        "A $100 discount on the $250 item.",
        "no math here at all",
        "email me at a@b.com for 50% off",
    ],
)
def test_non_math_is_untouched(text):
    assert normalize_latex(text) == text


def test_surrounding_prose_survives():
    assert (
        normalize_latex("Einstein said $E = mc^2$ and that was that.").strip()
        == "Einstein said E = mc squared and that was that."
    )


def test_bad_arg_count_returns_verbatim():
    text = "$\\frac{a$"
    assert normalize_latex(text) == text


def test_unsupported_construct_does_not_leak_markup():
    result = normalize_latex("$\\begin{matrix} a \\\\ b \\end{matrix}$")
    assert "\\" not in result and "%s" not in result


def test_span_longer_than_cap_is_skipped():
    text = "$" + "\\alpha " * 60 + "$"
    assert normalize_latex(text) == text


def test_disabled_by_default():
    text = "The value is $\\frac{a}{b}$."
    assert "over" not in normalize_text(text, NormalizationOptions())


def test_enabled_runs_before_symbol_replacement():
    text = "The value is $\\frac{a}{b}$."
    result = normalize_text(text, NormalizationOptions(latex_normalization=True))
    assert "a over b" in result
    assert "dollar" not in result
