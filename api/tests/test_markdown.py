"""Tests for markdown normalization."""

import time

import pytest

from api.src.services.text_processing.markdown import normalize_markdown
from api.src.services.text_processing.normalizer import normalize_text
from api.src.structures.schemas import NormalizationOptions


@pytest.mark.parametrize(
    "md,expected",
    [
        ("# Hello World", "Hello World."),
        ("## Sub heading", "Sub heading."),
        ("### Deep heading?", "Deep heading?"),
        ("**bold text**", "bold text"),
        ("*italic text*", "italic text"),
        ("***bold italic***", "bold italic"),
        ("__underline__", "underline"),
        ("~~struck~~", "struck"),
        ("`inline code`", "inline code"),
        ("[click here](https://example.com)", "click here"),
        ("![alt text](image.png)", "alt text"),
        ("> a quote", "a quote"),
        ("- list item", "list item."),
        ("* list item", "list item."),
        ("1. ordered item", "ordered item."),
        ("my_var and other_var", "my_var and other_var"),
        ("5 * 3 * 2", "5 * 3 * 2"),
        ("**Note:** this", "Note: this"),
        ("(**x**) and *y*.", "(x) and y."),
        ("$x_1 + x_2$", "$x_1 + x_2$"),
    ],
)
def test_strip_formatting(md, expected):
    assert normalize_markdown(md).strip() == expected


def test_fenced_code_block_removed():
    text = "before\n```python\nprint('hi')\n```\nafter"
    result = normalize_markdown(text)
    assert "print" not in result
    assert "before" in result
    assert "after" in result


def test_horizontal_rule_removed():
    text = "above\n---\nbelow"
    result = normalize_markdown(text)
    assert "---" not in result
    assert "above" in result
    assert "below" in result


def test_table_pipes_become_spaces():
    text = "| Name | Age |\n|---|---|\n| Alice | 30 |"
    result = normalize_markdown(text)
    assert result.split("\n") == ["Name, Age.", "", "Alice, 30."]


def test_html_tags_stripped():
    assert normalize_markdown("line one<br/>line two") == "line one line two"


def test_reference_links_removed():
    text = "See [link][1].\n\n[1]: https://example.com"
    result = normalize_markdown(text)
    assert "https" not in result


def test_plain_text_unchanged():
    text = "Just a normal sentence with no formatting."
    assert normalize_markdown(text) == text


def test_disabled_by_default():
    text = "# Title"
    result = normalize_text(text, NormalizationOptions())
    assert "number" in result.lower()


def test_enabled_prevents_hash_to_number():
    text = "# Title"
    result = normalize_text(text, NormalizationOptions(markdown_normalization=True))
    assert "number" not in result.lower()
    assert "Title" in result


def test_markdown_before_latex():
    text = "# Math\n\nThe formula is $x_1^2 = mc^2$."
    opts = NormalizationOptions(markdown_normalization=True, latex_normalization=True)
    result = normalize_text(text, opts)
    assert "number" not in result.lower()
    assert "x sub one squared" in result


@pytest.mark.parametrize(
    "payload",
    [" _a", " *a", "[", "[\n", "<a", "\n", "|", "```x\n"],
)
def test_adversarial_input_is_fast(payload):
    """Every pass must stay linear; unclosed markers used to rescan to end of input."""
    start = time.monotonic()
    normalize_markdown(payload * 40_000)
    assert time.monotonic() - start < 2.0
