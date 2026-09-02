"""Strip markdown formatting for clean TTS input.

Removes structural and inline markup so the text reads naturally.
Runs before the main normalizer to prevent # -> "number", * -> silence, etc.
"""

import re

_FENCED_BLOCK = re.compile(r"^```[^\n]*\n.*?^```", re.MULTILINE | re.DOTALL)
_HORIZONTAL_RULE = re.compile(r"^[ \t]*[-*_]{3,}[ \t]*$", re.MULTILINE)
_IMAGE = re.compile(r"!\[([^\[\]\n]*)\]\([^()\n]*\)")
_LINK = re.compile(r"\[([^\[\]\n]+)\]\([^()\n]*\)")
_STAR_MARK = re.compile(r"(?<![\w*])\*{1,3}(?=\S)|(?<=\S)\*{1,3}(?![\w*])")
_UNDERSCORE_MARK = re.compile(r"(?<!\w)_{1,3}(?=\w)|(?<=\w)_{1,3}(?![\w{])")
_STRIKETHROUGH = re.compile(r"~~")
_INLINE_CODE = re.compile(r"`+")
_BLOCKQUOTE = re.compile(r"^>\s?", re.MULTILINE)
_STRUCTURAL_LINE = re.compile(r"^(?:#{1,6}|[ \t]*[-*+]|[ \t]*\d+\.)[ \t]+(.*)$", re.MULTILINE)
_HTML_TAG = re.compile(r"</?[a-zA-Z][^<>\n]*>")
_REFERENCE_LINK = re.compile(r"^\[[^\[\]\n]+\]:[ \t]+.*$", re.MULTILINE)
_TABLE_SEPARATOR = re.compile(r"^\|?[ \t:]*-{3,}[ \t:|-]*$", re.MULTILINE)
_TABLE_ROW = re.compile(r"^\|(.+)\|[ \t]*$", re.MULTILINE)
_TABLE_PIPE = re.compile(r"\|")


def _own_sentence(m: re.Match[str]) -> str:
    body = m.group(1).rstrip()
    return body if not body or body[-1] in ".,;:!?" else body + "."


def _table_row(m: re.Match[str]) -> str:
    return ", ".join(c.strip() for c in m.group(1).split("|")) + "."


def normalize_markdown(text: str) -> str:
    """Strip markdown formatting, keep the readable text."""
    text = _FENCED_BLOCK.sub("", text)
    text = _REFERENCE_LINK.sub("", text)
    text = _IMAGE.sub(lambda m: m.group(1) or "", text)
    text = _LINK.sub(r"\1", text)
    text = _HORIZONTAL_RULE.sub("", text)
    text = _TABLE_SEPARATOR.sub("", text)
    text = _TABLE_ROW.sub(_table_row, text)
    text = _TABLE_PIPE.sub(" ", text)
    text = _BLOCKQUOTE.sub("", text)
    text = _STRUCTURAL_LINE.sub(_own_sentence, text)
    text = _STAR_MARK.sub("", text)
    text = _UNDERSCORE_MARK.sub("", text)
    text = _STRIKETHROUGH.sub("", text)
    text = _INLINE_CODE.sub("", text)
    text = _HTML_TAG.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text
