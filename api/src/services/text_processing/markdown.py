"""Strip markdown formatting for clean TTS input.

Removes structural and inline markup so the text reads naturally.
Runs before the main normalizer to prevent # -> "number", * -> silence, etc.
"""

import re

_FENCED_BLOCK = re.compile(r"^```[^\n]*\n.*?^```", re.MULTILINE | re.DOTALL)
_HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_HORIZONTAL_RULE = re.compile(r"^[\s]*[-*_]{3,}\s*$", re.MULTILINE)
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_BOLD_ITALIC = re.compile(r"\*{1,3}(.+?)\*{1,3}")
_UNDERSCORE_EMPHASIS = re.compile(r"_{1,3}(.+?)_{1,3}")
_STRIKETHROUGH = re.compile(r"~~(.+?)~~")
_INLINE_CODE = re.compile(r"`([^`]+)`")
_BLOCKQUOTE = re.compile(r"^>\s?", re.MULTILINE)
_UNORDERED_LIST = re.compile(r"^[\s]*[-*+]\s+", re.MULTILINE)
_ORDERED_LIST = re.compile(r"^[\s]*\d+\.\s+", re.MULTILINE)
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_REFERENCE_LINK = re.compile(r"^\[[^\]]+\]:\s+.*$", re.MULTILINE)
_TABLE_SEPARATOR = re.compile(r"^\|?[\s:]*-{3,}[\s:|-]*$", re.MULTILINE)
_TABLE_PIPE = re.compile(r"\|")


def normalize_markdown(text: str) -> str:
    """Strip markdown formatting, keep the readable text."""
    text = _FENCED_BLOCK.sub("", text)
    text = _REFERENCE_LINK.sub("", text)
    text = _IMAGE.sub(lambda m: m.group(1) or "", text)
    text = _LINK.sub(r"\1", text)
    text = _HORIZONTAL_RULE.sub("", text)
    text = _TABLE_SEPARATOR.sub("", text)
    text = _TABLE_PIPE.sub(" ", text)
    text = _HEADING.sub("", text)
    text = _BLOCKQUOTE.sub("", text)
    text = _UNORDERED_LIST.sub("", text)
    text = _ORDERED_LIST.sub("", text)
    text = _BOLD_ITALIC.sub(r"\1", text)
    text = _UNDERSCORE_EMPHASIS.sub(r"\1", text)
    text = _STRIKETHROUGH.sub(r"\1", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _HTML_TAG.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text
