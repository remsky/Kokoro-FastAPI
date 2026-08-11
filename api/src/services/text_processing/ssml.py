"""SSML to native inline-tag translation.

Input starting with <speak> is parsed as SSML and translated onto the
pipeline's native tokens: <break> to [pause:Ns], <voice> to [voice:name],
<phoneme> to [word](/ipa/), <sub> to its alias. Every other element is
stripped to its text content so valid SSML is never read aloud as markup.
"""

import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

from ...core.config import settings
from ...structures.schemas import clamp_rate

# element surface served by GET /dev/ssml, None means markup dropped and text spoken
SSML_ELEMENTS: Dict[str, Optional[str]] = {
    "speak": "Root element, container only",
    "break": "Silence from time= or strength=, becomes [pause:Ns]",
    "voice": "Speaker change from name=, becomes [voice:name] and reverts at the closing tag, needs a request voice",
    "prosody": "rate= only, becomes [rate:x] and reverts at the closing tag, pitch and volume are ignored",
    "phoneme": "Pronunciation from ph=, alphabet=ipa only, becomes [word](/ipa/)",
    "sub": "Speaks alias= in place of the text",
    "desc": "Dropped with its text, an audio description is not speech",
    "emphasis": None,
    "say-as": None,
    "lang": None,
    "audio": None,
    "mark": None,
    "p": None,
    "s": None,
    "w": None,
}

# break strength to seconds, medium matches a sentence-level pause
BREAK_STRENGTH_S = {
    "none": 0.0,
    "x-weak": 0.0,
    "weak": 0.25,
    "medium": 0.5,
    "strong": 1.0,
    "x-strong": 1.5,
}

TIME_ATTR = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ms|s)\s*$", re.IGNORECASE)

# an xml declaration or comments may precede the root, the spec's own examples carry one
PROLOG = re.compile(r"(?:\s|<\?[^>]*\?>|<!--.*?-->)*", re.DOTALL)

# block level, adjacent ones need a separator, everything else is inline and must not gain one
BLOCK_TAGS = {"p", "s"}

PROSODY_RATE = {
    "x-slow": 0.5,
    "slow": 0.75,
    "medium": 1.0,
    "default": 1.0,
    "fast": 1.25,
    "x-fast": 1.5,
}


def _break_seconds(el: ET.Element) -> float:
    time_attr = el.get("time")
    if time_attr:
        match = TIME_ATTR.match(time_attr)
        if match:
            value = float(match.group(1))
            return value / 1000 if match.group(2).lower() == "ms" else value
    return BREAK_STRENGTH_S.get(el.get("strength", "medium"), 0.5)


def _prosody_rate(value: str) -> float:
    """Parse a prosody rate attribute: keyword, percentage, or bare multiplier."""
    value = value.strip().lower()
    if value in PROSODY_RATE:
        return PROSODY_RATE[value]
    try:
        return clamp_rate(
            float(value[:-1]) / 100 if value.endswith("%") else float(value)
        )
    except ValueError:
        return 1.0


def _render(
    el: ET.Element,
    parts: List[str],
    control_tags: bool,
    current_voice: str,
    current_rate: float = 1.0,
    depth: int = 0,
) -> None:
    if depth > settings.ssml_max_depth:
        raise ET.ParseError(
            f"nesting deeper than {settings.ssml_max_depth} levels is not translated"
        )
    tag = el.tag.split("}")[-1].lower()  # tolerate namespaced tags

    if tag == "break":
        seconds = _break_seconds(el)
        if seconds > 0:
            parts.append(f" [pause:{seconds}s] ")
        return
    ph = el.get("ph")
    if tag == "phoneme" and ph and el.get("alphabet", "ipa").lower() == "ipa":
        word = "".join(el.itertext()).strip()
        parts.append(f"[{word}](/{ph.strip()}/)")
        return
    if tag == "sub":
        parts.append(el.get("alias") or "".join(el.itertext()))
        return
    if tag == "desc":
        return

    inner_voice = current_voice
    inner_rate = current_rate
    name = el.get("name")
    if tag == "voice" and control_tags and name:
        inner_voice = name.strip()
        parts.append(f" [voice:{inner_voice}] ")
    if tag == "prosody" and control_tags and el.get("rate"):
        inner_rate = _prosody_rate(el.get("rate", ""))
        if inner_rate != current_rate:
            parts.append(f" [rate:{inner_rate}] ")

    # emphasis, say-as, p, s, lang, etc are no-ops: keep text, drop markup
    if tag in BLOCK_TAGS:
        parts.append(" ")
    if el.text:
        parts.append(el.text)
    for child in el:
        _render(child, parts, control_tags, inner_voice, inner_rate, depth + 1)
        if child.tail:
            parts.append(child.tail)
    if tag in BLOCK_TAGS:
        parts.append(" ")

    if inner_voice != current_voice:
        parts.append(f" [voice:{current_voice}] ")
    if inner_rate != current_rate:
        parts.append(f" [rate:{current_rate}] ")


def translate_ssml(
    text: str, default_voice: str = "", allow_voice_tags: bool = False
) -> str:
    """Return native-tag text if the input is SSML, else the input unchanged.

    Malformed SSML raises ET.ParseError for the caller to surface.
    <voice> and <prosody rate> emit control tags only when allow_voice_tags
    is set, otherwise they are stripped and their content speaks unmodified.
    """
    body = PROLOG.sub("", text, count=1)
    # no SSML dialect needs a DTD, refusing one drops the entity expansion class outright
    if body[:9].upper() == "<!DOCTYPE":
        raise ET.ParseError("DTD is not supported")
    # the spec requires a <speak> root, so anything else is plain text
    if not body.startswith("<speak"):
        return text
    root = ET.fromstring(text)

    parts: List[str] = []
    _render(root, parts, allow_voice_tags and bool(default_voice), default_voice)
    return re.sub(r"\s+", " ", "".join(parts)).strip()
