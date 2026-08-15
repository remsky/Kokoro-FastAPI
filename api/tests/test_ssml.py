"""Tests for SSML to native inline-tag translation."""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from api.src.core.config import settings
from api.src.services.text_processing.ssml import SSML_ELEMENTS, translate_ssml

# ref: docs from Azure, Polly, Google TTS
CORPUS = sorted((Path(__file__).parent / "test_data" / "ssml").glob("*.xml"))
MARKUP = re.compile(r"</?[A-Za-z]")

assert CORPUS, "ssml corpus fixtures missing, the corpus test would collect nothing"


def test_plain_text_passthrough():
    assert translate_ssml("Hello world") == "Hello world"
    assert translate_ssml("[pause:1s] native tags untouched") == "[pause:1s] native tags untouched"


def test_roots_merely_starting_with_speak_pass_through():
    """<speaker> is not SSML, only an exact <speak> root may be translated."""
    for text in ("<speaker>Hello</speaker>", "<speak-notes>Hello</speak-notes>"):
        assert translate_ssml(text) == text


def test_namespaced_speak_root_still_translates():
    ssml = '<speak xmlns="http://www.w3.org/2001/10/synthesis">Hi<break time="1s"/>there</speak>'
    assert translate_ssml(ssml) == "Hi [pause:1.0s] there"


def test_break_time_and_strength():
    assert translate_ssml('<speak>Hi<break time="750ms"/>there</speak>') == "Hi [pause:0.75s] there"
    assert translate_ssml('<speak>Hi<break time="1.5s"/>there</speak>') == "Hi [pause:1.5s] there"
    assert translate_ssml('<speak>Hi<break strength="x-strong"/>there</speak>') == "Hi [pause:1.5s] there"
    assert translate_ssml('<speak>Hi<break strength="none"/>there</speak>') == "Hithere"


def test_phoneme_maps_to_custom_ipa():
    out = translate_ssml('<speak><phoneme alphabet="ipa" ph="wˈʊstər">Worcester</phoneme></speak>')
    assert out == "[Worcester](/wˈʊstər/)"
    # non-ipa alphabet speaks the text instead
    out = translate_ssml('<speak><phoneme alphabet="x-sampa" ph="wUst@r">Worcester</phoneme></speak>')
    assert out == "Worcester"


def test_sub_speaks_alias():
    assert translate_ssml('<speak><sub alias="World Wide Web">WWW</sub></speak>') == "World Wide Web"


def test_voice_gated_and_reverts():
    ssml = '<speak>one <voice name="am_michael">two</voice> three</speak>'
    gated = translate_ssml(ssml, default_voice="af_bella", allow_voice_tags=False)
    assert gated == "one two three"
    allowed = translate_ssml(ssml, default_voice="af_bella", allow_voice_tags=True)
    assert allowed == "one [voice:am_michael] two [voice:af_bella] three"


def test_prosody_rate_gated_and_reverts():
    ssml = '<speak>one <prosody rate="slow">two</prosody> three</speak>'
    gated = translate_ssml(ssml, default_voice="af_bella", allow_voice_tags=False)
    assert gated == "one two three"
    allowed = translate_ssml(ssml, default_voice="af_bella", allow_voice_tags=True)
    assert allowed == "one [rate:0.75] two [rate:1.0] three"


def test_prosody_spanning_a_voice_reasserts_the_rate():
    """A voice tag resets the pace downstream, so an enclosing prosody re-emits it"""
    ssml = '<speak><prosody rate="slow">a <voice name="am_michael">b</voice> c</prosody></speak>'
    out = translate_ssml(ssml, default_voice="af_bella", allow_voice_tags=True)
    assert out == (
        "[rate:0.75] a [voice:am_michael] [rate:0.75] b "
        "[voice:af_bella] [rate:0.75] c [rate:1.0]"
    )


def test_prosody_rate_percent_and_number():
    out = translate_ssml(
        '<speak><prosody rate="80%">a</prosody><prosody rate="1.2">b</prosody></speak>',
        default_voice="af_bella",
        allow_voice_tags=True,
    )
    assert out == "[rate:0.8] a [rate:1.0] [rate:1.2] b [rate:1.0]"


def test_prosody_rate_clamps_to_the_request_bounds():
    out = translate_ssml(
        '<speak><prosody rate="900%">a</prosody><prosody rate="0.05">b</prosody></speak>',
        default_voice="af_bella",
        allow_voice_tags=True,
    )
    assert out == "[rate:4.0] a [rate:1.0] [rate:0.25] b [rate:1.0]"


def test_prosody_pitch_only_stays_noop():
    out = translate_ssml(
        '<speak><prosody pitch="high">hi</prosody></speak>',
        default_voice="af_bella",
        allow_voice_tags=True,
    )
    assert out == "hi"


@pytest.mark.parametrize("tag", [k for k, v in SSML_ELEMENTS.items() if v is None])
def test_every_ignored_element_speaks_text_only(tag):
    """Each None entry in the published table really is a no-op, so the table can't drift quietly."""
    out = translate_ssml(
        f"<speak>a <{tag}>x</{tag}> b</speak>",
        default_voice="af_bella",
        allow_voice_tags=True,
    )
    assert out == "a x b"


def test_adjacent_blocks_keep_a_separator():
    ssml = "<speak><p><s>One.</s><s>Two.</s></p><p><s>Three.</s></p></speak>"
    assert translate_ssml(ssml) == "One. Two. Three."


def test_audio_desc_is_dropped_and_fallback_text_survives():
    ssml = '<speak><audio src="purr.ogg"><desc>a cat purring</desc>PURR</audio></speak>'
    assert translate_ssml(ssml) == "PURR"


def test_unsupported_tags_are_noops():
    ssml = '<speak><p><s>Take <emphasis level="strong">this</emphasis> as <prosody rate="slow">given</prosody>.</s></p></speak>'
    assert translate_ssml(ssml) == "Take this as given."


def test_malformed_ssml_raises():
    with pytest.raises(ET.ParseError):
        translate_ssml("<speak>unclosed <voice>")


@pytest.mark.parametrize(
    "prolog",
    [
        '<?xml version="1.0"?>',
        '<?xml version="1.0" encoding="UTF-8"?>\n',
        "<!-- a leading comment -->",
        '<?xml version="1.0"?>\n<!-- both -->\n',
    ],
)
def test_prolog_before_the_root_still_translates(prolog):
    """The spec's own examples all open with a declaration, so it can't defeat detection."""
    assert translate_ssml(f'{prolog}<speak>Hi<break time="1s"/>there</speak>') == "Hi [pause:1.0s] there"


def test_dtd_is_refused_rather_than_expanded():
    ssml = '<!DOCTYPE speak [<!ENTITY a "aaa">]><speak>&a;</speak>'
    with pytest.raises(ET.ParseError):
        translate_ssml(ssml)


def _nested(levels: int) -> str:
    return "<speak>" + "<s>x" * levels + "</s>" * levels + "</speak>"


def test_nesting_is_refused_past_the_configured_depth():
    assert translate_ssml(_nested(settings.ssml_max_depth)) == " ".join("x" * settings.ssml_max_depth)
    with pytest.raises(ET.ParseError):
        translate_ssml(_nested(settings.ssml_max_depth + 1))


def test_max_depth_is_configurable(monkeypatch):
    """The cap has to come from settings, so a deployment with deeper documents can raise it."""
    monkeypatch.setattr(settings, "ssml_max_depth", 3)
    translate_ssml(_nested(3))
    with pytest.raises(ET.ParseError):
        translate_ssml(_nested(4))


def test_depth_cap_fires_long_before_the_recursion_limit():
    """The stack must never get near exhaustion, the 400 path needs frames to run."""
    with pytest.raises(ET.ParseError):
        translate_ssml(_nested(5000))


@pytest.mark.parametrize("route", ["get", "post"])
def test_kill_switch_403s_both_routes(route, monkeypatch):
    from fastapi.testclient import TestClient

    from api.src.main import app

    monkeypatch.setattr(settings, "enable_ssml", False)
    client = TestClient(app)
    resp = client.get("/dev/ssml") if route == "get" else client.post("/dev/ssml", json={"text": "<speak>hi</speak>"})
    assert resp.status_code == 403
    assert resp.json()["detail"]["error"] == "permission_denied"


@pytest.mark.parametrize("doc", CORPUS, ids=lambda p: p.stem)
def test_provider_corpus_never_leaks_markup(doc):
    """Whatever elements a provider's SSML uses, none of it may reach the pipeline as markup."""
    out = translate_ssml(
        doc.read_text(encoding="utf-8"),
        default_voice="af_bella",
        allow_voice_tags=True,
    )
    assert not MARKUP.search(out)
    assert "xmlns" not in out and "interpret-as" not in out
