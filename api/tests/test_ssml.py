"""Tests for SSML to native inline-tag translation."""

import xml.etree.ElementTree as ET

import pytest

from api.src.services.text_processing.ssml import SSML_ELEMENTS, translate_ssml


def test_plain_text_passthrough():
    assert translate_ssml("Hello world") == "Hello world"
    assert translate_ssml("[pause:1s] native tags untouched") == "[pause:1s] native tags untouched"


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


def test_unsupported_tags_are_noops():
    ssml = '<speak><p><s>Take <emphasis level="strong">this</emphasis> as <prosody rate="slow">given</prosody>.</s></p></speak>'
    assert translate_ssml(ssml) == "Take this as given."


def test_malformed_ssml_raises():
    with pytest.raises(ET.ParseError):
        translate_ssml("<speak>unclosed <voice>")
