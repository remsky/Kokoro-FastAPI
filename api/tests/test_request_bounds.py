"""Bounds on the two request fields that used to accept anything."""

import pytest
from pydantic import ValidationError

from api.src.structures.schemas import VOLUME_MAX, OpenAISpeechRequest


def _req(**kwargs):
    return OpenAISpeechRequest(input="test", voice="af_heart", **kwargs)


@pytest.mark.parametrize("value", [0.0, 1.0, VOLUME_MAX])
def test_volume_multiplier_accepts_in_range(value):
    assert _req(volume_multiplier=value).volume_multiplier == value


@pytest.mark.parametrize("value", [-1.0, VOLUME_MAX + 0.1, 1e40])
def test_volume_multiplier_rejects_out_of_range(value):
    with pytest.raises(ValidationError):
        _req(volume_multiplier=value)


@pytest.mark.parametrize("value", ["a", "b", "j", "z"])
def test_lang_code_accepts_known(value):
    assert _req(lang_code=value).lang_code == value


@pytest.mark.parametrize("value", ["xx-INVALID", "zz", "en-us", ""])
def test_lang_code_rejects_unknown(value):
    with pytest.raises(ValidationError):
        _req(lang_code=value)


def test_lang_code_default_still_none():
    assert _req().lang_code is None
