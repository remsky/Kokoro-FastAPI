"""Invariants every registered normalizer must hold, whatever the language.

A new language gets these for free by appearing in LANGUAGES. Language-specific
expectations belong in that language's own table test, like test_normalizer.py
for English.
"""

import pytest
from hypothesis import (
    given,
    strategies as st,
)

from api.src.services.text_processing.normalization import (
    LANGUAGES,
    Normalizer,
    get_normalizer,
)
from api.src.structures.schemas import NormalizationOptions

OPTIONS = NormalizationOptions()
LANGUAGE_IDS = [normalizer.lang_codes[0] for normalizer in LANGUAGES]


@pytest.mark.parametrize("normalizer", LANGUAGES, ids=LANGUAGE_IDS)
@given(st.text(max_size=300))
def test_never_raises(normalizer, text):
    normalizer.normalize(text, OPTIONS)


@pytest.mark.parametrize("normalizer", LANGUAGES, ids=LANGUAGE_IDS)
@given(st.text(max_size=300))
def test_idempotent(normalizer, text):
    once = normalizer.normalize(text, OPTIONS)
    twice = normalizer.normalize(once, OPTIONS)
    assert once.strip() == twice.strip()


def test_every_lang_code_resolves_to_its_normalizer():
    for normalizer in LANGUAGES:
        for code in normalizer.lang_codes:
            assert get_normalizer(code) is normalizer


def test_unregistered_language_has_no_normalizer():
    assert get_normalizer("xx") is None


def test_base_only_touches_quotes_punctuation_and_whitespace():
    base = Normalizer()
    untouched = "Pay $3,497 at 10:35 on www.example.com, e.g. Dr. Who's 1st"
    assert base.normalize(untouched, OPTIONS) == untouched
    assert (
        base.normalize("“quoted”\t你好，世界\n\nnext", OPTIONS)
        == '"quoted" 你好, 世界 next'
    )
