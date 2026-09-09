"""Property tests for text normalization.

The table in test_normalizer.py pins exact outputs for known inputs. These
tests pin English invariants that must hold for any input, and let hypothesis
search for counterexamples. test_normalizer_contract.py holds the invariants
shared by every language. Counterexamples it has found are pinned with @example
so they run on every seed.
"""

import re

from hypothesis import (
    example,
    given,
    strategies as st,
)

from api.src.services.text_processing.normalization import normalize_text
from api.src.structures.schemas import NormalizationOptions

DEFAULTS = NormalizationOptions()
options = st.builds(
    NormalizationOptions,
    unit_normalization=st.booleans(),
    url_normalization=st.booleans(),
    email_normalization=st.booleans(),
    optional_pluralization_normalization=st.booleans(),
    phone_normalization=st.booleans(),
    caps_normalization=st.booleans(),
    replace_remaining_symbols=st.booleans(),
    remove_emoji=st.booleans(),
)

any_text = st.text(max_size=300)
dense_text = st.text(alphabet="0123456789 .,:;-+$%#_/&@=()'\nabkmstAKS", max_size=24)

number_token = st.one_of(
    st.integers(min_value=-999_999_999, max_value=999_999_999).map(str),
    st.integers(min_value=1000, max_value=999_999_999).map("{:,}".format),
    st.floats(min_value=-99_999, max_value=99_999, allow_nan=False).map(
        "{:.2f}".format
    ),
    st.floats(min_value=0, max_value=999, allow_nan=False).map("{:.1f}".format),
    st.integers(min_value=1, max_value=999).map(lambda n: f"{n}k"),
    st.integers(min_value=1, max_value=999).map(lambda n: f"${n}"),
    st.integers(min_value=1, max_value=999).map(lambda n: f"{n}%"),
    st.integers(min_value=0, max_value=23).flatmap(
        lambda h: st.integers(min_value=0, max_value=59).map(lambda m: f"{h}:{m:02d}")
    ),
)

digit_run = st.integers(min_value=1, max_value=5000).map("9".__mul__)
number_like = st.lists(digit_run, min_size=1, max_size=4).map(".".join)
number_prefix = st.sampled_from(["", "-", "$", "£", "€"])
number_suffix = st.sampled_from(["", "k", "m", "%", ",000"])

plain_word = st.from_regex(r"\A[a-z]{1,12}\Z").filter(
    lambda w: not w.endswith("re") and w not in ("yea", "yeah")
)
plain_prose = st.lists(plain_word, min_size=1, max_size=12).map(" ".join)
open_punct = st.sampled_from(["", "(", "'", '"'])
close_punct = st.sampled_from(["", ".", ",", "!", "?", ")", ":", ";", "'", '"'])


@given(plain_prose, open_punct, number_token, close_punct, plain_prose, options)
def test_standalone_numbers_are_spelled_out(before, open_, number, close, after, opts):
    out = normalize_text(f"{before} {open_}{number}{close} {after}", opts)
    assert not re.search(r"\d", out), out


@example("0_", DEFAULTS)
@example(".00A", DEFAULTS)
@example("1.5x faster", DEFAULTS)
@example("0K.0", DEFAULTS)
@example("0K,0", DEFAULTS)
@example("v1.0", DEFAULTS)
@example("0S", DEFAULTS)
@given(st.one_of(any_text, dense_text), options)
def test_idempotent(text, opts):
    once = normalize_text(text, opts)
    twice = normalize_text(once, opts)
    assert once.strip() == twice.strip()


@example("", "9" * 40, "", DEFAULTS)
@example("$", "9" * 43, "", DEFAULTS)
@given(number_prefix, number_like, number_suffix, options)
def test_oversized_numbers_never_raise(prefix, number, suffix, opts):
    normalize_text(f"{prefix}{number}{suffix}", opts)


@given(plain_prose, options)
def test_plain_prose_is_untouched(text, opts):
    assert normalize_text(text, opts) == text
