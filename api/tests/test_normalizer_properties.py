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

OPTIONS = NormalizationOptions()

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


@given(plain_prose, open_punct, number_token, close_punct, plain_prose)
def test_standalone_numbers_are_spelled_out(before, open_, number, close, after):
    out = normalize_text(f"{before} {open_}{number}{close} {after}", OPTIONS)
    assert not re.search(r"\d", out), out


@example("0_")
@example(".00A")
@example("1.5x faster")
@example("0K.0")
@example("0K,0")
@example("v1.0")
@example("0S")
@given(st.one_of(any_text, dense_text))
def test_idempotent(text):
    once = normalize_text(text, OPTIONS)
    twice = normalize_text(once, OPTIONS)
    assert once.strip() == twice.strip()


@example("", "9" * 40, "")
@example("$", "9" * 43, "")
@given(number_prefix, number_like, number_suffix)
def test_oversized_numbers_never_raise(prefix, number, suffix):
    normalize_text(f"{prefix}{number}{suffix}", OPTIONS)


@given(plain_prose)
def test_plain_prose_is_untouched(text):
    assert normalize_text(text, OPTIONS) == text
