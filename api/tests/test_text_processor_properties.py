"""Property tests for the chunker.

test_text_processor.py pins exact outputs for known inputs. These pin what
must hold for any input: line joining and voice splitting keep every word in
order, and the packer only ever moves whitespace between the sentences it is
given. Token counts are faked so no phonemizer runs.
"""

import asyncio
import re

import pytest
from hypothesis import (
    assume,
    given,
    strategies as st,
)
from unicode_segmentation_rs import unicode_sentences

from api.src.services.text_processing import text_processor
from api.src.services.text_processing.text_processor import (
    CONTROL_TAG_PATTERN,
    join_lines,
    pack,
    smart_split,
    split_by_voice,
)
from api.src.structures.schemas import NormalizationOptions

RAW = NormalizationOptions(normalize=False)

any_text = st.text(max_size=300)
lines = st.text(alphabet='ab.!?:…"”’) \t\r\n', max_size=60)
runs = st.text(alphabet="ab \n", max_size=300)
tag = st.sampled_from(
    [
        "",
        "[voice:af_bella]",
        "[voice:am_michael]",
        "[rate:1.5]",
        "[rate:1.0]",
        "[baserate:0.5]",
    ]
)
gap = st.sampled_from(["", " ", "\n\n"])
word = st.from_regex(r"\A[a-z]{1,8}[.,]?\Z")
pieces = st.lists(st.integers(min_value=1, max_value=12), max_size=20).map(
    lambda sizes: [(f"p{i}", [i] * n) for i, n in enumerate(sizes)]
)
tagged = st.lists(st.tuples(tag, gap, word, gap), min_size=1, max_size=8).map(
    lambda parts: "".join("".join(part) for part in parts)
)


def words(text: str) -> list[str]:
    return [word.rstrip(".") for word in text.split()]


def squash(text: str) -> str:
    return "".join(text.split())


def chunks_of(
    text: str, target_max: int, max_tokens: int = 1000
) -> list[tuple[str, list[int]]]:
    async def collect():
        return [
            (chunk, tokens)
            async for chunk, tokens, pause in smart_split(
                text, max_tokens, normalization_options=RAW
            )
            if pause is None
        ]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            text_processor,
            "process_text_chunk",
            lambda text, *args, **kwargs: [0] * len(text.strip()),
        )
        patch.setattr(text_processor.settings, "target_min_tokens", target_max // 2)
        patch.setattr(text_processor.settings, "target_max_tokens", target_max)
        return asyncio.run(collect())


@given(st.one_of(any_text, lines))
def test_join_lines_keeps_words(text):
    assert words(join_lines(text)) == words(text)


@given(st.one_of(any_text, lines))
def test_join_lines_is_idempotent(text):
    once = join_lines(text)
    assert join_lines(once) == once
    assert "\n" not in once


DOUBLED_ENDER = re.compile(r"[.!?:…][\"')\]”’]*\.")


@given(st.one_of(any_text, lines))
def test_join_lines_never_doubles_an_ender(text):
    joined = join_lines(text)
    assert len(DOUBLED_ENDER.findall(joined)) <= len(DOUBLED_ENDER.findall(text))


@given(st.one_of(any_text, tagged))
def test_split_by_voice_keeps_words(text):
    segments = split_by_voice(text, "af_heart")
    spoken = " ".join(segment_text for _, _, segment_text in segments)
    assert spoken.split() == CONTROL_TAG_PATTERN.sub(" ", text).split()
    assert all(0.25 <= rate <= 4.0 for _, rate, _ in segments)


@given(st.one_of(any_text, lines), st.integers(min_value=3, max_value=20))
def test_smart_split_only_moves_whitespace(text, target_max):
    assume("[pause" not in text.lower())
    chunks = [chunk for chunk, _ in chunks_of(text, target_max)]
    assert all(chunk and chunk == chunk.strip() for chunk in chunks)
    sentences = unicode_sentences(join_lines(text))
    assert squash(" ".join(chunks)) == squash("".join(sentences))


@given(
    runs, st.integers(min_value=3, max_value=20), st.integers(min_value=3, max_value=40)
)
def test_smart_split_honours_token_cap(text, target_max, max_tokens):
    chunks = chunks_of(text, target_max, max_tokens)
    assert words(" ".join(chunk for chunk, _ in chunks)) == words(text)
    assert all(
        len(tokens) <= max_tokens or " " not in chunk for chunk, tokens in chunks
    )


@given(
    pieces,
    st.integers(min_value=1, max_value=20),
    st.integers(min_value=0, max_value=20),
    st.integers(min_value=1, max_value=30),
)
def test_pack_keeps_every_piece_in_order(pieces, max_tokens, min_tokens, target_max):
    chunks = list(pack(pieces, max_tokens, min_tokens, target_max))
    assert [t for _, tokens in chunks for t in tokens] == [
        t for _, tokens in pieces for t in tokens
    ]
    assert " ".join(text for text, _ in chunks).split() == [text for text, _ in pieces]
    assert all(text and tokens for text, tokens in chunks)


@given(
    pieces,
    st.integers(min_value=1, max_value=20),
    st.integers(min_value=0, max_value=20),
    st.integers(min_value=1, max_value=30),
)
def test_pack_is_greedy_under_the_caps(pieces, max_tokens, min_tokens, target_max):
    """A chunk only closes when the next piece would not fit, and never passes max_tokens unless it is one piece."""
    chunks = list(pack(pieces, max_tokens, min_tokens, target_max))
    cap = min(target_max, max_tokens)
    assert all(
        len(tokens) <= max_tokens or len(set(tokens)) == 1 for _, tokens in chunks
    )
    for (_, tokens), (_, following) in zip(chunks, chunks[1:]):
        first_next = len(pieces[following[0]][1])
        total = len(tokens) + first_next
        assert total > cap
        assert total > max_tokens or len(tokens) >= min_tokens
