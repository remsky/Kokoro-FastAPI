import time

import pytest

from api.src.services.text_processing.original_words import OriginalWords, align_words
from api.src.services.text_processing.text_processor import smart_split
from api.src.structures.schemas import NormalizationOptions, WordTimestamp


def _stamp(word):
    return WordTimestamp(word=word, start_time=0.0, end_time=0.0)


async def _spoken(text, **options):
    """(spoken word, original word) for every word smart_split would send to the model."""
    words = OriginalWords()
    stamps = []
    async for chunk, _, pause in smart_split(
        text,
        normalization_options=NormalizationOptions(**options),
        original_words=words,
    ):
        if pause is None:
            chunk_stamps = [_stamp(word) for word in chunk.split()]
            words.annotate(chunk_stamps)
            stamps += chunk_stamps
    return [(stamp.word, stamp.original_word) for stamp in stamps]


async def test_number_words_point_back_to_the_digits():
    assert await _spoken("I have 105 apples.") == [
        ("I", "I"),
        ("have", "have"),
        ("one", "105"),
        ("hundred", "105"),
        ("and", "105"),
        ("five", "105"),
        ("apples.", "apples."),
    ]


async def test_range_from_the_issue_maps_to_the_written_range():
    spoken = await _spoken("It is 40–60°C today.")

    assert spoken[:2] == [("It", "It"), ("is", "is")]
    assert spoken[-1] == ("today.", "today.")
    assert {original for _, original in spoken[2:-1]} == {"40–60°C"}


async def test_context_dependent_rewrite_resyncs():
    """$5 million reads five million dollars, which no single word normalizes to."""
    spoken = await _spoken("A $5 million budget.")

    assert spoken[0] == ("A", "A")
    assert spoken[-1] == ("budget.", "budget.")
    for _, original in spoken[1:-1]:
        assert original in ("$5", "$5 million")


async def test_words_on_both_sides_of_a_pause():
    assert await _spoken("Room 12 [pause:0.5s] then 3 more") == [
        ("Room", "Room"),
        ("twelve", "12"),
        ("then", "then"),
        ("three", "3"),
        ("more", "more"),
    ]


async def test_without_normalization_each_word_is_itself():
    spoken = await _spoken("I have 105 apples.", normalize=False)

    assert all(word == original for word, original in spoken)


async def test_removed_emoji_leaves_the_words_around_it():
    assert await _spoken("hi 😀 there", remove_emoji=True) == [
        ("hi", "hi"),
        ("there", "there"),
    ]


def test_inserted_word_takes_its_neighbour():
    assert align_words("a b c", "a extra b c", lambda word: word) == [
        "a",
        "a",
        "b",
        "c",
    ]


def test_dropped_word_is_skipped():
    spoken_form = {"a": "a", "junk": "", "b": "b"}.get

    assert align_words("a junk b", "a b", spoken_form) == ["a", "b"]


def test_tokens_split_off_a_word_share_its_original():
    words = OriginalWords()
    words.add('"hello, world"', '"hello, world"', lambda word: word)
    stamps = [_stamp(token) for token in ['"', "hello", ",", "world", '"']]

    words.annotate(stamps)

    assert [stamp.original_word for stamp in stamps] == [
        '"hello,',
        '"hello,',
        '"hello,',
        'world"',
        'world"',
    ]


def test_unmatched_token_is_left_unset_and_does_not_move_the_cursor():
    words = OriginalWords()
    words.add("one two", "one two", lambda word: word)
    stamps = [_stamp("one"), _stamp("reworded"), _stamp("two")]

    words.annotate(stamps)

    assert [stamp.original_word for stamp in stamps] == ["one", None, "two"]


def test_original_word_is_omitted_unless_set():
    stamp = _stamp("forty")
    assert "original_word" not in stamp.model_dump()

    stamp.original_word = "40"
    assert stamp.model_dump()["original_word"] == "40"


@pytest.mark.parametrize("spoken_form", [lambda word: word, lambda word: "x"])
def test_alignment_stays_linear_on_long_input(spoken_form):
    """Every word mismatching is the worst case for resync."""
    text = " ".join(f"w{index}" for index in range(100_000))

    start = time.perf_counter()
    result = align_words(text, text, spoken_form)

    assert len(result) == 100_000
    assert time.perf_counter() - start < 10
