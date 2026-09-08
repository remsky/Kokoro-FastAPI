import time

import pytest

from api.src.services.text_processing import text_processor
from api.src.services.text_processing.normalization import Normalizer
from api.src.services.text_processing.text_processor import (
    check_speakable,
    get_sentence_info,
    join_lines,
    process_text_chunk,
    smart_split,
    split_by_voice,
    strip_emoji,
)
from api.src.structures.schemas import NormalizationOptions


class ShoutingNormalizer(Normalizer):
    lang_codes = ("xx",)

    def numbers(self, text: str) -> str:
        return text.upper()


@pytest.fixture
def shouting_language(monkeypatch):
    registry = {"xx": ShoutingNormalizer()}
    monkeypatch.setattr(text_processor, "get_normalizer", registry.get)
    monkeypatch.setattr(text_processor.settings, "advanced_text_normalization", True)


async def first_chunk_text(**kwargs) -> str:
    async for chunk_text, _, _ in smart_split("hello world", **kwargs):
        return chunk_text


@pytest.mark.asyncio
async def test_smart_split_normalizes_with_the_registered_language(shouting_language):
    assert await first_chunk_text(lang_code="xx") == "HELLO WORLD"


@pytest.mark.asyncio
async def test_smart_split_skips_unregistered_language(shouting_language):
    assert await first_chunk_text(lang_code="yy") == "hello world"


@pytest.mark.asyncio
async def test_smart_split_honors_request_normalize_switch(shouting_language):
    options = NormalizationOptions(normalize=False)
    assert (
        await first_chunk_text(lang_code="xx", normalization_options=options)
        == "hello world"
    )


@pytest.mark.asyncio
async def test_smart_split_honors_global_normalization_switch(
    shouting_language, monkeypatch
):
    monkeypatch.setattr(text_processor.settings, "advanced_text_normalization", False)
    assert await first_chunk_text(lang_code="xx") == "hello world"


@pytest.mark.asyncio
async def test_smart_split_keeps_custom_phonemes_out_of_the_normalizer(
    shouting_language,
):
    text = "hello [Kokoro](/kˈOkəɹO/) world"
    async for chunk_text, _, _ in smart_split(text, lang_code="xx"):
        assert chunk_text == "HELLO [Kokoro](/kˈOkəɹO/) WORLD"
        break


# issue #353
REMOVE_EMOJI = NormalizationOptions(remove_emoji=True)


@pytest.mark.asyncio
async def test_smart_split_keeps_emoji_by_default():
    async for chunk_text, _, _ in smart_split("hello 😊 world", lang_code="a"):
        assert "😊" in chunk_text


@pytest.mark.asyncio
@pytest.mark.parametrize("lang_code", ["a", "z"])
async def test_smart_split_remove_emoji_option(lang_code):
    async for chunk_text, _, _ in smart_split(
        "hello 😊 world", lang_code=lang_code, normalization_options=REMOVE_EMOJI
    ):
        assert chunk_text == "hello world"
    assert [
        chunk
        async for chunk in smart_split(
            "😊", lang_code=lang_code, normalization_options=REMOVE_EMOJI
        )
    ] == []


def test_check_speakable():
    check_speakable("hi")
    check_speakable("[pause:1s]")
    check_speakable("[voice:af_bella]")
    check_speakable("😊")
    for text in ["😊", "⏰ 1️⃣", "  ", "[voice:af_bella] [voice:bm_george]"]:
        with pytest.raises(ValueError, match="no speakable text"):
            check_speakable(
                text, allow_voice_tags=True, normalization_options=REMOVE_EMOJI
            )


def test_strip_emoji_sequences():
    for emoji in ["😊", "⏰", "1️⃣", "👨‍👩‍👧", "🇺🇸", "👍🏽"]:
        assert strip_emoji(f"a {emoji} b") == "a b"
    assert strip_emoji("call 1 or # now") == "call 1 or # now"


def test_strip_emoji_flood_is_fast():
    start = time.monotonic()
    strip_emoji("1\ufe0f" * 20_000 + "😊" * 20_000)
    assert time.monotonic() - start < 2.0


def test_process_text_chunk_basic():
    """Test basic text chunk processing."""
    text = "Hello world"
    tokens = process_text_chunk(text)
    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_process_text_chunk_empty():
    """Test processing empty text."""
    text = ""
    tokens = process_text_chunk(text)
    assert isinstance(tokens, list)
    assert len(tokens) == 0


def test_process_text_chunk_phonemes():
    """Test processing with skip_phonemize."""
    phonemes = "h @ l @U"  # Example phoneme sequence
    tokens = process_text_chunk(phonemes, skip_phonemize=True)
    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_get_sentence_info():
    """Test sentence splitting and info extraction."""
    text = "This is sentence one. This is sentence two! What about three?"
    results = list(get_sentence_info(text))

    assert len(results) == 3
    for sentence, tokens, count in results:
        assert isinstance(sentence, str)
        assert isinstance(tokens, list)
        assert isinstance(count, int)
        assert count == len(tokens)
        assert count > 0


def test_get_sentence_info_abbreviations():
    """Abbreviations, decimals, and ellipses do not force bogus sentence breaks (issue #308, pr #520)."""
    text = (
        "This, that, the other thing, etc. Another sentence... A, b, c, etc., and "
        "more. D, e, f, etc. and more. One, i. e. two. Three, i. e., four. Five, "
        "i.e. six. You have 4.2 messages. Property access: `a.b.c`. Bring pencils, "
        "paper, erasers, etc. Then we can start the exam. Some fruits, e.g. apples, "
        "keep well. Data (Eastern Harbor vs. outer harbor) may vary. Keep that in mind."
    )

    sentences = [s for s, _, _ in get_sentence_info(text)]

    assert sentences == [
        "This, that, the other thing, etc.",
        "Another sentence...",
        "A, b, c, etc., and more.",
        "D, e, f, etc. and more.",
        "One, i. e. two.",
        "Three, i. e., four.",
        "Five, i.e. six.",
        "You have 4.2 messages.",
        "Property access: `a.b.c`.",
        "Bring pencils, paper, erasers, etc.",
        "Then we can start the exam.",
        "Some fruits, e.g. apples, keep well.",
        "Data (Eastern Harbor vs. outer harbor) may vary.",
        "Keep that in mind.",
    ]


@pytest.mark.xfail(
    reason="unicode_sentences drops sentences with no alphanumeric character",
    strict=True,
)
def test_get_sentence_info_keeps_punctuation_only_sentences():
    assert [s for s, _, _ in get_sentence_info("!!! Ok.")] == ["!!!", "Ok."]


def test_get_sentence_info_is_lazy(monkeypatch):
    """Sentences are phonemized as they are pulled, not all up front (pr #513)."""
    calls = []

    def counting_process_text_chunk(text, *args, **kwargs):
        calls.append(text)
        return [1, 2, 3]

    monkeypatch.setattr(
        text_processor, "process_text_chunk", counting_process_text_chunk
    )

    sentences = text_processor.get_sentence_info("One. Two. Three.")
    assert calls == []

    assert next(sentences)[0] == "One."
    assert calls == ["One."]

    next(sentences)
    assert calls == ["One.", "Two."]


@pytest.mark.asyncio
async def test_smart_split_first_chunk_skips_rest_of_text(monkeypatch):
    """Time to first chunk stays flat: it must not phonemize the whole input (pr #513)."""
    calls = []

    def counting_process_text_chunk(text, *args, **kwargs):
        calls.append(text)
        return [1] * 200

    monkeypatch.setattr(
        text_processor, "process_text_chunk", counting_process_text_chunk
    )

    text = " ".join(f"This is sentence {i}." for i in range(200))
    chunks = text_processor.smart_split(text)
    try:
        await anext(chunks)
    finally:
        await chunks.aclose()

    assert len(calls) < 10


@pytest.mark.asyncio
async def test_smart_split_short_text():
    """Test smart splitting with text under max tokens."""
    text = "This is a short test sentence."
    chunks = []
    async for chunk_text, chunk_tokens, _ in smart_split(text):
        chunks.append((chunk_text, chunk_tokens))

    assert len(chunks) == 1
    assert isinstance(chunks[0][0], str)
    assert isinstance(chunks[0][1], list)


@pytest.mark.asyncio
async def test_smart_custom_phenomes():
    """Custom phoneme tags survive splitting (issue #348, pr #350)."""
    text = "This is a short test sentence. [Kokoro](/kˈOkəɹO/) has a feature called custom phenomes. This is made possible by [Misaki](/misˈɑki/), the custom phenomizer that [Kokoro](/kˈOkəɹO/) version 1.0 uses"
    chunks = []
    async for chunk_text, chunk_tokens, pause_duration in smart_split(text):
        chunks.append((chunk_text, chunk_tokens, pause_duration))

    # Should have 1 chunks: text
    assert len(chunks) == 1

    # First chunk: text
    assert chunks[0][2] is None  # No pause
    assert (
        "This is a short test sentence. [Kokoro](/kˈOkəɹO/) has a feature called custom phenomes. This is made possible by [Misaki](/misˈɑki/), the custom phenomizer that [Kokoro](/kˈOkəɹO/) version one uses"
        in chunks[0][0]
    )
    assert len(chunks[0][1]) > 0


@pytest.mark.asyncio
async def test_smart_split_only_phenomes():
    """Test input that is entirely made of phenome annotations (pr #350)."""
    text = "[Kokoro](/kˈOkəɹO/) [Misaki 1.2](/misˈɑki/) [Test](/tɛst/)"
    chunks = []
    async for chunk_text, chunk_tokens, pause_duration in smart_split(
        text, max_tokens=10
    ):
        chunks.append((chunk_text, chunk_tokens, pause_duration))

    assert [c for c, _, _ in chunks] == [
        "[Kokoro](/kˈOkəɹO/)",
        "[Misaki 1.2](/misˈɑki/)",
        "[Test](/tɛst/)",
    ]


@pytest.mark.asyncio
async def test_smart_split_long_text():
    """Test smart splitting with longer text."""
    # Create text that should split into multiple chunks
    text = ". ".join(["This is test sentence number " + str(i) for i in range(20)])

    chunks = []
    async for chunk_text, chunk_tokens, _ in smart_split(text):
        chunks.append((chunk_text, chunk_tokens))

    assert len(chunks) > 1
    for chunk_text, chunk_tokens in chunks:
        assert isinstance(chunk_text, str)
        assert isinstance(chunk_tokens, list)
        assert len(chunk_tokens) > 0


@pytest.mark.asyncio
async def test_smart_split_with_punctuation():
    """Test smart splitting handles punctuation correctly."""
    text = "First sentence! Second sentence? Third sentence; Fourth sentence: Fifth sentence."

    chunks = []
    async for chunk_text, chunk_tokens, _ in smart_split(text):
        chunks.append(chunk_text)

    # Verify punctuation is preserved
    assert all(any(p in chunk for p in "!?;:.") for chunk in chunks)


def test_process_text_chunk_chinese_phonemes():
    """Test processing with Chinese pinyin phonemes."""
    pinyin = "nǐ hǎo lì"  # Example pinyin sequence with tones
    tokens = process_text_chunk(pinyin, skip_phonemize=True, language="z")
    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_get_sentence_info_chinese():
    """Test Chinese sentence splitting and info extraction (pr #321)."""
    text = "这是一个句子。这是第二个句子！第三个问题？"
    results = list(get_sentence_info(text, lang_code="z"))

    assert len(results) == 3
    for sentence, tokens, count in results:
        assert isinstance(sentence, str)
        assert isinstance(tokens, list)
        assert isinstance(count, int)
        assert count == len(tokens)
        assert count > 0


@pytest.mark.asyncio
async def test_smart_split_chinese_short():
    """Test Chinese smart splitting with short text (pr #321)."""
    text = "这是一句话。"
    chunks = []
    async for chunk_text, chunk_tokens, _ in smart_split(text, lang_code="z"):
        chunks.append((chunk_text, chunk_tokens))

    assert len(chunks) == 1
    assert isinstance(chunks[0][0], str)
    assert isinstance(chunks[0][1], list)


@pytest.mark.asyncio
async def test_smart_split_chinese_long():
    """Test Chinese smart splitting with longer text (pr #321)."""
    text = "。".join([f"测试句子 {i}" for i in range(20)])

    chunks = []
    async for chunk_text, chunk_tokens, _ in smart_split(text, lang_code="z"):
        chunks.append((chunk_text, chunk_tokens))

    assert len(chunks) > 1
    for chunk_text, chunk_tokens in chunks:
        assert isinstance(chunk_text, str)
        assert isinstance(chunk_tokens, list)
        assert len(chunk_tokens) > 0


@pytest.mark.asyncio
async def test_smart_split_chinese_punctuation():
    """Test Chinese smart splitting with punctuation preservation (pr #321)."""
    text = "第一句！第二问？第三句；第四句：第五句。"

    chunks = []
    async for chunk_text, _, _ in smart_split(text, lang_code="z"):
        chunks.append(chunk_text)

    # Verify Chinese punctuation is preserved
    assert all(any(p in chunk for p in "！？；：。") for chunk in chunks)


@pytest.mark.asyncio
async def test_smart_split_with_pause():
    """Test smart splitting with pause tags (pr #322)."""
    text = "Hello world [pause:2.5s] How are you?"

    chunks = []
    async for chunk_text, chunk_tokens, pause_duration in smart_split(text):
        chunks.append((chunk_text, chunk_tokens, pause_duration))

    # Should have 3 chunks: text, pause, text
    assert len(chunks) == 3

    # First chunk: text
    assert chunks[0][2] is None  # No pause
    assert "Hello world" in chunks[0][0]
    assert len(chunks[0][1]) > 0

    # Second chunk: pause
    assert chunks[1][2] == 2.5  # 2.5 second pause
    assert chunks[1][0] == ""  # Empty text
    assert len(chunks[1][1]) == 0  # No tokens

    # Third chunk: text
    assert chunks[2][2] is None  # No pause
    assert "How are you?" in chunks[2][0]
    assert len(chunks[2][1]) > 0


@pytest.mark.asyncio
async def test_smart_split_with_two_pause():
    """Test smart splitting with two pause tags."""
    text = "[pause:0.5s][pause:1.67s]0.5"

    chunks = []
    async for chunk_text, chunk_tokens, pause_duration in smart_split(text):
        chunks.append((chunk_text, chunk_tokens, pause_duration))

    # Should have 3 chunks: pause, pause, text
    assert len(chunks) == 3

    # First chunk: pause
    assert chunks[0][2] == 0.5  # 0.5 second pause
    assert chunks[0][0] == ""  # Empty text
    assert len(chunks[0][1]) == 0

    # Second chunk: pause
    assert chunks[1][2] == 1.67  # 1.67 second pause
    assert chunks[1][0] == ""  # Empty text
    assert len(chunks[1][1]) == 0  # No tokens

    # Third chunk: text
    assert chunks[2][2] is None  # No pause
    assert "zero point five" in chunks[2][0]
    assert len(chunks[2][1]) > 0


def test_split_by_voice_no_tags():
    """Text without voice tags is a single segment on the default voice."""
    segments = split_by_voice("Just plain text.", "af_heart")
    assert segments == [("af_heart", 1.0, "Just plain text.")]


def test_split_by_voice_multiple_speakers():
    """Each tag starts a new segment."""
    text = "[voice:af_bella] Hello there. [voice:am_michael] Hi back."
    segments = split_by_voice(text, "af_heart")

    assert segments == [
        ("af_bella", 1.0, "Hello there."),
        ("am_michael", 1.0, "Hi back."),
    ]


def test_split_by_voice_leading_text_uses_default():
    """Text ahead of the first tag belongs to the request voice."""
    text = "Narrator opens. [voice:af_bella] Then Bella speaks."
    segments = split_by_voice(text, "af_heart")

    assert segments == [
        ("af_heart", 1.0, "Narrator opens."),
        ("af_bella", 1.0, "Then Bella speaks."),
    ]


def test_split_by_voice_merges_repeated_voice():
    """A tag that doesn't change speaker shouldn't fragment chunking."""
    text = "[voice:af_bella] One. [voice:af_bella] Two."
    segments = split_by_voice(text, "af_heart")

    assert segments == [("af_bella", 1.0, "One.  Two.")]


def test_split_by_voice_merge_keeps_paragraph_break():
    """Merging same-voice runs keeps the blank line for join_lines."""
    text = "[voice:af_bella]Heading\n\n[voice:af_bella]Body."
    segments = split_by_voice(text, "af_heart")

    assert segments == [("af_bella", 1.0, "Heading\n\nBody.")]


def test_split_by_voice_merge_keeps_words_apart():
    """A no-op tag with no whitespace around it must not glue the words."""
    assert split_by_voice("Hello[voice:af_bella]world", "af_bella") == [
        ("af_bella", 1.0, "Hello world")
    ]


def test_split_by_voice_accepts_combined_voices():
    """Weighted combine syntax survives the tag round trip."""
    segments = split_by_voice("[voice: af_bella(2)+af_sky ] Mixed.", "af_heart")
    assert segments == [("af_bella(2)+af_sky", 1.0, "Mixed.")]


def test_split_by_voice_keeps_pause_tags():
    """Pause tags stay in the text for smart_split to handle."""
    text = "[voice:af_bella] One. [pause:0.5s] [voice:am_michael] Two."
    segments = split_by_voice(text, "af_heart")

    assert segments == [
        ("af_bella", 1.0, "One. [pause:0.5s]"),
        ("am_michael", 1.0, "Two."),
    ]


def test_split_by_voice_empty_text():
    """Empty input still yields one segment so callers keep their fallback."""
    assert split_by_voice("", "af_heart") == [("af_heart", 1.0, "")]


def test_split_by_voice_tags_only():
    """Tags with no speech yield nothing rather than reading the markup aloud."""
    assert split_by_voice("[voice:af_bella] [voice:am_michael]", "af_heart") == []


def test_split_by_rate_segments():
    """Rate tags open segments on the same voice, reverting when re-tagged."""
    text = "Normal. [rate:1.5] Fast. [rate:1.0] Normal again."
    assert split_by_voice(text, "af_heart") == [
        ("af_heart", 1.0, "Normal."),
        ("af_heart", 1.5, "Fast."),
        ("af_heart", 1.0, "Normal again."),
    ]


def test_split_by_rate_clamps_to_speed_bounds():
    assert split_by_voice("[rate:99] Whoa.", "af_heart") == [("af_heart", 4.0, "Whoa.")]
    assert split_by_voice("[rate:0.01] Crawl.", "af_heart") == [
        ("af_heart", 0.25, "Crawl.")
    ]


def test_split_voice_tag_resets_rate():
    """A pace stays with the voice it was set on, a voice change reverts to 1.0."""
    text = "[voice:af_bella] [rate:0.75] Slow Bella. [voice:am_michael] Normal Michael."
    assert split_by_voice(text, "af_heart") == [
        ("af_bella", 0.75, "Slow Bella."),
        ("am_michael", 1.0, "Normal Michael."),
    ]


def test_split_rate_scales_baserate():
    """An explicit rate stays relative to the voice's calibrated base pace."""
    text = "[baserate:0.5] Base. [rate:1.5] Faster. [rate:1.0] Base again."
    assert split_by_voice(text, "af_heart") == [
        ("af_heart", 0.5, "Base."),
        ("af_heart", 0.75, "Faster."),
        ("af_heart", 0.5, "Base again."),
    ]


def test_split_baserate_product_clamps_to_speed_bounds():
    assert split_by_voice("[baserate:2.5] [rate:2.0] Whoa.", "af_heart") == [
        ("af_heart", 4.0, "Whoa.")
    ]


# issue #519, pr #525
JOIN_LINES_CASES = [
    (
        "Steven Erikson\n\nFive riders drew rein in the pass.",
        "Steven Erikson. Five riders drew rein in the pass.",
    ),
    ("End of chapter.\n\nNew chapter begins.", "End of chapter. New chapter begins."),
    ("Really?\n\nYes!\n\nOk", "Really? Yes! Ok"),
    ('"Really?"\n\nYes.', '"Really?" Yes.'),
    ("“Really?”\n\nYes.", "“Really?” Yes."),
    ("It read as follows:\n\nText.", "It read as follows: Text."),
    ("He paused…\n\nThen spoke.", "He paused… Then spoke."),
    ("Heading  \r\n\r\n  Body.", "Heading. Body."),
    ("Heading\n \n\t\nBody.", "Heading. Body."),
    ("Heading\n\xa0\nBody.", "Heading. Body."),
    ("Heading\n\f\nBody.", "Heading. Body."),
    ("Five riders drew\nrein in the pass.", "Five riders drew rein in the pass."),
    ("No trailing period", "No trailing period"),
    ("  padded  ", "padded"),
]


@pytest.mark.parametrize("text, expected", JOIN_LINES_CASES)
def test_join_lines(text, expected):
    assert join_lines(text) == expected


def test_split_words_fits_each_piece(monkeypatch):
    monkeypatch.setattr(
        text_processor, "process_text_chunk", lambda text, *a, **k: [0] * len(text)
    )

    assert text_processor.split_words("a b c d e", 3) == [
        ("a b", [0, 0, 0]),
        ("c d", [0, 0, 0]),
        ("e", [0]),
    ]
    assert text_processor.split_words("abcdef gh", 3) == [
        ("abcdef", [0] * 6),
        ("gh", [0, 0]),
    ]
    assert text_processor.split_words("a [b c](/d/) e", 3) == [
        ("a", [0]),
        ("[b c](/d/)", [0] * 10),
        ("e", [0]),
    ]


@pytest.mark.asyncio
async def test_smart_split_caps_unpunctuated_runs():
    """A long run with no punctuation is cut by words, never truncated by the model (issue #71, #95)."""
    text = "hello\n" * 100
    chunks = [
        (c, t)
        async for c, t, _ in smart_split(
            text, normalization_options=NormalizationOptions(normalize=False)
        )
    ]

    assert " ".join(c for c, _ in chunks).split() == ["hello"] * 100
    assert all(len(t) <= text_processor.settings.absolute_max_tokens for _, t in chunks)
    assert len(chunks) == 2


@pytest.mark.asyncio
async def test_smart_split_paragraph_break_is_a_sentence_boundary():
    """A blank line reads as a sentence break and still packs into one chunk (issue #519)."""
    text = "Steven Erikson\n\nFive riders drew rein in the pass."
    chunks = [c async for c, _, _ in smart_split(text)]

    assert chunks == ["Steven Erikson. Five riders drew rein in the pass."]


@pytest.mark.asyncio
async def test_smart_split_normalize_off_still_joins_lines():
    """Line handling is chunking, not normalization, so it applies either way."""
    text = "Steven Erikson\n\nFive riders drew\nrein in the pass. Then 3 more."
    chunks = [
        c
        async for c, _, _ in smart_split(
            text, normalization_options=NormalizationOptions(normalize=False)
        )
    ]

    assert chunks == ["Steven Erikson. Five riders drew rein in the pass. Then 3 more."]
