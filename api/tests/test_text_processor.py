import pytest

from api.src.services.text_processing.text_processor import (
    get_sentence_info,
    process_text_chunk,
    smart_split,
)


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
    results = get_sentence_info(text, {})

    assert len(results) == 3
    for sentence, tokens, count in results:
        assert isinstance(sentence, str)
        assert isinstance(tokens, list)
        assert isinstance(count, int)
        assert count == len(tokens)
        assert count > 0


def test_get_sentence_info_phenomoes():
    """Test sentence splitting and info extraction."""
    text = (
        "This is sentence one. This is </|custom_phonemes_0|/> two! What about three?"
    )
    results = get_sentence_info(text, {"</|custom_phonemes_0|/>": r"sˈɛntᵊns"})

    assert len(results) == 3
    assert "sˈɛntᵊns" in results[1][0]
    for sentence, tokens, count in results:
        assert isinstance(sentence, str)
        assert isinstance(tokens, list)
        assert isinstance(count, int)
        assert count == len(tokens)
        assert count > 0


@pytest.mark.asyncio
async def test_smart_split_short_text():
    """Test smart splitting with text under max tokens."""
    text = "This is a short test sentence."
    chunks = []
    async for chunk_text, chunk_tokens in smart_split(text):
        chunks.append((chunk_text, chunk_tokens))

    assert len(chunks) == 1
    assert isinstance(chunks[0][0], str)
    assert isinstance(chunks[0][1], list)


@pytest.mark.asyncio
async def test_smart_split_long_text():
    """Test smart splitting with longer text."""
    # Create text that should split into multiple chunks
    text = ". ".join(["This is test sentence number " + str(i) for i in range(20)])

    chunks = []
    async for chunk_text, chunk_tokens in smart_split(text):
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
    async for chunk_text, chunk_tokens in smart_split(text):
        chunks.append(chunk_text)

    # Verify punctuation is preserved
    assert all(any(p in chunk for p in "!?;:.") for chunk in chunks)
