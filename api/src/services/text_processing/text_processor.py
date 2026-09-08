"""Unified text processing for TTS with smart chunking."""

import math
import re
import time
from itertools import groupby
from typing import AsyncGenerator, Iterable, Iterator, List, Optional, Tuple

import regex
from loguru import logger
from unicode_segmentation_rs import unicode_sentences

from ...core.config import settings
from ...structures.schemas import VOICE_NAME_BODY, NormalizationOptions, clamp_rate
from .normalization import get_normalizer
from .phonemizer import phonemize
from .vocabulary import tokenize

# Pre-compiled regex patterns for performance
# Updated regex to be more strict and avoid matching isolated brackets
# Only matches complete patterns like [word](/ipa/) and prevents catastrophic backtracking
CUSTOM_PHONEMES = re.compile(r"(\[[^\[\]]*?\]\(\/[^\/\(\)]*?\/\))")
# Pattern to find pause tags like [pause:0.5s]
PAUSE_TAG_PATTERN = re.compile(r"\[pause:(\d+(?:\.\d+)?)s\]", re.IGNORECASE)
EMOJI_PATTERN = regex.compile(
    r" ?(?:[0-9#*]\uFE0F?\u20E3|[\p{Extended_Pictographic}\p{Regional_Indicator}]"
    r"[\p{Extended_Pictographic}\p{Regional_Indicator}\p{Emoji_Modifier}"
    r"\u200D\uFE0F\u20E3\U000E0020-\U000E007F]*) ?"
)
PARAGRAPH_PATTERN = re.compile(r"(?:\r?\n[^\S\r\n]*){2,}")
LINE_PATTERN = re.compile(r"\r?\n[^\S\r\n]*")
# Pattern to find voice tags like [voice:af_bella] or [voice:af_bella(2)+af_sky]
VOICE_TAG_PATTERN = re.compile(rf"\[voice:\s*({VOICE_NAME_BODY}?)\s*\]", re.IGNORECASE)
# Pattern to find voice, rate, and alias base-rate tags in one split pass, like [voice:af_bella] or [rate:1.2]
CONTROL_TAG_PATTERN = re.compile(
    rf"\[voice:\s*({VOICE_NAME_BODY}?)\s*\]|\[rate:\s*(\d+(?:\.\d+)?)\s*\]|\[baserate:\s*(\d+(?:\.\d+)?)\s*\]",
    re.IGNORECASE,
)


def process_text_chunk(
    text: str, language: str = "a", skip_phonemize: bool = False
) -> List[int]:
    """Process a chunk of text through normalization, phonemization, and tokenization.

    Args:
        text: Text chunk to process
        language: Language code for phonemization
        skip_phonemize: If True, treat input as phonemes and skip normalization/phonemization

    Returns:
        List of token IDs
    """
    start_time = time.time()

    # Strip input text to remove any leading/trailing spaces that could cause artifacts
    text = text.strip()

    if not text:
        return []

    if skip_phonemize:
        # Input is already phonemes, just tokenize
        tokens = tokenize(text)
    else:
        # Normal text processing pipeline
        phonemes = phonemize(text, language)
        # Strip phonemes result to ensure no extra spaces
        phonemes = phonemes.strip()
        tokens = tokenize(phonemes)

    total_time = time.time() - start_time
    logger.debug(
        f"Total processing took {total_time * 1000:.2f}ms for chunk: '{text[:50]}{'...' if len(text) > 50 else ''}'"
    )

    return tokens


def get_sentence_info(
    text: str, lang_code: str = "a"
) -> Iterator[Tuple[str, List[int], int]]:
    """Yield (sentence, tokens, token_count) per sentence, phonemizing lazily."""
    for sentence in unicode_sentences(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        tokens = process_text_chunk(sentence)
        yield sentence, tokens, len(tokens)


def split_by_voice(text: str, default_voice: str) -> List[Tuple[str, float, str]]:
    """Split text into (voice, rate, text) segments on [voice:name] and [rate:x] tags.

    Text ahead of the first tag belongs to default_voice at rate 1.0. A
    [baserate:x] tag (injected by alias resolution) carries the speaking
    voice's calibrated pace; [rate:y] tags scale that base rather than
    replace it, so an explicit rate stays relative to how fast the voice
    normally speaks. A voice tag resets both, keeping a pace with the voice
    that was calibrated for it. Runs of segments sharing (voice, rate) are
    merged so a tag that changes nothing costs nothing downstream, and so
    chunking still sees whole paragraphs. Effective rates clamp to the
    request speed bounds (0.25-4.0).
    """
    parts = CONTROL_TAG_PATTERN.split(text)
    if len(parts) == 1:
        return [(default_voice, 1.0, text)]

    segments: List[Tuple[str, float, str]] = []
    current_voice = default_voice
    base_rate = 1.0
    tag_rate = 1.0

    for index, part in enumerate(parts):
        # split() with three groups cycles text, voice, rate, baserate, ... (None for unmatched groups)
        group = index % 4
        if group == 1:
            if part is not None:
                current_voice = part.strip()
                base_rate = 1.0
                tag_rate = 1.0
            continue
        if group == 2:
            if part is not None:
                tag_rate = float(part)
            continue
        if group == 3:
            if part is not None:
                base_rate = float(part)
            continue
        if not part.strip():
            continue
        current_rate = clamp_rate(base_rate * tag_rate)
        if segments and segments[-1][:2] == (current_voice, current_rate):
            previous = segments[-1][2]
            if not previous[-1].isspace() and not part[0].isspace():
                part = " " + part
            segments[-1] = (current_voice, current_rate, previous + part)
        else:
            segments.append((current_voice, current_rate, part))

    # tags were present, so an empty result means there was nothing to say
    return [(voice, rate, text.strip()) for voice, rate, text in segments]


def strip_emoji(text: str) -> str:
    return EMOJI_PATTERN.sub(" ", text)


def check_speakable(
    text: str,
    allow_voice_tags: bool = False,
    normalization_options: Optional[NormalizationOptions] = None,
) -> None:
    """Raise before a stream opens when nothing would be spoken (issue #353)."""
    if allow_voice_tags:
        text = CONTROL_TAG_PATTERN.sub(" ", text)
    if normalization_options is not None and normalization_options.remove_emoji:
        text = strip_emoji(text)
    if not text.strip():
        raise ValueError("Input contains no speakable text")


def check_pause_budget(text: str) -> None:
    """Cap aggregate silence across the whole request, before any segmentation."""
    total_pause_s = math.fsum(
        min(float(match.group(1)), settings.max_pause_duration_s)
        for match in PAUSE_TAG_PATTERN.finditer(text)
    )
    if total_pause_s > settings.max_total_pause_s:
        raise ValueError(
            f"Total pause duration {total_pause_s:.1f}s exceeds the "
            f"{settings.max_total_pause_s:.1f}s limit"
        )


def join_lines(text: str) -> str:
    """Blank lines end a sentence, single newlines are hard wraps."""
    paragraphs = []
    for paragraph in PARAGRAPH_PATTERN.split(text):
        paragraph = LINE_PATTERN.sub(" ", paragraph).strip()
        if paragraph:
            paragraphs.append(paragraph)
    for index in range(len(paragraphs) - 1):
        if paragraphs[index].rstrip("\"')]”’")[-1:] not in ".!?:…":
            paragraphs[index] += "."
    return " ".join(paragraphs)


def split_words(text: str, max_tokens: int) -> List[Tuple[str, List[int]]]:
    """Cut a run of words into pieces that each fit in max_tokens."""
    words = []
    for index, segment in enumerate(CUSTOM_PHONEMES.split(text)):
        words.extend([segment] if index % 2 else segment.split())
    pieces = []
    while words:
        low, high = 1, len(words)
        while low < high:
            mid = (low + high + 1) // 2
            if len(process_text_chunk(" ".join(words[:mid]))) <= max_tokens:
                low = mid
            else:
                high = mid - 1
        piece = " ".join(words[:low])
        pieces.append((piece, process_text_chunk(piece)))
        words = words[low:]
    return pieces


def split_clauses(sentence: str, max_tokens: int) -> Iterator[Tuple[str, List[int]]]:
    """Cut an oversized sentence at clause punctuation, then by words where a clause still does not fit."""
    clauses = re.split(r"([,;:，、；：])", sentence)
    for index in range(0, len(clauses), 2):
        clause = clauses[index].strip()
        if not clause:
            continue
        if index + 1 < len(clauses):
            clause += clauses[index + 1]
        tokens = process_text_chunk(clause)
        if len(tokens) <= max_tokens:
            yield clause, tokens
        else:
            yield from split_words(clause, max_tokens)


def pack(
    pieces: Iterable[Tuple[str, List[int]]],
    max_tokens: int,
    min_tokens: int,
    target_max: int,
) -> Iterator[Tuple[str, List[int]]]:
    """Greedy packer: fill to target_max, run past it up to max_tokens only while under min_tokens."""
    target_max = min(target_max, max_tokens)
    chunk: List[str] = []
    chunk_tokens: List[int] = []
    for text, tokens in pieces:
        total = len(chunk_tokens) + len(tokens)
        if total <= target_max or (
            total <= max_tokens and len(chunk_tokens) < min_tokens
        ):
            chunk.append(text)
            chunk_tokens.extend(tokens)
            continue
        if chunk:
            yield " ".join(chunk).strip(), chunk_tokens
        chunk, chunk_tokens = [text], list(tokens)
    if chunk:
        yield " ".join(chunk).strip(), chunk_tokens


def chunk_sentences(
    sentences: Iterable[Tuple[str, List[int], int]], max_tokens: int
) -> Iterator[Tuple[str, List[int]]]:
    """Pack sentences into chunks. An oversized sentence is packed alone from its clauses."""
    for oversized, group in groupby(sentences, key=lambda item: item[2] > max_tokens):
        if oversized:
            for sentence, _, _ in group:
                yield from pack(
                    split_clauses(sentence, max_tokens),
                    max_tokens,
                    0,
                    settings.target_max_tokens,
                )
        else:
            yield from pack(
                ((sentence, tokens) for sentence, tokens, _ in group),
                max_tokens,
                settings.target_min_tokens,
                settings.target_max_tokens,
            )


async def smart_split(
    text: str,
    max_tokens: int = settings.absolute_max_tokens,
    lang_code: str = "a",
    normalization_options: NormalizationOptions = NormalizationOptions(),
) -> AsyncGenerator[Tuple[str, List[int], Optional[float]], None]:
    """Build optimal chunks targeting 300-400 tokens, never exceeding max_tokens.

    Yields:
        Tuple of (text_chunk, tokens, pause_duration_s).
        If pause_duration_s is not None, it's a pause chunk with empty text/tokens.
        Otherwise, it's a text chunk containing the original text.
    """
    start_time = time.time()
    chunk_count = 0
    logger.info(f"Starting smart split for {len(text)} chars")

    # Split First by Pause Tags
    # This operates on the raw input text
    parts = PAUSE_TAG_PATTERN.split(text)
    logger.debug(f"Split raw text into {len(parts)} parts by pause tags.")

    part_idx = 0
    while part_idx < len(parts):
        text_part_raw = parts[part_idx]  # This part is raw text
        part_idx += 1

        # Processing Text Part
        if (
            text_part_raw and text_part_raw.strip()
        ):  # Only process if the part is not empty string
            if normalization_options.remove_emoji:
                text_part_raw = strip_emoji(text_part_raw)
            processed_text = join_lines(text_part_raw)

            # Normalize text (original logic)
            if settings.advanced_text_normalization and normalization_options.normalize:
                normalizer = get_normalizer(lang_code)
                if normalizer is None:
                    logger.info(
                        f"Skipping text normalization, none registered for lang_code '{lang_code}'"
                    )
                else:
                    processed_text = CUSTOM_PHONEMES.split(processed_text)
                    for index in range(0, len(processed_text), 2):
                        processed_text[index] = normalizer.normalize(
                            processed_text[index], normalization_options
                        )

                    processed_text = "".join(processed_text).strip()

            sentences = get_sentence_info(processed_text, lang_code=lang_code)
            for chunk_text, chunk_tokens in chunk_sentences(sentences, max_tokens):
                chunk_count += 1
                logger.info(
                    f"Yielding chunk {chunk_count}: '{chunk_text[:50]}{'...' if len(chunk_text) > 50 else ''}' ({len(chunk_tokens)} tokens)"
                )
                yield chunk_text, chunk_tokens, None

        # Handle Pause
        # Check if the next part is a pause duration string
        if part_idx < len(parts):
            duration_str = parts[part_idx]
            # Check if it looks like a valid number string captured by the regex group
            if re.fullmatch(r"\d+(?:\.\d+)?", duration_str):
                part_idx += 1  # Consume the duration string as it's been processed
                duration = min(float(duration_str), settings.max_pause_duration_s)
                if duration > 0:
                    chunk_count += 1
                    logger.info(f"Yielding pause chunk {chunk_count}: {duration}s")
                    yield "", [], duration  # Yield pause chunk

    # End of parts loop
    total_time = time.time() - start_time
    logger.debug(
        f"Synthesis completed in {total_time * 1000:.2f}ms, produced {chunk_count} chunks (including pauses)"
    )
