"""Unified text processing for TTS with smart chunking."""

import re
import time
from typing import AsyncGenerator, Dict, List, Tuple, Optional

from loguru import logger

from ...core.config import settings
from ...structures.schemas import NormalizationOptions
from .normalizer import normalize_text
from .phonemizer import phonemize
from .vocabulary import tokenize

# Pre-compiled regex patterns for performance
# Updated regex to be more strict and avoid matching isolated brackets
# Only matches complete patterns like [word](/ipa/) and prevents catastrophic backtracking
CUSTOM_PHONEMES = re.compile(r"(\[[^\[\]]*?\]\(\/[^\/\(\)]*?\/\))")
# Pattern to find pause tags like [pause:0.5s]
PAUSE_TAG_PATTERN = re.compile(r"\[pause:(\d+(?:\.\d+)?)s\]", re.IGNORECASE)


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
        t0 = time.time()
        tokens = tokenize(text)
        t1 = time.time()
    else:
        # Normal text processing pipeline
        t0 = time.time()
        t1 = time.time()

        t0 = time.time()
        phonemes = phonemize(text, language)
        # Strip phonemes result to ensure no extra spaces
        phonemes = phonemes.strip()
        t1 = time.time()

        t0 = time.time()
        tokens = tokenize(phonemes)
        t1 = time.time()

    total_time = time.time() - start_time
    logger.debug(
        f"Total processing took {total_time * 1000:.2f}ms for chunk: '{text[:50]}{'...' if len(text) > 50 else ''}'"
    )

    return tokens


async def yield_chunk(
    text: str, tokens: List[int], chunk_count: int
) -> Tuple[str, List[int]]:
    """Yield a chunk with consistent logging."""
    logger.debug(
        f"Yielding chunk {chunk_count}: '{text[:50]}{'...' if len(text) > 50 else ''}' ({len(tokens)} tokens)"
    )
    return text, tokens


def process_text(text: str, language: str = "a") -> List[int]:
    """Process text into token IDs.

    Args:
        text: Text to process
        language: Language code for phonemization

    Returns:
        List of token IDs
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    text = text.strip()
    if not text:
        return []

    return process_text_chunk(text, language)


def get_sentence_info(
    text: str, lang_code: str = "a"
) -> List[Tuple[str, List[int], int]]:
    """Process all sentences and return info"""
    # Detect Chinese text
    is_chinese = lang_code.startswith("z") or re.search(r"[\u4e00-\u9fff]", text)
    if is_chinese:
        # Split using Chinese punctuation
        sentences = re.split(r"([，。！？；])+", text)
    else:
        sentences = re.split(r"([.!?;:])(?=\s|$)", text)

    results = []
    for i in range(0, len(sentences), 2):
        sentence = sentences[i].strip()
        punct = sentences[i + 1] if i + 1 < len(sentences) else ""
        if not sentence:
            continue
        full = sentence + punct
        # Strip the full sentence to remove any leading/trailing spaces before processing
        full = full.strip()
        if not full:  # Skip if empty after stripping
            continue
        tokens = process_text_chunk(full)
        results.append((full, tokens, len(tokens)))
    return results


def handle_custom_phonemes(s: re.Match[str], phenomes_list: Dict[str, str]) -> str:
    latest_id = f"</|custom_phonemes_{len(phenomes_list)}|/>"
    phenomes_list[latest_id] = s.group(0).strip()
    return latest_id


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

    # --- Step 1: Split by Pause Tags FIRST ---
    # This operates on the raw input text
    parts = PAUSE_TAG_PATTERN.split(text)
    logger.debug(f"Split raw text into {len(parts)} parts by pause tags.")

    part_idx = 0
    while part_idx < len(parts):
        text_part_raw = parts[part_idx]  # This part is raw text
        part_idx += 1

        # --- Process Text Part ---
        if text_part_raw and text_part_raw.strip():  # Only process if the part is not empty string
            # Strip leading and trailing spaces to prevent pause tag splitting artifacts
            text_part_raw = text_part_raw.strip()

            # Normalize text (original logic)
            processed_text = text_part_raw
            if settings.advanced_text_normalization and normalization_options.normalize:
                if lang_code in ["a", "b", "en-us", "en-gb"]:
                    processed_text = CUSTOM_PHONEMES.split(processed_text)
                    for index in range(0, len(processed_text), 2):
                        processed_text[index] = normalize_text(processed_text[index], normalization_options)


                    processed_text = "".join(processed_text).strip()
                else:
                    logger.info(
                        "Skipping text normalization as it is only supported for english"
                    )

            # Process all sentences (original logic)
            sentences = get_sentence_info(processed_text, lang_code=lang_code)

            current_chunk = []
            current_tokens = []
            current_count = 0

            for sentence, tokens, count in sentences:
                # Handle sentences that exceed max tokens (original logic)
                if count > max_tokens:
                    # Yield current chunk if any
                    if current_chunk:
                        chunk_text = " ".join(current_chunk).strip()
                        chunk_count += 1
                        logger.debug(
                            f"Yielding chunk {chunk_count}: '{chunk_text[:50]}{'...' if len(processed_text) > 50 else ''}' ({current_count} tokens)"
                        )
                        yield chunk_text, current_tokens, None
                        current_chunk = []
                        current_tokens = []
                        current_count = 0

                    # Split long sentence on commas (original logic)
                    clauses = re.split(r"([,])", sentence)
                    clause_chunk = []
                    clause_tokens = []
                    clause_count = 0

                    for j in range(0, len(clauses), 2):
                        clause = clauses[j].strip()
                        comma = clauses[j + 1] if j + 1 < len(clauses) else ""

                        if not clause:
                            continue

                        full_clause = clause + comma

                        tokens = process_text_chunk(full_clause)
                        count = len(tokens)

                        # If adding clause keeps us under max and not optimal yet
                        if (
                            clause_count + count <= max_tokens
                            and clause_count + count <= settings.target_max_tokens
                        ):
                            clause_chunk.append(full_clause)
                            clause_tokens.extend(tokens)
                            clause_count += count
                        else:
                            # Yield clause chunk if we have one
                            if clause_chunk:
                                chunk_text = " ".join(clause_chunk).strip()
                                chunk_count += 1
                                logger.debug(
                                    f"Yielding clause chunk {chunk_count}: '{chunk_text[:50]}{'...' if len(processed_text) > 50 else ''}' ({clause_count} tokens)"
                                )
                                yield chunk_text, clause_tokens, None
                            clause_chunk = [full_clause]
                            clause_tokens = tokens
                            clause_count = count

                    # Don't forget last clause chunk
                    if clause_chunk:
                        chunk_text = " ".join(clause_chunk).strip()
                        chunk_count += 1
                        logger.debug(
                            f"Yielding final clause chunk {chunk_count}: '{chunk_text[:50]}{'...' if len(processed_text) > 50 else ''}' ({clause_count} tokens)"
                        )
                        yield chunk_text, clause_tokens, None

                # Regular sentence handling (original logic)
                elif (
                    current_count >= settings.target_min_tokens
                    and current_count + count > settings.target_max_tokens
                ):
                    # If we have a good sized chunk and adding next sentence exceeds target,
                    # yield current chunk and start new one
                    chunk_text = " ".join(current_chunk).strip()
                    chunk_count += 1
                    logger.info(
                        f"Yielding chunk {chunk_count}: '{chunk_text[:50]}{'...' if len(processed_text) > 50 else ''}' ({current_count} tokens)"
                    )
                    yield chunk_text, current_tokens, None
                    current_chunk = [sentence]
                    current_tokens = tokens
                    current_count = count
                elif current_count + count <= settings.target_max_tokens:
                    # Keep building chunk while under target max
                    current_chunk.append(sentence)
                    current_tokens.extend(tokens)
                    current_count += count
                elif (
                    current_count + count <= max_tokens
                    and current_count < settings.target_min_tokens
                ):
                    # Only exceed target max if we haven't reached minimum size yet
                    current_chunk.append(sentence)
                    current_tokens.extend(tokens)
                    current_count += count
                else:
                    # Yield current chunk and start new one
                    if current_chunk:
                        chunk_text = " ".join(current_chunk).strip()
                        chunk_count += 1
                        logger.info(
                            f"Yielding chunk {chunk_count}: '{chunk_text[:50]}{'...' if len(processed_text) > 50 else ''}' ({current_count} tokens)"
                        )
                        yield chunk_text, current_tokens, None
                    current_chunk = [sentence]
                    current_tokens = tokens
                    current_count = count

            # Don't forget the last chunk for this text part
            if current_chunk:
                chunk_text = " ".join(current_chunk).strip()
                chunk_count += 1
                logger.info(
                    f"Yielding final chunk {chunk_count} for part: '{chunk_text[:50]}{'...' if len(processed_text) > 50 else ''}' ({current_count} tokens)"
                )
                yield chunk_text, current_tokens, None

        # --- Handle Pause Part ---
        # Check if the next part is a pause duration string
        if part_idx < len(parts):
            duration_str = parts[part_idx]
            # Check if it looks like a valid number string captured by the regex group
            if re.fullmatch(r"\d+(?:\.\d+)?", duration_str):
                part_idx += 1  # Consume the duration string as it's been processed
                try:
                    duration = float(duration_str)
                    if duration > 0:
                        chunk_count += 1
                        logger.info(f"Yielding pause chunk {chunk_count}: {duration}s")
                        yield "", [], duration  # Yield pause chunk
                except (ValueError, TypeError):
                    # This case should be rare if re.fullmatch passed, but handle anyway
                    logger.warning(f"Could not parse valid-looking pause duration: {duration_str}")

    # --- End of parts loop ---
    total_time = time.time() - start_time
    logger.info(
        f"Split completed in {total_time * 1000:.2f}ms, produced {chunk_count} chunks (including pauses)"
    )
