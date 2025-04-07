"""Unified text processing for TTS with smart chunking."""

import re
import time
from typing import AsyncGenerator, Dict, List, Tuple, Optional # Add Optional import

from loguru import logger

from ...core.config import settings
from ...structures.schemas import NormalizationOptions
from .normalizer import normalize_text
from .phonemizer import phonemize
from .vocabulary import tokenize

# Pre-compiled regex patterns for performance
CUSTOM_PHONEMES = re.compile(r"(\[([^\]]|\n)*?\])(\(\/([^\/)]|\n)*?\/\))")
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
        # Normalize step is usually done before smart_split, but phonemize itself might do basic norm
        phonemes = phonemize(text, language, normalize=False)
        t1 = time.time()

        t0 = time.time()
        tokens = tokenize(phonemes)
        t1 = time.time()

    total_time = time.time() - start_time
    logger.debug(
        f"Tokenization took {total_time * 1000:.2f}ms for chunk: '{text[:50]}{'...' if len(text) > 50 else ''}'"
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
    text: str, custom_phenomes_list: Dict[str, str]
) -> List[Tuple[str, List[int], int]]:
    """Process all sentences and return info, preserving trailing newlines."""
    # Split by sentence-ending punctuation, keeping the punctuation
    sentences_parts = re.split(r'([.!?]+|\n+)', text)
    sentences = []
    current_sentence = ""
    for part in sentences_parts:
        if not part:
            continue
        current_sentence += part
        # If the part ends with sentence punctuation or newline, consider it a sentence end
        if re.search(r'[.!?\n]$', part):
            sentences.append(current_sentence)
            current_sentence = ""
    if current_sentence: # Add any remaining part
        sentences.append(current_sentence)


    phoneme_length = len(custom_phenomes_list)
    restored_phoneme_keys = list(custom_phenomes_list.keys()) # Keys to restore

    results = []
    for original_sentence in sentences:
        sentence_text_part = original_sentence.rstrip('\n') # Text without trailing newline for processing
        trailing_newlines = original_sentence[len(sentence_text_part):] # Capture trailing newlines

        if not sentence_text_part.strip(): # Skip empty or whitespace-only sentences
            if trailing_newlines: # If only newlines, represent as empty text with newline marker
                 results.append(("\n", [], 0)) # Store newline marker, no tokens
            continue

        # Restore custom phonemes for this sentence *before* tokenization
        sentence_to_tokenize = sentence_text_part
        restored_count = 0
        # Iterate through *all* possible phoneme IDs that might be in this sentence
        for ph_id in restored_phoneme_keys:
            if ph_id in sentence_to_tokenize:
                sentence_to_tokenize = sentence_to_tokenize.replace(ph_id, custom_phenomes_list[ph_id])
                restored_count+=1
        if restored_count > 0:
             logger.debug(f"Restored {restored_count} custom phonemes for tokenization in: '{sentence_text_part[:30]}...'")


        # Tokenize the text part (without trailing newlines)
        tokens = process_text_chunk(sentence_to_tokenize)

        # Store the original sentence text (including trailing newlines) along with tokens
        results.append((original_sentence, tokens, len(tokens)))

    return results


def handle_custom_phonemes(s: re.Match[str], phenomes_list: Dict[str, str]) -> str:
    latest_id = f"</|custom_phonemes_{len(phenomes_list)}|/>"
    phenomes_list[latest_id] = s.group(0).strip() # Store the full original tag [phoneme](/ipa/)
    logger.debug(f"Replacing custom phoneme {phenomes_list[latest_id]} with ID {latest_id}")
    return latest_id


async def smart_split(
    text: str,
    max_tokens: int = settings.absolute_max_tokens,
    lang_code: str = "a",
    normalization_options: NormalizationOptions = NormalizationOptions(),
) -> AsyncGenerator[Tuple[str, List[int], Optional[float]], None]:
    """Build optimal chunks targeting token limits, handling pause tags and newlines.

    Yields:
        Tuple of (text_chunk, tokens, pause_duration_s).
        If pause_duration_s is not None, it's a pause chunk with empty text/tokens.
        Otherwise, it's a text chunk. text_chunk may end with '\n'.
    """
    start_time = time.time()
    chunk_count = 0
    logger.info(f"Starting smart split for {len(text)} chars, max_tokens={max_tokens}")

    custom_phoneme_list = {}

    # 1. Temporarily replace custom phonemes like [word](/ipa/) with unique IDs
    text_with_ids = CUSTOM_PHONEMES.sub(
        lambda s: handle_custom_phonemes(s, custom_phoneme_list), text
    )
    if custom_phoneme_list:
        logger.debug(f"Found custom phonemes: {custom_phoneme_list}")


    # 2. Normalize the text *with IDs* if required
    normalized_text = text_with_ids
    if settings.advanced_text_normalization and normalization_options.normalize:
        if lang_code in ["a", "b", "en-us", "en-gb"]:
            normalized_text = normalize_text(normalized_text, normalization_options)
            logger.debug("Applied text normalization.")
        else:
            logger.info(
                "Skipping text normalization as it is only supported for english"
            )

    # 3. Split the normalized text by pause tags
    parts = PAUSE_TAG_PATTERN.split(normalized_text)
    logger.debug(f"Split into {len(parts)} parts by pause tags.")


    part_idx = 0
    while part_idx < len(parts):
        text_part = parts[part_idx] # This part contains text and custom phoneme IDs
        part_idx += 1

        if text_part:
            # Process this text part using sentence splitting
            # We pass the text_part *with IDs* to get_sentence_info
            # get_sentence_info will handle restoring phonemes just before tokenization
            sentences = get_sentence_info(text_part, custom_phoneme_list)

            current_chunk_texts = [] # Store original sentence texts for the current chunk
            current_chunk_tokens = []
            current_token_count = 0

            for sentence_text, sentence_tokens, sentence_token_count in sentences:
                 # --- Chunking Logic ---

                # Condition 1: Current sentence alone exceeds max tokens
                if sentence_token_count > max_tokens:
                    logger.warning(f"Single sentence exceeds max_tokens ({sentence_token_count} > {max_tokens}): '{sentence_text[:50]}...'")
                    # Yield any existing chunk first
                    if current_chunk_texts:
                        chunk_text_joined = " ".join(current_chunk_texts) # Join original texts
                        chunk_count += 1
                        logger.info(f"Yielding text chunk {chunk_count} (before oversized sentence): '{chunk_text_joined[:50]}...' ({current_token_count} tokens)")
                        yield chunk_text_joined, current_chunk_tokens, None
                        current_chunk_texts = []
                        current_chunk_tokens = []
                        current_token_count = 0

                    # Yield the oversized sentence as its own chunk
                    # Restore phonemes before yielding the text
                    text_to_yield = sentence_text
                    for p_id, p_val in custom_phoneme_list.items():
                         if p_id in text_to_yield:
                              text_to_yield = text_to_yield.replace(p_id, p_val)

                    chunk_count += 1
                    logger.info(f"Yielding oversized text chunk {chunk_count}: '{text_to_yield[:50]}...' ({sentence_token_count} tokens)")
                    yield text_to_yield, sentence_tokens, None
                    continue # Move to the next sentence

                # Condition 2: Adding the current sentence would exceed max_tokens
                elif current_token_count + sentence_token_count > max_tokens:
                    # Yield the current chunk first
                    if current_chunk_texts:
                        chunk_text_joined = " ".join(current_chunk_texts) # Join original texts
                        chunk_count += 1
                        logger.info(f"Yielding text chunk {chunk_count} (max_tokens limit): '{chunk_text_joined[:50]}...' ({current_token_count} tokens)")
                        yield chunk_text_joined, current_chunk_tokens, None
                    # Start a new chunk with the current sentence
                    current_chunk_texts = [sentence_text]
                    current_chunk_tokens = sentence_tokens
                    current_token_count = sentence_token_count

                # Condition 3: Adding exceeds target_max_tokens when already above target_min_tokens
                elif (current_token_count >= settings.target_min_tokens and
                      current_token_count + sentence_token_count > settings.target_max_tokens):
                    # Yield the current chunk
                    chunk_text_joined = " ".join(current_chunk_texts) # Join original texts
                    chunk_count += 1
                    logger.info(f"Yielding text chunk {chunk_count} (target_max limit): '{chunk_text_joined[:50]}...' ({current_token_count} tokens)")
                    yield chunk_text_joined, current_chunk_tokens, None
                    # Start a new chunk
                    current_chunk_texts = [sentence_text]
                    current_chunk_tokens = sentence_tokens
                    current_token_count = sentence_token_count

                # Condition 4: Add sentence to current chunk (fits within max_tokens and either below target_max or below target_min)
                else:
                    current_chunk_texts.append(sentence_text)
                    current_chunk_tokens.extend(sentence_tokens)
                    current_token_count += sentence_token_count

            # --- End of sentence loop for this text part ---

            # Yield any remaining accumulated chunk for this text part
            if current_chunk_texts:
                chunk_text_joined = " ".join(current_chunk_texts) # Join original texts
                # Restore phonemes before yielding
                text_to_yield = chunk_text_joined
                for p_id, p_val in custom_phoneme_list.items():
                     if p_id in text_to_yield:
                          text_to_yield = text_to_yield.replace(p_id, p_val)

                chunk_count += 1
                logger.info(f"Yielding final text chunk {chunk_count} for part: '{text_to_yield[:50]}...' ({current_token_count} tokens)")
                yield text_to_yield, current_chunk_tokens, None


        # Check if the next part is a pause duration
        if part_idx < len(parts):
            duration_str = parts[part_idx]
            part_idx += 1 # Move past the duration string
            try:
                duration = float(duration_str)
                if duration > 0:
                    chunk_count += 1
                    logger.info(f"Yielding pause chunk {chunk_count}: {duration}s")
                    yield "", [], duration  # Yield pause chunk
            except (ValueError, TypeError):
                 logger.warning(f"Could not parse pause duration: {duration_str}")
                 # If parsing fails, potentially treat the duration_str as text?
                 # For now, just log a warning and skip.


    total_time = time.time() - start_time
    logger.info(
        f"Split completed in {total_time * 1000:.2f}ms, produced {chunk_count} chunks (including pauses)"
    )