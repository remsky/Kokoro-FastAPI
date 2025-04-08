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
    """
    Processes all sentences and returns info, preserving trailing newlines.

    Args:
        text: The input text.
        custom_phenomes_list: A dictionary mapping custom phoneme IDs to their original IPA string representations.

    Returns:
        A list of tuples, where each tuple contains: (original sentence text, list of tokens, token count).
    """
    # 1. Use regex to split sentences, keeping delimiters (.!?) and newlines (\n)
    #    Pattern explanation:
    #    [^.!?\n]+     : Match one or more characters that are not delimiters/newlines (sentence body)
    #    (?:[.!?]+|\n+)? : Optionally match one or more sentence-ending punctuation marks OR one or more newlines
    #    |             : OR
    #    \n+           : Match one or more newlines (handles cases of pure newlines)
    sentences = re.findall(r'[^.!?\n]+(?:[.!?]+|\n+)?|\n+', text)

    restored_phoneme_keys = list(custom_phenomes_list.keys()) # List of phoneme IDs to restore
    results = []

    for original_sentence in sentences:
        # 2. Separate the sentence text content from trailing newlines
        sentence_text_part = original_sentence.rstrip('\n')
        # trailing_newlines = original_sentence[len(sentence_text_part):] # No longer need to store trailing newlines separately, as original_sentence contains them

        # 3. Handle pure newlines or "sentences" containing only whitespace
        if not sentence_text_part.strip():
            # If the original sentence consists of newline(s), represent it as a special newline marker
            if '\n' in original_sentence:
                 results.append(("\n", [], 0)) # Use "\n" to represent pure newline/blank lines, with empty tokens
            continue # Skip completely empty or whitespace-only sentence parts

        # 4. Before tokenization, restore custom phonemes in the sentence text part
        sentence_to_tokenize = sentence_text_part
        restored_count = 0
        if custom_phenomes_list: # Only attempt restoration if the list is not empty
            for ph_id in restored_phoneme_keys:
                if ph_id in sentence_to_tokenize:
                    # Note: The value replaced here should be the phoneme string itself, not the original tag "[word](/ipa/)"
                    # Assume custom_phenomes_list stores {id: ipa_phoneme_string}
                    # If it stores the original tag, the logic in handle_custom_phonemes needs adjustment
                    # Assume the value in custom_phenomes_list is a phoneme string ready for tokenization
                    sentence_to_tokenize = sentence_to_tokenize.replace(ph_id, custom_phenomes_list[ph_id])
                    restored_count += 1
            if restored_count > 0:
                logger.debug(f"Restored {restored_count} custom phonemes for tokenization in: '{sentence_text_part[:30]}...'")

        # 5. Tokenize the text part after restoring phonemes
        # Note: process_text_chunk should be able to handle strings containing actual phoneme symbols
        try:
            tokens = process_text_chunk(sentence_to_tokenize)
        except Exception as e:
             logger.error(f"Tokenization failed for sentence part '{sentence_to_tokenize[:50]}...': {e}")
             tokens = [] # Return empty list on error

        # 6. Store the result: Use original_sentence which includes trailing newlines
        results.append((original_sentence, tokens, len(tokens)))

    return results


def handle_custom_phonemes(s: re.Match[str], phenomes_list: Dict[str, str]) -> str:
    # Stores the *original tag* like "[word](/ipa/)" mapped to the ID
    original_tag = s.group(0).strip()
    latest_id = f"</|custom_phonemes_{len(phenomes_list)}|/>"
    phenomes_list[latest_id] = original_tag
    logger.debug(f"Replacing custom phoneme tag '{original_tag}' with ID {latest_id}")
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
        Otherwise, it's a text chunk containing original formatting (incl. custom phoneme tags).
    """
    start_time = time.time()
    chunk_count = 0
    logger.info(f"Starting smart split for {len(text)} chars, max_tokens={max_tokens}, lang_code={lang_code}")

    # --- Determine if normalization and ID replacement are needed ---
    apply_normalization = (
        settings.advanced_text_normalization
        and normalization_options.normalize
        and lang_code in ["a", "b", "en-us", "en-gb"] # Normalization only for English
    )
    use_ids = apply_normalization # Only use IDs if we are normalizing
    logger.debug(f"Normalization active: {apply_normalization}. Using ID replacement: {use_ids}")

    custom_phoneme_map = {} # Map ID -> Original Tag OR empty if use_ids is False
    processed_text = text # Start with original text

    # --- Step 1: Optional ID Replacement ---
    if use_ids:
        processed_text = CUSTOM_PHONEMES.sub(
            lambda s: handle_custom_phonemes(s, custom_phoneme_map), text
        )
        if custom_phoneme_map:
            logger.debug(f"Found and replaced custom phonemes with IDs: {custom_phoneme_map}")

    # --- Step 2: Optional Normalization ---
    if apply_normalization:
        processed_text = normalize_text(processed_text, normalization_options)
        logger.debug("Applied text normalization.")

    # --- Step 3: Split by Pause Tags ---
    # This operates on `processed_text` which either has IDs or original tags
    parts = PAUSE_TAG_PATTERN.split(processed_text)
    logger.debug(f"Split into {len(parts)} parts by pause tags.")

    part_idx = 0
    while part_idx < len(parts):
        text_part = parts[part_idx] # This part contains text (with IDs or original tags)
        part_idx += 1

        if text_part:
            # --- Process Text Part ---
            # get_sentence_info MUST be able to handle BOTH inputs with IDs (using custom_phoneme_map)
            # AND inputs with original [word](/ipa/) tags (when custom_phoneme_map is empty)
            # It needs to extract IPA phonemes correctly in both cases for tokenization.
            # Crucially, it should return the *original format* sentence text (with IDs or tags)
            try:
                sentences = get_sentence_info(text_part, custom_phoneme_map)
            except Exception as e:
                logger.error(f"get_sentence_info failed for part '{text_part[:50]}...': {e}", exc_info=True)
                continue # Skip this part if sentence processing fails

            current_chunk_texts = [] # Store original format sentence texts for the current chunk
            current_chunk_tokens = []
            current_token_count = 0

            for sentence_text_original_format, sentence_tokens, sentence_token_count in sentences:
                # --- Chunking Logic (remains the same) ---

                # Condition 1: Current sentence alone exceeds max tokens
                if sentence_token_count > max_tokens:
                    logger.warning(f"Single sentence exceeds max_tokens ({sentence_token_count} > {max_tokens}): '{sentence_text_original_format[:50]}...'")
                    # Yield any existing chunk first
                    if current_chunk_texts:
                        chunk_text_to_yield = " ".join(current_chunk_texts)
                        # Restore original tags IF we used IDs
                        if use_ids:
                            for p_id, original_tag_val in custom_phoneme_map.items():
                                chunk_text_to_yield = chunk_text_to_yield.replace(p_id, original_tag_val)
                        chunk_count += 1
                        logger.info(f"Yielding text chunk {chunk_count} (before oversized): '{chunk_text_to_yield[:50]}...' ({current_token_count} tokens)")
                        yield chunk_text_to_yield, current_chunk_tokens, None
                        current_chunk_texts = []
                        current_chunk_tokens = []
                        current_token_count = 0

                    # Yield the oversized sentence as its own chunk
                    text_to_yield = sentence_text_original_format
                    # Restore original tags IF we used IDs
                    if use_ids:
                         for p_id, original_tag_val in custom_phoneme_map.items():
                              text_to_yield = text_to_yield.replace(p_id, original_tag_val)

                    chunk_count += 1
                    logger.info(f"Yielding oversized text chunk {chunk_count}: '{text_to_yield[:50]}...' ({sentence_token_count} tokens)")
                    yield text_to_yield, sentence_tokens, None
                    continue # Move to the next sentence

                # Condition 2: Adding the current sentence would exceed max_tokens
                elif current_token_count + sentence_token_count > max_tokens:
                    # Yield the current chunk first
                    if current_chunk_texts:
                        chunk_text_to_yield = " ".join(current_chunk_texts)
                        if use_ids:
                            for p_id, original_tag_val in custom_phoneme_map.items():
                                chunk_text_to_yield = chunk_text_to_yield.replace(p_id, original_tag_val)
                        chunk_count += 1
                        logger.info(f"Yielding text chunk {chunk_count} (max_tokens limit): '{chunk_text_to_yield[:50]}...' ({current_token_count} tokens)")
                        yield chunk_text_to_yield, current_chunk_tokens, None
                    # Start a new chunk with the current sentence
                    current_chunk_texts = [sentence_text_original_format]
                    current_chunk_tokens = sentence_tokens
                    current_token_count = sentence_token_count

                # Condition 3: Adding exceeds target_max_tokens when already above target_min_tokens
                elif (current_token_count >= settings.target_min_tokens and
                      current_token_count + sentence_token_count > settings.target_max_tokens):
                    # Yield the current chunk
                    chunk_text_to_yield = " ".join(current_chunk_texts)
                    if use_ids:
                        for p_id, original_tag_val in custom_phoneme_map.items():
                            chunk_text_to_yield = chunk_text_to_yield.replace(p_id, original_tag_val)
                    chunk_count += 1
                    logger.info(f"Yielding text chunk {chunk_count} (target_max limit): '{chunk_text_to_yield[:50]}...' ({current_token_count} tokens)")
                    yield chunk_text_to_yield, current_chunk_tokens, None
                    # Start a new chunk
                    current_chunk_texts = [sentence_text_original_format]
                    current_chunk_tokens = sentence_tokens
                    current_token_count = sentence_token_count

                # Condition 4: Add sentence to current chunk
                else:
                    current_chunk_texts.append(sentence_text_original_format)
                    current_chunk_tokens.extend(sentence_tokens)
                    current_token_count += sentence_token_count

            # --- End of sentence loop for this text part ---

            # Yield any remaining accumulated chunk for this text part
            if current_chunk_texts:
                chunk_text_to_yield = " ".join(current_chunk_texts)
                # Restore original tags IF we used IDs
                if use_ids:
                    for p_id, original_tag_val in custom_phoneme_map.items():
                        chunk_text_to_yield = chunk_text_to_yield.replace(p_id, original_tag_val)

                chunk_count += 1
                logger.info(f"Yielding final text chunk {chunk_count} for part: '{chunk_text_to_yield[:50]}...' ({current_token_count} tokens)")
                yield chunk_text_to_yield, current_chunk_tokens, None

        # --- Handle Pause Part ---
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
                 # Treat as text if parsing fails? For now, just log and skip.

    # --- End of parts loop ---
    total_time = time.time() - start_time
    logger.info(
        f"Split completed in {total_time * 1000:.2f}ms, produced {chunk_count} chunks (including pauses)"
    )