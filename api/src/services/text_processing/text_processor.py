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
        # Normalize step is usually done before smart_split, but phonemize itself might do basic norm
        # Note: normalize=False is passed to phonemize because normalization happens earlier in smart_split now
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

    # Process the whole text as one chunk (includes normalization if applicable)
    # Note: Normalization/phonemization is handled inside process_text_chunk based on skip_phonemize flag
    # However, smart_split handles normalization *before* calling this, so it might be redundant here.
    # Keep simple for now.
    return process_text_chunk(text, language, skip_phonemize=False)


def get_sentence_info(
    text: str, custom_phonemes_list: Dict[str, str]
) -> List[Tuple[str, List[int], int]]:
    """
    Processes text (potentially containing custom phoneme IDs) into sentences
    and tokenizes each sentence after restoring the actual phonemes for tokenization.
    Returns info including the original sentence format (with IDs or tags).

    Args:
        text: The input text, potentially containing custom phoneme IDs like </|custom_phonemes_0|/>.
        custom_phonemes_list: A dictionary mapping custom phoneme IDs to their original IPA string representations
                              (extracted from the [/ipa/] part of the tag).

    Returns:
        A list of tuples, where each tuple contains:
        (original sentence text with IDs/tags, list of tokens, token count).
    """
    # Split sentences, keeping delimiters and newlines
    sentences = re.findall(r'[^.!?\n]+(?:[.!?]+|\n+)?|\n+', text)

    results = []
    restored_phoneme_keys = list(custom_phonemes_list.keys()) # List of phoneme IDs to restore

    for original_sentence_with_ids in sentences:
        sentence_text_part = original_sentence_with_ids.rstrip('\n')

        if not sentence_text_part.strip():
            if '\n' in original_sentence_with_ids:
                results.append(("\n", [], 0)) # Represent pure newline/blank lines
            continue

        # Prepare text for tokenization: Restore actual phonemes from IDs
        sentence_to_tokenize = sentence_text_part
        restored_count = 0
        if custom_phonemes_list:
            for ph_id in restored_phoneme_keys:
                if ph_id in sentence_to_tokenize:
                    # Ensure the value in custom_phonemes_list is the IPA phoneme string
                    ipa_phoneme_string = custom_phonemes_list.get(ph_id, "")
                    # Extract IPA string like /ipa/ from original tag like [word](/ipa/)
                    match = re.search(r'\(\/([^\/)]+)\/\)', ipa_phoneme_string)
                    if match:
                        ipa_phoneme_string = match.group(1)
                    else:
                         logger.warning(f"Could not extract IPA phonemes from stored tag for ID {ph_id}: {ipa_phoneme_string}. Replacing ID with empty string.")
                         ipa_phoneme_string = "" # Fallback

                    sentence_to_tokenize = sentence_to_tokenize.replace(ph_id, ipa_phoneme_string)
                    restored_count += 1

            if restored_count > 0:
                logger.debug(f"Restored {restored_count} custom phonemes for tokenization in: '{sentence_text_part[:30]}...'")


        # Tokenize the text part containing restored phonemes (or original text if no IDs)
        # process_text_chunk handles phonemization if needed (skip_phonemize=False default)
        # Pass skip_phonemize=True because we have already handled custom phonemes and normalization
        try:
            tokens = process_text_chunk(sentence_to_tokenize, skip_phonemize=True)
        except Exception as e:
            logger.error(f"Tokenization failed for sentence part '{sentence_to_tokenize[:50]}...': {e}")
            tokens = []

        # Store the result: Use the original sentence format (which might contain IDs)
        results.append((original_sentence_with_ids, tokens, len(tokens)))

    return results


def handle_custom_phonemes(s: re.Match[str], phenomes_list: Dict[str, str]) -> str:
    """Stores the original tag and returns a unique ID."""
    original_tag = s.group(0).strip() # e.g., "[word](/ipa/)"
    latest_id = f"</|custom_phonemes_{len(phenomes_list)}|/>"
    # Store the full original tag, get_sentence_info will extract IPA later if needed
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
        Otherwise, it's a text chunk containing original formatting (incl. restored custom phoneme tags).
    """
    start_time = time.time()
    chunk_count = 0
    logger.info(f"Starting smart split for {len(text)} chars, max_tokens={max_tokens}, lang_code={lang_code}")

    # --- Step 1: Split by Pause Tags FIRST ---
    # This operates on the raw input text
    parts = PAUSE_TAG_PATTERN.split(text)
    logger.debug(f"Split raw text into {len(parts)} parts by pause tags.")

    # --- Determine if normalization and ID replacement are needed ---
    apply_normalization = (
        settings.advanced_text_normalization
        and normalization_options.normalize
        and lang_code in ["a", "b", "en-us", "en-gb"] # Normalization only for English
    )
    use_ids = (
        (settings.enable_custom_phoneme_ids if hasattr(settings, 'enable_custom_phoneme_ids') else apply_normalization)
        and apply_normalization # Still require normalization to be enabled
    )
    logger.debug(f"Normalization active: {apply_normalization}. Using ID replacement: {use_ids}")

    part_idx = 0
    while part_idx < len(parts):
        text_part_raw = parts[part_idx] # This part is raw text
        part_idx += 1

        processed_text_part = text_part_raw # Start with raw part for this iteration
        custom_phoneme_map_part = {} # Reset map for each part

        # --- Process Text Part (Apply ID replacement and Normalization here) ---
        if text_part_raw: # Only process if the part is not empty string
            # --- Step 1.1: Optional ID Replacement on the part ---
            if use_ids:
                processed_text_part = CUSTOM_PHONEMES.sub(
                    lambda s: handle_custom_phonemes(s, custom_phoneme_map_part), processed_text_part
                )
                if custom_phoneme_map_part:
                    logger.debug(f"Found and replaced custom phonemes with IDs in part: {custom_phoneme_map_part}")

            # --- Step 1.2: Optional Normalization on the part ---
            if apply_normalization:
                processed_text_part = normalize_text(processed_text_part, normalization_options)
                logger.debug("Applied text normalization to part.")

            # --- Step 1.3: Get sentence info from the processed part ---
            # Pass the potentially ID-replaced and normalized text part
            try:
                sentences = get_sentence_info(processed_text_part, custom_phoneme_map_part)
            except Exception as e:
                logger.error(f"get_sentence_info failed for processed part '{processed_text_part[:50]}...': {e}", exc_info=True)
                # Log the error and continue to the next part, checking for pauses
                # Check for pause part *after* skipping the problematic text part processing
                if part_idx < len(parts):
                    duration_str = parts[part_idx]
                    # Check if it looks like a duration before trying to parse
                    if re.fullmatch(r"\d+(?:\.\d+)?", duration_str): # Check if it's a number string
                        part_idx += 1 # Consume the duration string
                        try:
                            duration = float(duration_str)
                            if duration > 0:
                                chunk_count += 1
                                logger.info(f"Yielding pause chunk {chunk_count} after text processing error: {duration}s")
                                yield "", [], duration
                        except (ValueError, TypeError):
                            logger.warning(f"Could not parse potential pause duration after text error: {duration_str}")
                    else:
                         logger.debug(f"Part after text error was not a pause duration: '{duration_str[:50]}...'")
                continue # Go to next iteration of while loop


            current_chunk_texts_restored = [] # Store final, restored sentence texts for the current chunk
            current_chunk_tokens = []
            current_token_count = 0

            for sentence_text_processed_format, sentence_tokens, sentence_token_count in sentences:
                # sentence_text_processed_format has IDs if use_ids was true for this part.
                # Restore original custom phoneme tags *before* accumulating/yielding
                text_to_yield_or_accumulate = sentence_text_processed_format
                if use_ids and custom_phoneme_map_part:
                    for p_id, original_tag_val in custom_phoneme_map_part.items():
                         # Replace ID with the original tag like [word](/ipa/)
                        text_to_yield_or_accumulate = text_to_yield_or_accumulate.replace(p_id, original_tag_val)

                # --- Chunking Logic (uses restored text) ---
                # Condition 1: Current sentence alone exceeds max tokens
                if sentence_token_count > max_tokens:
                    logger.warning(f"Single sentence exceeds max_tokens ({sentence_token_count} > {max_tokens}): '{text_to_yield_or_accumulate[:50]}...'")
                    # Yield any existing chunk first
                    if current_chunk_texts_restored:
                        chunk_text_final = " ".join(current_chunk_texts_restored)
                        chunk_count += 1
                        logger.info(f"Yielding text chunk {chunk_count} (before oversized): '{chunk_text_final[:50]}...' ({current_token_count} tokens)")
                        yield chunk_text_final, current_chunk_tokens, None
                        current_chunk_texts_restored = []
                        current_chunk_tokens = []
                        current_token_count = 0

                    # Yield the oversized sentence (already restored)
                    chunk_count += 1
                    logger.info(f"Yielding oversized text chunk {chunk_count}: '{text_to_yield_or_accumulate[:50]}...' ({sentence_token_count} tokens)")
                    yield text_to_yield_or_accumulate, sentence_tokens, None
                    continue # Move to the next sentence

                # Condition 2: Adding the current sentence would exceed max_tokens
                elif current_token_count + sentence_token_count > max_tokens:
                    # Yield the current chunk first
                    if current_chunk_texts_restored:
                        chunk_text_final = " ".join(current_chunk_texts_restored)
                        chunk_count += 1
                        logger.info(f"Yielding text chunk {chunk_count} (max_tokens limit): '{chunk_text_final[:50]}...' ({current_token_count} tokens)")
                        yield chunk_text_final, current_chunk_tokens, None
                    # Start a new chunk with the current sentence (restored)
                    current_chunk_texts_restored = [text_to_yield_or_accumulate]
                    current_chunk_tokens = sentence_tokens
                    current_token_count = sentence_token_count

                # Condition 3: Adding exceeds target_max_tokens when already above target_min_tokens
                elif (current_token_count >= settings.target_min_tokens and
                      current_token_count + sentence_token_count > settings.target_max_tokens):
                    # Yield the current chunk
                    chunk_text_final = " ".join(current_chunk_texts_restored)
                    chunk_count += 1
                    logger.info(f"Yielding text chunk {chunk_count} (target_max limit): '{chunk_text_final[:50]}...' ({current_token_count} tokens)")
                    yield chunk_text_final, current_chunk_tokens, None
                    # Start a new chunk (restored)
                    current_chunk_texts_restored = [text_to_yield_or_accumulate]
                    current_chunk_tokens = sentence_tokens
                    current_token_count = sentence_token_count

                # Condition 4: Add sentence to current chunk
                else:
                    current_chunk_texts_restored.append(text_to_yield_or_accumulate) # Add restored text
                    current_chunk_tokens.extend(sentence_tokens)
                    current_token_count += sentence_token_count

            # --- End of sentence loop for this text part ---
            # Yield any remaining accumulated chunk for this text part
            if current_chunk_texts_restored:
                chunk_text_final = " ".join(current_chunk_texts_restored)
                chunk_count += 1
                logger.info(f"Yielding final text chunk {chunk_count} for part: '{chunk_text_final[:50]}...' ({current_token_count} tokens)")
                yield chunk_text_final, current_chunk_tokens, None

        # --- Handle Pause Part ---
        # Check if the next part *is* a pause duration string
        if part_idx < len(parts):
            duration_str = parts[part_idx]
            # Check if it looks like a valid number string captured by the regex group
            if re.fullmatch(r"\d+(?:\.\d+)?", duration_str):
                part_idx += 1 # Consume the duration string as it's been processed
                try:
                    duration = float(duration_str)
                    if duration > 0:
                        chunk_count += 1
                        logger.info(f"Yielding pause chunk {chunk_count}: {duration}s")
                        yield "", [], duration  # Yield pause chunk
                except (ValueError, TypeError):
                    # This case should be rare if re.fullmatch passed, but handle anyway
                    logger.warning(f"Could not parse valid-looking pause duration: {duration_str}")
                    # If parsing fails unexpectedly, we've already consumed part_idx.
                    # The next loop iteration will process the part *after* the failed duration.
            # else:
                # This part wasn't a captured duration string; it's the next text segment.
                # It will be handled correctly in the next iteration when part_idx naturally points to it.
                # logger.debug(f"Part {part_idx} ('{duration_str[:20]}...') is text, not pause duration.")

    # --- End of parts loop ---
    total_time = time.time() - start_time
    logger.info(
        f"Split completed in {total_time * 1000:.2f}ms, produced {chunk_count} chunks (including pauses)"
    )