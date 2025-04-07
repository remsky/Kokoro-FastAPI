"""Unified text processing for TTS with smart chunking."""

import re
import time
from typing import AsyncGenerator, Dict, List, Tuple

from loguru import logger

from ...core.config import settings
from ...structures.schemas import NormalizationOptions
from .normalizer import normalize_text
from .phonemizer import phonemize
from .vocabulary import tokenize

# Pre-compiled regex patterns for performance
CUSTOM_PHONEMES = re.compile(r"(\[([^\]]|\n)*?\])(\(\/([^\/)]|\n)*?\/\))")


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
        phonemes = phonemize(text, language, normalize=False)  # Already normalized
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
    text: str, custom_phenomes_list: Dict[str, str]
) -> List[Tuple[str, List[int], int]]:
    """Process all sentences and return info."""
    sentences = re.split(r"([.!?;:])(?=\s|$)", text)
    phoneme_length, min_value = len(custom_phenomes_list), 0

    results = []
    for i in range(0, len(sentences), 2):
        sentence = sentences[i].strip()
        for replaced in range(min_value, phoneme_length):
            current_id = f"</|custom_phonemes_{replaced}|/>"
            if current_id in sentence:
                sentence = sentence.replace(
                    current_id, custom_phenomes_list.pop(current_id)
                )
                min_value += 1

        punct = sentences[i + 1] if i + 1 < len(sentences) else ""

        if not sentence:
            continue

        # Check if the original text segment ended with newline(s) before punctuation
        original_segment = sentences[i]
        trailing_newlines = ""
        match = re.search(r"(\n+)$", original_segment)
        if match:
            trailing_newlines = match.group(1)

        full = sentence + punct + trailing_newlines # Append trailing newlines
        # Tokenize without the trailing newlines for accurate TTS processing
        tokens = process_text_chunk(sentence + punct)
        # Store the full text including newlines for later check
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
) -> AsyncGenerator[Tuple[str, List[int]], None]:
    """Build optimal chunks targeting 300-400 tokens, never exceeding max_tokens."""
    start_time = time.time()
    chunk_count = 0
    logger.info(f"Starting smart split for {len(text)} chars")

    custom_phoneme_list = {}

    # Normalize text
    if settings.advanced_text_normalization and normalization_options.normalize:
        print(lang_code)
        if lang_code in ["a", "b", "en-us", "en-gb"]:
            text = CUSTOM_PHONEMES.sub(
                lambda s: handle_custom_phonemes(s, custom_phoneme_list), text
            )
            text = normalize_text(text, normalization_options)
        else:
            logger.info(
                "Skipping text normalization as it is only supported for english"
            )

    # Process all sentences
    sentences = get_sentence_info(text, custom_phoneme_list)

    current_chunk = []
    current_tokens = []
    current_count = 0

    for sentence, tokens, count in sentences:
        # Handle sentences that exceed max tokens
        if count > max_tokens:
            # Yield current chunk if any
            if current_chunk:
                # Join with space, but preserve original trailing newline of the last sentence if present
                last_sentence_original = current_chunk[-1]
                chunk_text_joined = " ".join(current_chunk)
                if last_sentence_original.endswith("\n"):
                     chunk_text_joined += "\n" # Preserve the newline marker

                chunk_count += 1
                logger.debug(
                    f"Yielding text chunk {chunk_count}: '{chunk_text_joined[:50]}{'...' if len(chunk_text_joined) > 50 else ''}' ({current_count} tokens)"
                )
                yield chunk_text_joined, current_tokens, None # Pass the text with potential trailing newline
                current_chunk = []
                current_tokens = []
                current_count = 0

            # Split long sentence on commas (simple approach)
            # Keep original sentence text ('sentence' now includes potential trailing newline)
            clauses = re.split(r"([,])", sentence.rstrip('\n')) # Split text part only
            trailing_newline_in_sentence = "\n" if sentence.endswith("\n") else ""
            clause_chunk = [] # Stores original clause text including potential trailing newline
            clause_tokens = []
            clause_count = 0

            for j in range(0, len(clauses), 2):
                # clause = clauses[j].strip() # Don't strip here to preserve internal structure
                clause = clauses[j]
                comma = clauses[j + 1] if j + 1 < len(clauses) else ""

                if not clause.strip(): # Check if clause is just whitespace

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
                        # Join with space, preserve last clause's potential trailing newline
                        last_clause_original = clause_chunk[-1]
                        chunk_text_joined = " ".join(clause_chunk)
                        if last_clause_original.endswith("\n"):
                            chunk_text_joined += "\n"

                        chunk_count += 1
                        logger.debug(
                            f"Yielding clause chunk {chunk_count}: '{chunk_text_joined[:50]}{'...' if len(chunk_text_joined) > 50 else ''}' ({clause_count} tokens)"
                        )
                        yield chunk_text_joined, clause_tokens, None
                    # Start new clause chunk with original text
                    clause_chunk = [full_clause + (trailing_newline_in_sentence if j == len(clauses) - 2 else "")]
                    clause_tokens = clause_token_list
                    clause_count = clause_token_count

                # Don't forget last clause chunk
                if clause_chunk:
                    # Join with space, preserve last clause's potential trailing newline
                    last_clause_original = clause_chunk[-1]
                    chunk_text_joined = " ".join(clause_chunk)
                    # The trailing newline logic was added when creating the chunk above
                    #if last_clause_original.endswith("\n"):
                    #     chunk_text_joined += "\n"

                    chunk_count += 1
                    logger.debug(
                        f"Yielding final clause chunk {chunk_count}: '{chunk_text_joined[:50]}{'...' if len(chunk_text_joined) > 50 else ''}' ({clause_count} tokens)"
                    )
                    yield chunk_text_joined, clause_tokens, None
            current_count >= settings.target_min_tokens
            and current_count + count > settings.target_max_tokens
        ):
            # If we have a good sized chunk and adding next sentence exceeds target,
            # Yield current chunk and start new one
            last_sentence_original = current_chunk[-1]
            chunk_text_joined = " ".join(current_chunk)
            if last_sentence_original.endswith("\n"):
                chunk_text_joined += "\n"
            chunk_count += 1
            logger.info(
                f"Yielding text chunk {chunk_count}: '{chunk_text_joined[:50]}{'...' if len(chunk_text_joined) > 50 else ''}' ({current_count} tokens)"
            )
            yield chunk_text_joined, current_tokens, None
            current_chunk = [sentence] # sentence includes potential trailing newline
            current_tokens = tokens
            current_count = count
        elif current_count + count <= settings.target_max_tokens:
            # Keep building chunk
            current_chunk.append(sentence) # sentence includes potential trailing newline
            current_tokens.extend(tokens)
            current_count += count
        elif (
            current_count + count <= max_tokens
            and current_count < settings.target_min_tokens
        ):
             # Exceed target max only if below min size
            current_chunk.append(sentence) # sentence includes potential trailing newline
            current_tokens.extend(tokens)
            current_count += count
        else:
            # Yield current chunk and start new one
            if current_chunk:
                last_sentence_original = current_chunk[-1]
                chunk_text_joined = " ".join(current_chunk)
                if last_sentence_original.endswith("\n"):
                    chunk_text_joined += "\n"
                chunk_count += 1
                logger.info(
                    f"Yielding text chunk {chunk_count}: '{chunk_text_joined[:50]}{'...' if len(chunk_text_joined) > 50 else ''}' ({current_count} tokens)"
                )
                yield chunk_text_joined, current_tokens, None
            current_chunk = [sentence] # sentence includes potential trailing newline
            current_tokens = tokens
            current_count = count

    # Yield any remaining text chunk
    if current_chunk:
        last_sentence_original = current_chunk[-1]
        chunk_text_joined = " ".join(current_chunk)
        if last_sentence_original.endswith("\n"):
            chunk_text_joined += "\n"
        chunk_count += 1
        logger.info(
            f"Yielding final text chunk {chunk_count}: '{chunk_text_joined[:50]}{'...' if len(chunk_text_joined) > 50 else ''}' ({current_count} tokens)"
        )
        yield chunk_text_joined, current_tokens, None

    total_time = time.time() - start_time
    logger.info(
        f"Split completed in {total_time * 1000:.2f}ms, produced {chunk_count} chunks"
    )