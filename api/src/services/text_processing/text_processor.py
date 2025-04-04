"""Unified text processing for TTS with smart chunking."""

import re
import time
from typing import AsyncGenerator, Dict, List, Tuple

from loguru import logger

from ...core.config import settings
from .normalizer import normalize_text
from .phonemizer import phonemize
from .vocabulary import tokenize
from ...structures.schemas import NormalizationOptions

# Pre-compiled regex patterns for performance
CUSTOM_PHONEMES = re.compile(r"(\[([^\]]|\n)*?\])(\(\/([^\/)]|\n)*?\/\))")
# Matching: [silent 1s], [silent 0.5s], [silent .5s]
SILENCE_TAG = re.compile(r"\[silent (\d*\.?\d+)s\]")

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
        phonemes = phonemize(
            text, language, normalize=False
        )  # Already normalized
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


def get_sentence_info(text: str, custom_phenomes_list: Dict[str, str]) -> List[Tuple[str, List[int], int]]:
    """
    Process all sentences and return info.

    Possible List Values:
    - (sentence, tokens, token_count)
    - (silence_tag, [], 0)
    """

    sentences = re.split(r"([.!?;:])(?=\s|$)", text)
    
    results = []
    for i in range(0, len(sentences), 2):
        sentence = sentences[i].strip()
        for key in list(custom_phenomes_list.keys()):
            if key in sentence:
                sentence = sentence.replace(key, custom_phenomes_list[key])
                del custom_phenomes_list[key]
                
        # Handle silence tags
        # Eg: "This is a test sentence, [silent](/1s/) with silence for one second."
        while match := SILENCE_TAG.search(sentence):
            match_prefix = sentence[:match.start()] # `This is a test sentence, `
            match_text = match.group(0)             # `[silent](/1s/)`
            match_suffix = sentence[match.end():]   # ` with silence for one second.`
            if match_prefix.strip():
                tokens = process_text_chunk(match_prefix.strip())
                results.append((match_prefix, tokens, len(tokens)))

            # Insert silence tag with empty tokens
            results.append((match_text, [], 0))
            sentence = match_suffix 
            
        punct = sentences[i + 1] if i + 1 < len(sentences) else ""

        if not sentence:
            continue

        full = sentence + punct
        tokens = process_text_chunk(full)
        results.append((full, tokens, len(tokens)))

    return results

def handle_custom_phonemes(s: re.Match[str], phenomes_list: Dict[str,str]) -> str:
    """
    Replace [text](/phonemes/) with a <|custom_phonemes_X|/> tag to avoid being normalized.
    Silence tags like [silence 1.5s] are replaced too.
    """
    latest_id = f"</|custom_phonemes_{len(phenomes_list)}|/>"
    phenomes_list[latest_id] = s.group(0).strip()
    return latest_id

async def smart_split(
    text: str, 
    max_tokens: int = settings.absolute_max_tokens,
    lang_code: str = "a",
    normalization_options: NormalizationOptions = NormalizationOptions()
) -> AsyncGenerator[Tuple[str, List[int]], None]:
    """Build optimal chunks targeting 300-400 tokens, never exceeding max_tokens."""
    start_time = time.time()
    chunk_count = 0
    logger.info(f"Starting smart split for {len(text)} chars")

    custom_phoneme_list = {}

    text = SILENCE_TAG.sub(lambda s: handle_custom_phonemes(s, custom_phoneme_list), text)

    # Normalize text
    if settings.advanced_text_normalization and normalization_options.normalize:
        if lang_code in ["a","b","en-us","en-gb"]:
            text = CUSTOM_PHONEMES.sub(lambda s: handle_custom_phonemes(s, custom_phoneme_list), text)
            text=normalize_text(text,normalization_options)
        else:
            logger.info("Skipping text normalization as it is only supported for english")

    # Process all sentences
    sentences = get_sentence_info(text, custom_phoneme_list)

    current_chunk = []
    current_tokens = []
    current_count = 0

    for sentence, tokens, count in sentences:
        # Handle silence tags
        if SILENCE_TAG.match(sentence):
            # Yield any existing chunk if present.
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk_count += 1
                logger.debug(
                    f"Yielding chunk {chunk_count} before silence tag: '{chunk_text[:50]}{'...' if len(text) > 50 else ''}' ({current_count} tokens)"
                )
                yield chunk_text, current_tokens
                current_chunk = []
                current_tokens = []
                current_count = 0
            
            # Silent tags is not sent to Kokoro, we we don't increment `chunk_count`
            logger.debug(f"Yielding silence tag: '{sentence}'")
            yield sentence, []
            continue

        # Handle sentences that exceed max tokens
        if count > max_tokens:
            # Yield current chunk if any
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk_count += 1
                logger.debug(
                    f"Yielding chunk {chunk_count}: '{chunk_text[:50]}{'...' if len(text) > 50 else ''}' ({current_count} tokens)"
                )
                yield chunk_text, current_tokens
                current_chunk = []
                current_tokens = []
                current_count = 0

            # Split long sentence on commas
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
                        chunk_text = " ".join(clause_chunk)
                        chunk_count += 1
                        logger.debug(
                            f"Yielding clause chunk {chunk_count}: '{chunk_text[:50]}{'...' if len(text) > 50 else ''}' ({clause_count} tokens)"
                        )
                        yield chunk_text, clause_tokens
                    clause_chunk = [full_clause]
                    clause_tokens = tokens
                    clause_count = count

            # Don't forget last clause chunk
            if clause_chunk:
                chunk_text = " ".join(clause_chunk)
                chunk_count += 1
                logger.debug(
                    f"Yielding final clause chunk {chunk_count}: '{chunk_text[:50]}{'...' if len(text) > 50 else ''}' ({clause_count} tokens)"
                )
                yield chunk_text, clause_tokens

        # Regular sentence handling
        elif (
            current_count >= settings.target_min_tokens
            and current_count + count > settings.target_max_tokens
        ):
            # If we have a good sized chunk and adding next sentence exceeds target,
            # yield current chunk and start new one
            chunk_text = " ".join(current_chunk)
            chunk_count += 1
            logger.info(
                f"Yielding chunk {chunk_count}: '{chunk_text[:50]}{'...' if len(text) > 50 else ''}' ({current_count} tokens)"
            )
            yield chunk_text, current_tokens
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
                chunk_text = " ".join(current_chunk)
                chunk_count += 1
                logger.info(
                    f"Yielding chunk {chunk_count}: '{chunk_text[:50]}{'...' if len(text) > 50 else ''}' ({current_count} tokens)"
                )
                yield chunk_text, current_tokens
            current_chunk = [sentence]
            current_tokens = tokens
            current_count = count

    # Don't forget the last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunk_count += 1
        logger.info(
            f"Yielding final chunk {chunk_count}: '{chunk_text[:50]}{'...' if len(text) > 50 else ''}' ({current_count} tokens)"
        )
        yield chunk_text, current_tokens

    total_time = time.time() - start_time
    logger.info(
        f"Split completed in {total_time * 1000:.2f}ms, produced {chunk_count} chunks"
    )
