"""Text chunking service"""

import re

from ...core.config import settings


def split_text(text: str, max_chunk=None):
    """Split text into chunks on natural pause points

    Args:
        text: Text to split into chunks
        max_chunk: Maximum chunk size (defaults to settings.max_chunk_size)
    """
    if max_chunk is None:
        max_chunk = settings.max_chunk_size

    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    text = text.strip()
    if not text:
        return

    # First split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # For medium-length sentences, split on punctuation
        if len(sentence) > max_chunk:  # Lower threshold for more consistent sizes
            # First try splitting on semicolons and colons
            parts = re.split(r"(?<=[;:])\s+", sentence)

            for part in parts:
                part = part.strip()
                if not part:
                    continue

                # If part is still long, split on commas
                if len(part) > max_chunk:
                    subparts = re.split(r"(?<=,)\s+", part)
                    for subpart in subparts:
                        subpart = subpart.strip()
                        if subpart:
                            yield subpart
                else:
                    yield part
        else:
            yield sentence
