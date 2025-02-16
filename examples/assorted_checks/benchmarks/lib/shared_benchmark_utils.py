"""Shared utilities specific to TTS benchmarking."""

import time
from typing import List, Tuple, Optional

import requests
import tiktoken

from .shared_utils import save_audio_file, get_audio_length

# Global tokenizer instance
enc = tiktoken.get_encoding("cl100k_base")


def get_text_for_tokens(text: str, num_tokens: int) -> str:
    """Get a slice of text that contains exactly num_tokens tokens.

    Args:
        text: Input text to slice
        num_tokens: Desired number of tokens

    Returns:
        str: Text slice containing exactly num_tokens tokens
    """
    tokens = enc.encode(text)
    if num_tokens > len(tokens):
        return text
    return enc.decode(tokens[:num_tokens])


def make_tts_request(
    text: str,
    output_dir: str = None,
    timeout: int = 1800,
    prefix: str = "",
    stream: bool = True,
) -> Tuple[Optional[float], Optional[float]]:
    """Make TTS request using OpenAI-compatible endpoint.

    Args:
        text: Input text to convert to speech
        output_dir: Directory to save audio files. If None, audio won't be saved.
        timeout: Request timeout in seconds
        prefix: Optional prefix for output filenames

    Returns:
        tuple: (processing_time, audio_length) in seconds, or (None, None) on error
    """
    try:
        start_time = time.time()
        if stream:
            # For streaming, we need to collect all chunks
            audio_chunks = []
            response = requests.post(
                "http://localhost:8880/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": text,
                    "voice": "af_heart",
                    "response_format": "wav",
                    "stream": True,
                },
                timeout=timeout,
                stream=True,
            )
            response.raise_for_status()

            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    audio_chunks.append(chunk)

            # Combine all chunks
            audio_data = b"".join(audio_chunks)
        else:
            response = requests.post(
                "http://localhost:8880/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": text,
                    "voice": "af_heart",
                    "response_format": "wav",
                    "stream": False,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            audio_data = response.content

        processing_time = round(time.time() - start_time, 2)
        # Calculate audio length from audio data
        audio_length = get_audio_length(audio_data)

        # Save the audio file if output_dir is provided
        if output_dir:
            token_count = len(enc.encode(text))
            output_file = save_audio_file(
                audio_data, f"chunk_{token_count}_tokens", output_dir
            )
            print(f"Saved audio to {output_file}")

        return processing_time, audio_length

    except requests.exceptions.RequestException as e:
        print(f"Error making request for text: {text[:50]}... Error: {str(e)}")
        return None, None
    except Exception as e:
        print(f"Error processing text: {text[:50]}... Error: {str(e)}")
        return None, None


def generate_token_sizes(
    max_tokens: int,
    dense_step: int = 100,
    dense_max: int = 1000,
    sparse_step: int = 1000,
) -> List[int]:
    """Generate token size ranges with dense sampling at start.

    Args:
        max_tokens: Maximum number of tokens to generate sizes up to
        dense_step: Step size for dense sampling range
        dense_max: Maximum value for dense sampling
        sparse_step: Step size for sparse sampling range

    Returns:
        list: Sorted list of token sizes
    """
    # Dense sampling at start
    dense_range = list(range(dense_step, dense_max + 1, dense_step))

    if max_tokens <= dense_max or sparse_step < dense_max:
        return sorted(dense_range)
    # Sparse sampling for larger sizes
    sparse_range = list(range(dense_max + sparse_step, max_tokens + 1, sparse_step))

    # Combine and deduplicate
    return sorted(list(set(dense_range + sparse_range)))
