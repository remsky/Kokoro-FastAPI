"""Audio conversion service"""

import math
import struct
import time
from io import BytesIO
from typing import Tuple

import numpy as np
import scipy.io.wavfile as wavfile
import soundfile as sf
from loguru import logger
from pydub import AudioSegment
from torch import norm

from ..core.config import settings
from ..inference.base import AudioChunk
from .streaming_audio_writer import StreamingAudioWriter


class AudioNormalizer:
    """Handles audio normalization state for a single stream"""

    def __init__(self):
        self.chunk_trim_ms = settings.gap_trim_ms
        self.sample_rate = 24000  # Sample rate of the audio
        self.samples_to_trim = int(self.chunk_trim_ms * self.sample_rate / 1000)
        self.samples_to_pad_start = int(50 * self.sample_rate / 1000)

    def find_first_last_non_silent(
        self,
        audio_data: np.ndarray,
        chunk_text: str,
        speed: float,
        silence_threshold_db: int = -45,
        is_last_chunk: bool = False,
    ) -> tuple[int, int]:
        """Finds the indices of the first and last non-silent samples in audio data.

        Args:
            audio_data: Input audio data as numpy array
            chunk_text: The text sent to the model to generate the resulting speech
            speed: The speaking speed of the voice
            silence_threshold_db: How quiet audio has to be to be conssidered silent
            is_last_chunk: Whether this is the last chunk

        Returns:
            A tuple with the start of the non silent portion and with the end of the non silent portion
        """

        pad_multiplier = 1
        split_character = chunk_text.strip()
        if len(split_character) > 0:
            split_character = split_character[-1]
            if split_character in settings.dynamic_gap_trim_padding_char_multiplier:
                pad_multiplier = settings.dynamic_gap_trim_padding_char_multiplier[
                    split_character
                ]

        if not is_last_chunk:
            samples_to_pad_end = max(
                int(
                    (
                        settings.dynamic_gap_trim_padding_ms
                        * self.sample_rate
                        * pad_multiplier
                    )
                    / 1000
                )
                - self.samples_to_pad_start,
                0,
            )
        else:
            samples_to_pad_end = self.samples_to_pad_start
        # Convert dBFS threshold to amplitude
        amplitude_threshold = np.iinfo(audio_data.dtype).max * (
            10 ** (silence_threshold_db / 20)
        )
        # Find the first samples above the silence threshold at the start and end of the audio
        non_silent_index_start, non_silent_index_end = None, None

        for X in range(0, len(audio_data)):
            if abs(audio_data[X]) > amplitude_threshold:
                non_silent_index_start = X
                break

        for X in range(len(audio_data) - 1, -1, -1):
            if abs(audio_data[X]) > amplitude_threshold:
                non_silent_index_end = X
                break

        # Handle the case where the entire audio is silent
        if non_silent_index_start == None or non_silent_index_end == None:
            return 0, len(audio_data)

        return max(non_silent_index_start - self.samples_to_pad_start, 0), min(
            non_silent_index_end + math.ceil(samples_to_pad_end / speed),
            len(audio_data),
        )

    def normalize(self, audio_data: np.ndarray) -> np.ndarray:
        """Convert audio data to int16 range

        Args:
            audio_data: Input audio data as numpy array
        Returns:
            Normalized audio data
        """
        if audio_data.dtype != np.int16:
            # Scale directly to int16 range with clipping
            return np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
        return audio_data


class AudioService:
    """Service for audio format conversions with streaming support"""

    # Supported formats
    SUPPORTED_FORMATS = {"wav", "mp3", "opus", "flac", "aac", "pcm"}

    # Default audio format settings balanced for speed and compression
    DEFAULT_SETTINGS = {
        "mp3": {
            "bitrate_mode": "CONSTANT",  # Faster than variable bitrate
            "compression_level": 0.0,  # Balanced compression
        },
        "opus": {
            "compression_level": 0.0,  # Good balance for speech
        },
        "flac": {
            "compression_level": 0.0,  # Light compression, still fast
        },
        "aac": {
            "bitrate": "192k",  # Default AAC bitrate
        },
    }

    @staticmethod
    async def convert_audio(
        audio_chunk: AudioChunk,
        output_format: str,
        writer: StreamingAudioWriter,
        speed: float = 1,
        chunk_text: str = "",
        is_last_chunk: bool = False,
        trim_audio: bool = True,
        normalizer: AudioNormalizer = None,
    ) -> AudioChunk:
        """Convert audio data to specified format with streaming support

        Args:
            audio_data: Numpy array of audio samples
            output_format: Target format (wav, mp3, ogg, pcm)
            writer: The StreamingAudioWriter to use
            speed: The speaking speed of the voice
            chunk_text: The text sent to the model to generate the resulting speech
            is_last_chunk: Whether this is the last chunk
            trim_audio: Whether audio should be trimmed
            normalizer: Optional AudioNormalizer instance for consistent normalization

        Returns:
            Bytes of the converted audio chunk
        """

        try:
            # Validate format
            if output_format not in AudioService.SUPPORTED_FORMATS:
                raise ValueError(f"Format {output_format} not supported")

            # Always normalize audio to ensure proper amplitude scaling
            if normalizer is None:
                normalizer = AudioNormalizer()

            audio_chunk.audio = normalizer.normalize(audio_chunk.audio)

            if trim_audio == True:
                audio_chunk = AudioService.trim_audio(
                    audio_chunk, chunk_text, speed, is_last_chunk, normalizer
                )

            # Write audio data first
            if len(audio_chunk.audio) > 0:
                chunk_data = writer.write_chunk(audio_chunk.audio)

            # Then finalize if this is the last chunk
            if is_last_chunk:
                final_data = writer.write_chunk(finalize=True)

                if final_data:
                    audio_chunk.output = final_data
                return audio_chunk

            if chunk_data:
                audio_chunk.output = chunk_data
            return audio_chunk

        except Exception as e:
            logger.error(f"Error converting audio stream to {output_format}: {str(e)}")
            raise ValueError(
                f"Failed to convert audio stream to {output_format}: {str(e)}"
            )

    @staticmethod
    def trim_audio(
        audio_chunk: AudioChunk,
        chunk_text: str = "",
        speed: float = 1,
        is_last_chunk: bool = False,
        normalizer: AudioNormalizer = None,
    ) -> AudioChunk:
        """Trim silence from start and end

        Args:
            audio_data: Input audio data as numpy array
            chunk_text: The text sent to the model to generate the resulting speech
            speed: The speaking speed of the voice
            is_last_chunk: Whether this is the last chunk
            normalizer: Optional AudioNormalizer instance for consistent normalization

        Returns:
            Trimmed audio data
        """
        if normalizer is None:
            normalizer = AudioNormalizer()

        audio_chunk.audio = normalizer.normalize(audio_chunk.audio)

        trimed_samples = 0
        # Trim start and end if enough samples
        if len(audio_chunk.audio) > (2 * normalizer.samples_to_trim):
            audio_chunk.audio = audio_chunk.audio[
                normalizer.samples_to_trim : -normalizer.samples_to_trim
            ]
            trimed_samples += normalizer.samples_to_trim

        # Find non silent portion and trim
        start_index, end_index = normalizer.find_first_last_non_silent(
            audio_chunk.audio, chunk_text, speed, is_last_chunk=is_last_chunk
        )

        audio_chunk.audio = audio_chunk.audio[start_index:end_index]
        trimed_samples += start_index

        if audio_chunk.word_timestamps is not None:
            for timestamp in audio_chunk.word_timestamps:
                timestamp.start_time -= trimed_samples / 24000
                timestamp.end_time -= trimed_samples / 24000
        return audio_chunk
