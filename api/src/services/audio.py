"""Audio conversion service"""

import math
from fractions import Fraction

import av
import numpy as np
from loguru import logger

from ..core.config import settings
from ..inference.base import AudioChunk
from .streaming_audio_writer import StreamingAudioWriter

# per-filter range on older ffmpeg builds, chained past it
ATEMPO_MIN, ATEMPO_MAX = 0.5, 2.0


def _atempo_factors(speed: float) -> list[float]:
    factors = []
    while speed > ATEMPO_MAX:
        factors.append(ATEMPO_MAX)
        speed /= ATEMPO_MAX
    while speed < ATEMPO_MIN:
        factors.append(ATEMPO_MIN)
        speed /= ATEMPO_MIN
    return factors + [speed]


class AudioFilterStream:
    """One sequential, request-owned filter graph; arrays are samples by channels."""

    def __init__(
        self, speed=1.0, sample_rate=24000, channels=1, normalize=False, gain=1.0
    ):
        if not math.isfinite(speed) or not 0.25 <= speed <= 4.0:
            raise ValueError("speed must be between 0.25 and 4.0")
        if sample_rate <= 0 or channels not in (1, 2):
            raise ValueError("positive sample rate and mono/stereo audio required")
        self.speed = speed
        self.normalize = normalize
        self.sample_rate = sample_rate
        self.channels = channels
        self.layout = "mono" if channels == 1 else "stereo"
        self.time_base = Fraction(1, sample_rate)
        self.graph = None
        self.input_samples = 0
        self.output_samples = 0
        self._pts = 0
        self._finished = False
        self._pending = self._empty()
        self._last_sample = 0.0
        if speed != 1.0 or normalize:
            self.graph = av.filter.Graph()
            nodes = [
                self.graph.add_abuffer(
                    format="flt",
                    sample_rate=sample_rate,
                    layout=self.layout,
                    time_base=self.time_base,
                )
            ]
            if speed != 1.0:
                nodes += [
                    self.graph.add("atempo", str(f)) for f in _atempo_factors(speed)
                ]
            if normalize:
                # Short lookahead, bounded gain and a noise gate for streaming speech.
                nodes.append(
                    self.graph.add("dynaudnorm", "f=40:g=5:p=0.9:m=4:r=0.12:t=0.005")
                )
                if gain != 1.0:
                    nodes.append(self.graph.add("volume", str(gain)))
                nodes.append(
                    self.graph.add("alimiter", "limit=0.95:level=false:latency=true")
                )
                nodes.append(
                    self.graph.add(
                        "aformat",
                        f"sample_fmts=flt:sample_rates={sample_rate}:channel_layouts={self.layout}",
                    )
                )
            nodes.append(self.graph.add("abuffersink"))
            for source, sink in zip(nodes, nodes[1:]):
                source.link_to(sink)
            self.graph.configure()
            self.graph.set_audio_frame_size(1024)

    def _empty(self):
        shape = (0,) if self.channels == 1 else (0, self.channels)
        return np.empty(shape, dtype=np.float32)

    def _push_frame(self, audio):
        # atempo sizes output from input frames; small feeds avoid chunk-sized latency.
        for start in range(0, len(audio), 1024):
            block = audio[start : start + 1024]
            frame = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(block.reshape(1, -1)),
                format="flt",
                layout=self.layout,
            )
            frame.sample_rate = self.sample_rate
            frame.time_base = self.time_base
            frame.pts = self._pts
            self._pts += len(block)
            self.graph.push(frame)

    def _drain(self):
        frames = [self._pending]
        while True:
            try:
                frame = self.graph.pull()
            except (av.BlockingIOError, av.EOFError):
                break
            values = frame.to_ndarray().reshape(-1, self.channels)
            frames.append(values[:, 0] if self.channels == 1 else values)
        available = np.concatenate(frames)
        remaining = max(round(self.input_samples / self.speed) - self.output_samples, 0)
        result = available[:remaining]
        self._pending = available[len(result) :].copy()
        self.output_samples += len(result)
        return result

    def push(self, audio):
        if self._finished:
            raise RuntimeError("audio filter is closed")
        audio = np.asarray(audio)
        expected = 1 if self.channels == 1 else 2
        if audio.ndim != expected or (
            expected == 2 and audio.shape[1] != self.channels
        ):
            raise ValueError("audio shape does not match channel layout")
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        else:
            audio = audio.astype(np.float32, copy=False)
        self.input_samples += len(audio)
        if not len(audio):
            return self._empty()
        self._last_sample = audio[-1].copy()
        if self.graph is None:
            self.output_samples += len(audio)
            return audio
        self._push_frame(audio)
        return self._drain()

    def flush(self):
        if self._finished:
            return self._empty()
        self._finished = True
        if self.graph is None or not self.input_samples:
            return self._empty()
        # Sacrifice silence only at EOF, never at model chunk boundaries.
        if self.speed != 1.0:
            padding = math.ceil(self.sample_rate * 0.2 * max(self.speed, 1.0))
            shape = (padding,) if self.channels == 1 else (padding, self.channels)
            fill = self._last_sample if self.normalize else 0.0
            self._push_frame(np.full(shape, fill, dtype=np.float32))
        self.graph.push(None)
        result = self._drain()
        if self.output_samples != round(self.input_samples / self.speed):
            raise RuntimeError("audio filter failed to flush the complete tail")
        return result

    def close(self):
        self._finished = True
        self.graph = None
        self._pending = self._empty()


def time_stretch(
    audio: np.ndarray, speed: float, sample_rate: int = 24000
) -> np.ndarray:
    """Convenience wrapper for callers with a complete mono clip."""
    if speed == 1.0 or len(audio) == 0:
        return audio
    processor = AudioFilterStream(speed, sample_rate)
    try:
        result = np.concatenate([processor.push(audio), processor.flush()])
        if audio.dtype == np.int16:
            return np.clip(np.rint(result * 32768), -32768, 32767).astype(np.int16)
        return result
    finally:
        processor.close()


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
            if abs(int(audio_data[X])) > amplitude_threshold:
                non_silent_index_start = X
                break

        for X in range(len(audio_data) - 1, -1, -1):
            if abs(int(audio_data[X])) > amplitude_threshold:
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

            chunk_data = b""
            if len(audio_chunk.audio) > 0:
                chunk_data = writer.write_chunk(audio_chunk.audio)

            if is_last_chunk:
                chunk_data += writer.write_chunk(finalize=True)

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
