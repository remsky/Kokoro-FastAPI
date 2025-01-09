"""Audio conversion service"""

from io import BytesIO

import numpy as np
import soundfile as sf
import scipy.io.wavfile as wavfile
from loguru import logger
from ..core.config import settings

class AudioNormalizer:
    """Handles audio normalization state for a single stream"""
    def __init__(self):
        self.int16_max = np.iinfo(np.int16).max
        self.chunk_trim_ms = settings.gap_trim_ms
        self.sample_rate = 24000  # Sample rate of the audio
        self.samples_to_trim = int(self.chunk_trim_ms * self.sample_rate / 1000)
        self.samples_to_pad_start= int(50 * self.sample_rate / 1000)
        self.samples_to_pad_end= max(int(settings.dynamic_gap_trim_padding_ms * self.sample_rate / 1000) - self.samples_to_pad_start, 0)
        
    def find_first_last_non_silent(self,audio_data: np.ndarray, silence_threshold_db: int = -45) -> tuple[int, int]:
        """
        Finds the indices of the first and last non-silent samples in audio data.
        """

        # Convert dBFS threshold to amplitude
        amplitude_threshold = self.int16_max * (10 ** (silence_threshold_db / 20))
 
        # Find all samples above the silence threshold
        non_silent_index_start, non_silent_index_end = None,None 
        
        for X in range(0,len(audio_data)):
            if audio_data[X] > amplitude_threshold:
                non_silent_index_start=X
                break
            
        for X in range(len(audio_data) - 1, -1, -1):
            if audio_data[X] > amplitude_threshold:
                non_silent_index_end=X
                break

        # Handle the case where the entire audio is silent
        if non_silent_index_start == None or non_silent_index_end == None:
            return 0, len(audio_data)

        return max(non_silent_index_start - self.samples_to_pad_start,0), min(non_silent_index_end + self.samples_to_pad_end,len(audio_data))

    def normalize(self, audio_data: np.ndarray, is_last_chunk: bool = False) -> np.ndarray:
        """Normalize audio data to int16 range and trim chunk boundaries"""
        # Convert to float32 if not already
        audio_float = audio_data.astype(np.float32)
        
        # Normalize to [-1, 1] range first
        if np.max(np.abs(audio_float)) > 0:
            audio_float = audio_float / np.max(np.abs(audio_float))
        
        # Trim end of non-final chunks to reduce gaps
        if not is_last_chunk and len(audio_float) > self.samples_to_trim:
            audio_float = audio_float[:-self.samples_to_trim]
            
        audio_int=(audio_float * self.int16_max).astype(np.int16)

        start_index,end_index=self.find_first_last_non_silent(audio_int)
 
        # Scale to int16 range
        return audio_int[start_index:end_index]

class AudioService:
    """Service for audio format conversions"""
    
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
        }
    }
    
    @staticmethod
    def convert_audio(
        audio_data: np.ndarray, 
        sample_rate: int, 
        output_format: str, 
        is_first_chunk: bool = True,
        is_last_chunk: bool = False,
        normalizer: AudioNormalizer = None,
        format_settings: dict = None,
        stream: bool = True
    ) -> bytes:
        """Convert audio data to specified format

        Args:
            audio_data: Numpy array of audio samples
            sample_rate: Sample rate of the audio
            output_format: Target format (wav, mp3, opus, flac, pcm)
            is_first_chunk: Whether this is the first chunk of a stream
            normalizer: Optional AudioNormalizer instance for consistent normalization across chunks
            format_settings: Optional dict of format-specific settings to override defaults
                Example: {
                    "mp3": {
                        "bitrate_mode": "VARIABLE",
                        "compression_level": 0.8
                    }
                }
                Default settings balance speed and compression:
                optimized for localhost @ 0.0
                - MP3: constant bitrate, no compression (0.0)
                - OPUS: no compression (0.0)
                - FLAC: no compression (0.0)

        Returns:
            Bytes of the converted audio
        """
        buffer = BytesIO()

        try:
            # Always normalize audio to ensure proper amplitude scaling
            if stream:
                if normalizer is None:
                    normalizer = AudioNormalizer()
                normalized_audio = normalizer.normalize(audio_data, is_last_chunk=is_last_chunk)
            else:
                normalized_audio = audio_data

            if output_format == "pcm":
                # Raw 16-bit PCM samples, no header
                buffer.write(normalized_audio.tobytes())
            elif output_format == "wav":
                if stream:
                    # Use soundfile for streaming to ensure proper headers
                    sf.write(buffer, normalized_audio, sample_rate, format="WAV", subtype='PCM_16')
                else:
                    # Trying scipy.io.wavfile for non-streaming WAV generation 
                    # seems faster than soundfile
                    # avoids overhead from header generation and PCM encoding
                    wavfile.write(buffer, sample_rate, normalized_audio)
            elif output_format == "mp3":
                # Use format settings or defaults
                settings = format_settings.get("mp3", {}) if format_settings else {}
                settings = {**AudioService.DEFAULT_SETTINGS["mp3"], **settings}
                sf.write(
                    buffer, normalized_audio, 
                    sample_rate, format="MP3",
                    **settings
                    )
                
            elif output_format == "opus":
                settings = format_settings.get("opus", {}) if format_settings else {}
                settings = {**AudioService.DEFAULT_SETTINGS["opus"], **settings}
                sf.write(buffer, normalized_audio, sample_rate, format="OGG", 
                        subtype="OPUS", **settings)
                
            elif output_format == "flac":
                if is_first_chunk:
                    logger.info("Starting FLAC stream...")
                settings = format_settings.get("flac", {}) if format_settings else {}
                settings = {**AudioService.DEFAULT_SETTINGS["flac"], **settings}
                sf.write(buffer, normalized_audio, sample_rate, format="FLAC",
                        subtype='PCM_16', **settings)
            else:
                if output_format == "aac":
                    raise ValueError(
                        "Format aac not supported. Supported formats are: wav, mp3, opus, flac, pcm."
                    )
                else:
                    raise ValueError(
                        f"Format {output_format} not supported. Supported formats are: wav, mp3, opus, flac, pcm."
                    )

            buffer.seek(0)
            return buffer.getvalue()

        except Exception as e:
            logger.error(f"Error converting audio to {output_format}: {str(e)}")
            raise ValueError(f"Failed to convert audio to {output_format}: {str(e)}")
