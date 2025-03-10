"""Audio conversion service with proper streaming support"""

import struct
from io import BytesIO
from typing import Optional, Dict

import numpy as np
import soundfile as sf
from loguru import logger
from pydub import AudioSegment
import av

class StreamingAudioWriter:
    """Handles streaming audio format conversions"""

    def __init__(self, format: str, sample_rate: int, channels: int = 1):
        self.format = format.lower()
        self.sample_rate = sample_rate
        self.channels = channels
        self.bytes_written = 0
        self.pts = 0
        
        # Format-specific setup
        if self.format not in ["wav", "opus", "flac", "mp3", "aac", "pcm"]:
            raise ValueError(f"Unsupported format: {format}")
            
        # Codec mapping
        self.codec_map = {
            "wav": "pcm_s16le",
            "mp3": "mp3",
            "opus": "libopus",
            "flac": "flac", 
            "aac": "aac"
        }

    def _create_container(self):
        """Create a new container for each write operation"""
        if self.format == "pcm":
            return None, None
            
        buffer = BytesIO()
        container = av.open(buffer, mode="w", format=self.format)
        stream = container.add_stream(
            self.codec_map[self.format],
            sample_rate=self.sample_rate,
            layout='mono' if self.channels == 1 else 'stereo'
        )
        stream.bit_rate = 96000
        return container, buffer

    def write_chunk(
        self, audio_data: Optional[np.ndarray] = None, finalize: bool = False
    ) -> bytes:
        """Write a chunk of audio data and return bytes in the target format.

        Args:
            audio_data: Audio data to write, or None if finalizing
            finalize: Whether this is the final write to close the stream
        """
        # Handle PCM format separately as it doesn't use PyAV
        if self.format == "pcm":
            if finalize or audio_data is None or len(audio_data) == 0:
                return b""
            return audio_data.tobytes()
            
        # Handle empty input
        if not finalize and (audio_data is None or len(audio_data) == 0):
            return b""
            
        try:
            # Create a new container for this operation
            container, buffer = self._create_container()
            stream = container.streams[0]
            
            if finalize:
                # Just return empty bytes for finalize in the new design
                return b""
                
            # Create audio frame
            frame = av.AudioFrame.from_ndarray(
                audio_data.reshape(1, -1), 
                format='s16', 
                layout='mono' if self.channels == 1 else 'stereo'
            )
            frame.sample_rate = self.sample_rate
            frame.pts = self.pts
            self.pts += frame.samples
            
            # Encode the frame
            for packet in stream.encode(frame):
                container.mux(packet)
                
            # Flush any remaining packets
            for packet in stream.encode(None):
                container.mux(packet)
                
            # Close the container and get the data
            container.close()
            data = buffer.getvalue()
            
            return data
            
        except Exception as e:
            logger.error(f"Error encoding audio chunk: {e}")
            return b""

