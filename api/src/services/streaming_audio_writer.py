"""Audio conversion service with proper streaming support"""

import struct
from io import BytesIO
from typing import Optional

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
        self.pts=0

        codec_map = {"wav":"pcm_s16le","mp3":"mp3","opus":"libopus","flac":"flac", "aac":"aac"}
        # Format-specific setup
        if self.format in ["wav", "opus","flac","mp3","aac","pcm"]:
            if self.format != "pcm":
                self.output_buffer = BytesIO()
                self.container = av.open(self.output_buffer, mode="w", format=self.format)
                self.stream = self.container.add_stream(codec_map[self.format],sample_rate=self.sample_rate,layout='mono' if self.channels == 1 else 'stereo')
                self.stream.bit_rate = 96000
        else:
            raise ValueError(f"Unsupported format: {format}")

    def write_chunk(
        self, audio_data: Optional[np.ndarray] = None, finalize: bool = False
    ) -> bytes:
        """Write a chunk of audio data and return bytes in the target format.

        Args:
            audio_data: Audio data to write, or None if finalizing
            finalize: Whether this is the final write to close the stream
        """

        if finalize:
            if self.format != "pcm":
                # Flush encoder buffers
                for packet in self.stream.encode(None):
                    self.container.mux(packet)
                self.container.close()
                data = self.output_buffer.getvalue()
                self.output_buffer.seek(0)
                self.output_buffer.truncate(0)
                return data
            return b""

        if audio_data is None or len(audio_data) == 0:
            return b""

        if self.format == "pcm":
            return audio_data.tobytes()
        else:
            frame = av.AudioFrame.from_ndarray(audio_data.reshape(1, -1), format='s16', layout='mono' if self.channels == 1 else 'stereo')
            frame.sample_rate = self.sample_rate
            frame.pts = self.pts
            self.pts += frame.samples
            
            encoded_data = b""
            for packet in self.stream.encode(frame):
                self.container.mux(packet)
                # Get the encoded data from the buffer
                encoded_data = self.output_buffer.getvalue()
                # Clear the buffer for next write
                self.output_buffer.seek(0)
                self.output_buffer.truncate(0)
            
            return encoded_data

