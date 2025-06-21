"""Audio conversion service with proper streaming support"""

import struct
from io import BytesIO
from typing import Optional

import av
import numpy as np
import soundfile as sf
from loguru import logger
from pydub import AudioSegment


class StreamingAudioWriter:
    """Handles streaming audio format conversions"""

    def __init__(self, format: str, sample_rate: int, channels: int = 1):
        self.format = format.lower()
        self.sample_rate = sample_rate
        self.channels = channels
        self.bytes_written = 0
        self.pts = 0

        codec_map = {
            "wav": "pcm_s16le",
            "mp3": "mp3",
            "opus": "libopus",
            "flac": "flac",
            "aac": "aac",
        }
        # Format-specific setup
        if self.format in ["wav", "flac", "mp3", "pcm", "aac", "opus"]:
            if self.format != "pcm":
                self.output_buffer = BytesIO()
                container_options = {}
                # Try disabling Xing VBR header for MP3 to fix iOS timeline reading issues
                if self.format == 'mp3':
                    # Disable Xing VBR header
                    container_options = {'write_xing': '0'}
                    logger.debug("Disabling Xing VBR header for MP3 encoding.")

                self.container = av.open(
                    self.output_buffer,
                    mode="w",
                    format=self.format if self.format != "aac" else "adts",
                    options=container_options # Pass options here
                )
                self.stream = self.container.add_stream(
                    codec_map[self.format],
                    rate=self.sample_rate,
                    layout="mono" if self.channels == 1 else "stereo",
                )
                # Set bit_rate only for codecs where it's applicable and useful
                if self.format in ['mp3', 'aac', 'opus']:
                    self.stream.bit_rate = 128000
        else:
            raise ValueError(f"Unsupported format: {self.format}") # Use self.format here

    def close(self):
        if hasattr(self, "container"):
            self.container.close()

        if hasattr(self, "output_buffer"):
            self.output_buffer.close()

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
                # Flush stream encoder
                packets = self.stream.encode(None)
                for packet in packets:
                    self.container.mux(packet)

                # Closing the container handles writing the trailer and finalizing the file.
                # No explicit flush method is available or needed here.
                logger.debug("Muxed final packets.")

                # Get the final bytes from the buffer *before* closing it
                data = self.output_buffer.getvalue()
                self.close() # Close container and buffer
                return data

        if audio_data is None or len(audio_data) == 0:
            return b""

        if self.format == "pcm":
            # Write raw bytes
            return audio_data.tobytes()
        else:
            frame = av.AudioFrame.from_ndarray(
                audio_data.reshape(1, -1),
                format="s16",
                layout="mono" if self.channels == 1 else "stereo",
            )
            frame.sample_rate = self.sample_rate

            frame.pts = self.pts
            self.pts += frame.samples

            packets = self.stream.encode(frame)
            for packet in packets:
                self.container.mux(packet)

            data = self.output_buffer.getvalue()
            self.output_buffer.seek(0)
            self.output_buffer.truncate(0)
            return data
