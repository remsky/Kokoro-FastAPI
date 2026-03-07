"""Audio conversion service with proper streaming support"""

from io import BytesIO
from typing import Optional

import av
import numpy as np
from loguru import logger


class StreamingAudioWriter:
    """Handles streaming audio format conversions"""

    def __init__(self, format: str, sample_rate: int, channels: int = 1):
        self.format = format.lower()
        self.sample_rate = sample_rate
        self.channels = channels
        self.bytes_written = 0
        self.pts = 0
        # Opus is muxed on a 48 kHz clock even when the source PCM is 24 kHz.
        self.codec_sample_rate = 48000 if self.format == "opus" else self.sample_rate

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
                    rate=self.codec_sample_rate,
                    layout="mono" if self.channels == 1 else "stereo",
                )
                # Set bit_rate only for codecs where it's applicable and useful
                if self.format in ['mp3', 'aac', 'opus']:
                    self.stream.bit_rate = 128000

                if self.format == "opus":
                    # Resample the model's 24 kHz PCM into the codec clock expected by Opus.
                    self.resampler = av.AudioResampler(
                        format="s16",
                        layout="mono" if self.channels == 1 else "stereo",
                        rate=self.codec_sample_rate,
                    )
        else:
            raise ValueError(f"Unsupported format: {self.format}") # Use self.format here

    def close(self):
        if hasattr(self, "container"):
            self.container.close()
            del self.container

        if hasattr(self, "output_buffer"):
            self.output_buffer.close()
            del self.output_buffer

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

                # The Opus/Ogg muxer keeps the final pages in memory until close().
                # Reading the buffer before close truncates the tail of the stream.
                self.container.close()
                del self.container

                # Get the final bytes from the buffer *before* closing it
                data = self.output_buffer.getvalue()
                self.output_buffer.close()
                del self.output_buffer
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

            frames_to_encode = [frame]
            if self.format == "opus":
                resampled = self.resampler.resample(frame)
                if resampled is None:
                    frames_to_encode = []
                elif isinstance(resampled, list):
                    frames_to_encode = resampled
                else:
                    frames_to_encode = [resampled]

            for encode_frame in frames_to_encode:
                encode_frame.pts = self.pts
                self.pts += encode_frame.samples

                packets = self.stream.encode(encode_frame)
                for packet in packets:
                    self.container.mux(packet)

            data = self.output_buffer.getvalue()
            self.output_buffer.seek(0)
            self.output_buffer.truncate(0)
            return data
