"""Temporary file writer for audio downloads"""

import os
import tempfile
from typing import List, Optional

import aiofiles
from fastapi import HTTPException
from loguru import logger

from ..core.config import settings


async def cleanup_temp_files() -> None:
    """Clean up old temp files"""
    try:
        if not await aiofiles.os.path.exists(settings.temp_file_dir):
            await aiofiles.os.makedirs(settings.temp_file_dir, exist_ok=True)
            return

        # Get all temp files with stats
        files = []
        total_size = 0

        # Use os.scandir for sync iteration, but aiofiles.os.stat for async stats
        for entry in os.scandir(settings.temp_file_dir):
            if entry.is_file():
                stat = await aiofiles.os.stat(entry.path)
                files.append((entry.path, stat.st_mtime, stat.st_size))
                total_size += stat.st_size

        # Sort by modification time (oldest first)
        files.sort(key=lambda x: x[1])

        # Remove files if:
        # 1. They're too old
        # 2. We have too many files
        # 3. Directory is too large
        current_time = (await aiofiles.os.stat(settings.temp_file_dir)).st_mtime
        max_age = settings.max_temp_dir_age_hours * 3600

        for path, mtime, size in files:
            should_delete = False

            # Check age
            if current_time - mtime > max_age:
                should_delete = True
                logger.info(f"Deleting old temp file: {path}")

            # Check count limit
            elif len(files) > settings.max_temp_dir_count:
                should_delete = True
                logger.info(f"Deleting excess temp file: {path}")

            # Check size limit
            elif total_size > settings.max_temp_dir_size_mb * 1024 * 1024:
                should_delete = True
                logger.info(f"Deleting to reduce directory size: {path}")

            if should_delete:
                try:
                    await aiofiles.os.remove(path)
                    total_size -= size
                    logger.info(f"Deleted temp file: {path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {path}: {e}")

    except Exception as e:
        logger.warning(f"Error during temp file cleanup: {e}")


class TempFileWriter:
    """Handles writing audio chunks to a temp file"""

    def __init__(self, format: str):
        """Initialize temp file writer

        Args:
            format: Audio format extension (mp3, wav, etc)
        """
        self.format = format
        self.temp_file = None
        self._finalized = False
        self._write_error = False  # Flag to track if we've had a write error

    async def __aenter__(self):
        """Async context manager entry"""
        try:
            # Clean up old files first
            await cleanup_temp_files()

            # Create temp file with proper extension
            await aiofiles.os.makedirs(settings.temp_file_dir, exist_ok=True)
            temp = tempfile.NamedTemporaryFile(
                dir=settings.temp_file_dir,
                delete=False,
                suffix=f".{self.format}",
                mode="wb",
            )
            self.temp_file = await aiofiles.open(temp.name, mode="wb")
            self.temp_path = temp.name
            temp.close()  # Close sync file, we'll use async version

            # Generate download path immediately
            self.download_path = f"/download/{os.path.basename(self.temp_path)}"
        except Exception as e:
            # Handle permission issues or other errors gracefully
            logger.error(f"Failed to create temp file: {e}")
            self._write_error = True
            # Set a placeholder path so the API can still function
            self.temp_path = f"unavailable_{self.format}"
            self.download_path = f"/download/{self.temp_path}"

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        try:
            if self.temp_file and not self._finalized:
                await self.temp_file.close()
                self._finalized = True
        except Exception as e:
            logger.error(f"Error closing temp file: {e}")
            self._write_error = True

    async def write(self, chunk: bytes) -> None:
        """Write a chunk of audio data

        Args:
            chunk: Audio data bytes to write
        """
        if self._finalized:
            raise RuntimeError("Cannot write to finalized temp file")

        # Skip writing if we've already encountered an error
        if self._write_error or not self.temp_file:
            return

        try:
            await self.temp_file.write(chunk)
            await self.temp_file.flush()
        except Exception as e:
            # Handle permission issues or other errors gracefully
            logger.error(f"Failed to write to temp file: {e}")
            self._write_error = True

    async def finalize(self) -> str:
        """Close temp file and return download path

        Returns:
            Path to use for downloading the temp file
        """
        if self._finalized:
            raise RuntimeError("Temp file already finalized")

        # Skip finalizing if we've already encountered an error
        if self._write_error or not self.temp_file:
            self._finalized = True
            return self.download_path

        try:
            await self.temp_file.close()
            self._finalized = True
        except Exception as e:
            # Handle permission issues or other errors gracefully
            logger.error(f"Failed to finalize temp file: {e}")
            self._write_error = True
            self._finalized = True

        return self.download_path
