from importlib.metadata import (
    PackageNotFoundError,
    version as _pkg_version,
)
from pathlib import Path

import torch
from dotenv import dotenv_values
from pydantic_settings import BaseSettings


def _read_version() -> str:
    version_file = Path(__file__).resolve().parents[3] / "VERSION"
    if version_file.exists():
        return version_file.read_text().strip()
    try:
        return _pkg_version("kokoro-fastapi")
    except PackageNotFoundError:
        return "0.0.0"


class Settings(BaseSettings):
    # API Settings
    api_title: str = "Kokoro TTS API"
    api_description: str = "API for text-to-speech generation using Kokoro"
    api_version: str = _read_version()
    host: str = "0.0.0.0"
    port: int = 8880

    # Application Settings
    default_voice: str = "af_heart"
    default_voice_code: str | None = (
        None  # If set, overrides the first letter of voice name, though api call param still takes precedence
    )
    use_gpu: bool = True  # Whether to use GPU acceleration if available
    device_type: str | None = (
        None  # Will be auto-detected if None, can be "cuda", "mps", or "cpu"
    )
    allow_local_voice_saving: bool = (
        False  # Whether to allow saving combined voices locally
    )
    allow_dev_unload: bool = False  # Whether to expose the POST /dev/unload endpoint
    enable_debug_endpoints: bool = (
        False  # Whether to expose /debug/* host and process introspection routes
    )
    enable_voice_tags: bool = True  # Kill switch for [voice:...] parsing and /dev/dialogue, for deployments proxying untrusted text
    enable_ssml: bool = True  # Kill switch for SSML translation, the /dev/ssml routes and ssml=true on the speech endpoints 403 when off

    # Container absolute paths
    model_dir: str = "/app/api/src/models"  # Absolute path in container
    voices_dir: str = "/app/api/src/voices/v1_0"  # Absolute path in container
    model_repo_id: str = "hexgrad/Kokoro-82M"  # default if model not present in model_dir; silences warnings

    # Audio Settings
    default_volume_multiplier: float = 1.0
    # Text Processing Settings
    target_min_tokens: int = 175  # Target minimum tokens per chunk
    target_max_tokens: int = 250  # Target maximum tokens per chunk
    absolute_max_tokens: int = 450  # Absolute maximum tokens per chunk
    ssml_max_depth: int = 10  # Deepest SSML element nesting translated, real documents sit at 2-5
    max_pause_duration_s: float = 60.0
    max_total_pause_s: float = 300.0  # Total silence one request may ask for
    max_input_length: int = 1_000_000  # Characters of text one request may submit
    advanced_text_normalization: bool = True  # Preproesses the text before misiki
    voice_weight_normalization: bool = (
        True  # Normalize the voice weights so they add up to 1
    )

    gap_trim_ms: int = (
        1  # Base amount to trim from streaming chunk ends in milliseconds
    )
    dynamic_gap_trim_padding_ms: int = 410  # Padding to add to dynamic gap trim
    dynamic_gap_trim_padding_char_multiplier: dict[str, float] = {
        ".": 1,
        "!": 0.9,
        "?": 1,
        ",": 0.8,
    }

    # Web Player Settings
    enable_web_player: bool = True  # Whether to serve the web player UI
    web_player_path: str = "web"  # Path to web player static files
    cors_origins: list[str] = ["*"]  # CORS origins for web player
    cors_enabled: bool = True  # Whether to enable CORS

    # Temp File Settings for WEB Ui
    temp_file_dir: str = "api/temp_files"  # Directory for temporary audio files (relative to project root)
    max_temp_dir_size_mb: int = 2048  # Maximum size of temp directory (2GB)
    max_temp_dir_age_hours: int = 1  # Remove temp files older than 1 hour
    max_temp_dir_count: int = 3  # Maximum number of temp files to keep

    class Config:
        env_file = ".env"
        extra = "ignore"  # a stale or unrelated key in a user's .env must not stop the server booting

    def get_device(self) -> str:
        """Get the appropriate device based on settings and availability"""
        if not self.use_gpu:
            return "cpu"

        if self.device_type:
            return self.device_type

        # Auto-detect device
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"


settings = Settings()


def unrecognized_env_file_keys() -> list[str]:
    """Keys in the env file that match no setting, ignored at load so the caller can warn about them"""
    env_file = Path(str(Settings.model_config.get("env_file") or ""))
    if not env_file.is_file():
        return []
    known = {name.lower() for name in Settings.model_fields}
    return sorted(k for k in dotenv_values(env_file) if k.lower() not in known)
