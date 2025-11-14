import torch
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Settings
    api_title: str = "ZipVoice TTS API"
    api_description: str = "API for zero-shot text-to-speech generation using ZipVoice"
    api_version: str = "2.0.0"
    host: str = "0.0.0.0"
    port: int = 8880

    # Application Settings
    output_dir: str = "output"
    output_dir_size_limit_mb: float = 500.0  # Maximum size of output directory in MB
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

    # Container absolute paths
    model_dir: str = "/app/api/src/models"  # Absolute path in container
    voices_dir: str = "/app/api/src/voices/v1_0"  # Absolute path in container

    # Audio Settings
    sample_rate: int = 24000
    default_volume_multiplier: float = 1.0
    # Text Processing Settings
    target_min_tokens: int = 175  # Target minimum tokens per chunk
    target_max_tokens: int = 250  # Target maximum tokens per chunk
    absolute_max_tokens: int = 450  # Absolute maximum tokens per chunk
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

    # TTS Backend Settings (ZipVoice-only repository)
    default_backend: str = "zipvoice"  # Default TTS backend
    enable_kokoro: bool = False  # DISABLED - This is a ZipVoice-only repository
    enable_zipvoice: bool = True  # Enable ZipVoice TTS backend

    # ZipVoice Core Settings
    zipvoice_model: str = "zipvoice"  # Model variant: zipvoice, zipvoice_distill, zipvoice_dialog, zipvoice_dialog_stereo
    zipvoice_num_steps: int = 8  # Inference steps (lower = faster, range: 1-32)
    zipvoice_cache_dir: str = "api/src/voices/zipvoice_prompts"  # Directory for voice prompt cache
    zipvoice_max_prompt_duration: float = 3.0  # Maximum duration for prompt_wav in seconds
    zipvoice_remove_long_silence: bool = True  # Remove long silences from output
    zipvoice_speed_multiplier: float = 1.0  # Default speed adjustment
    zipvoice_max_download_size_mb: float = 10.0  # Maximum size for voice prompt URL downloads
    zipvoice_allow_url_download: bool = True  # Allow downloading voice prompts from URLs
    zipvoice_allow_base64: bool = True  # Allow base64 encoded voice prompts

    # Auto-Transcription Settings (Whisper integration)
    enable_auto_transcription: bool = True  # Enable automatic transcription with Whisper
    whisper_model_size: str = "base"  # Whisper model: tiny, base, small, medium, large
    auto_transcribe_on_upload: bool = True  # Auto-transcribe when registering voices
    whisper_device: str | None = None  # Device for Whisper (None = auto-detect)

    # Optimization Settings
    enable_onnx: bool = False  # Enable ONNX optimized inference (faster)
    enable_tensorrt: bool = False  # Enable TensorRT optimization (fastest, requires GPU)
    onnx_cache_dir: str = "api/src/models/onnx_cache"  # ONNX model cache
    tensorrt_cache_dir: str = "api/src/models/tensorrt_cache"  # TensorRT engine cache

    # Smart Features
    enable_smart_tuning: bool = True  # Auto-tune parameters based on input
    enable_quality_detection: bool = True  # Detect and warn about low-quality prompts
    enable_intelligent_caching: bool = True  # Smart caching with prefetching
    quality_threshold: float = 0.7  # Minimum quality score for voice prompts (0-1)

    class Config:
        env_file = ".env"

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
