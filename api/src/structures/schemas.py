import re
from enum import Enum
from typing import Annotated, Dict, List, Literal, Mapping, Optional, Union

from pydantic import (
    AfterValidator,
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)


class TTSStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"  # For files removed by cleanup


# pace bounds shared by every speed field, the ssml translator and the final clamp
RATE_MIN, RATE_MAX = 0.25, 4.0
Rate = Annotated[float, Field(ge=RATE_MIN, le=RATE_MAX)]


# gain bounds shared by every volume_multiplier field
VOLUME_MIN, VOLUME_MAX = 0.0, 10.0
Volume = Annotated[float, Field(ge=VOLUME_MIN, le=VOLUME_MAX)]


def _known_lang_code(value: str) -> str:
    """Fail at the request boundary, not on KPipeline's assert deep inside inference."""
    # deferred import, keeps torch out of every schema import
    from kokoro.pipeline import LANG_CODES

    if value not in LANG_CODES:
        raise ValueError(
            f"unsupported lang_code {value!r}, expected one of {sorted(LANG_CODES)}"
        )
    return value


LangCode = Annotated[str, AfterValidator(_known_lang_code)]


def _within_input_limit(value: str) -> str:
    # deferred import, keeps torch out of every schema import
    from ..core.config import settings

    if len(value) > settings.max_input_length:
        raise ValueError(
            f"input is {len(value)} characters, the limit is {settings.max_input_length}"
        )
    return value


InputText = Annotated[str, AfterValidator(_within_input_limit)]


def clamp_rate(value: float) -> float:
    """Hold a computed pace inside the bounds the request fields already enforce."""
    return min(max(value, RATE_MIN), RATE_MAX)


# OpenAI-compatible schemas
class WordTimestamp(BaseModel):
    """Word-level timestamp information"""

    word: str = Field(..., description="The word or token")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    voice: Optional[str] = Field(
        None,
        description="Resolved voice that spoke the word, present only when allow_voice_tags is on",
    )

    @model_serializer(mode="wrap")
    def _omit_unset_voice(self, handler):
        # keeps responses byte-identical to before the field existed when tags are off
        data = handler(self)
        if data.get("voice") is None:
            data.pop("voice", None)
        return data


class CaptionedSpeechResponse(BaseModel):
    """Response schema for captioned speech endpoint"""

    audio: str = Field(..., description="The generated audio data encoded in base 64")
    audio_format: str = Field(..., description="The format of the output audio")
    timestamps: Optional[List[WordTimestamp]] = Field(
        ..., description="Word-level timestamps"
    )


class VoiceTuneRequest(BaseModel):
    """Request schema for POST /dev/voices/tune"""

    name: str = Field(
        ...,
        pattern=r"^[abefhijpz][fm]_[A-Za-z0-9_]{1,61}$",
        description="Voice name to save as, in the stock <language><gender>_name form (af_..., bm_...); the language letter picks the pipeline",
    )
    audio: str = Field(
        ..., description="Reference clip (wav, flac, ogg or mp3) encoded in base 64"
    )
    strength: float = Field(
        1.0,
        ge=0.0,
        le=2.0,
        description="How far the enrolled voice sits from the adapter's mean voice: 0 is the mean voice, 1 (default) the prediction as is, above 1 exaggerates it at a quality cost",
    )


class VoiceTuneResponse(BaseModel):
    """Response schema for POST /dev/voices/tune"""

    voice: str = Field(..., description="Saved voice name, usable in every speech route")
    adapter: str = Field(..., description="Tune adapter that enrolled the voice")
    speed: float = Field(
        ..., description="Enrolled pace relative to Kokoro's default for this voice"
    )
    f0_mean_st: float = Field(
        ..., description="Reference pitch mean in semitones re 100 Hz"
    )


class NormalizationOptions(BaseModel):
    """Options for the normalization system"""

    normalize: bool = Field(
        default=True,
        description="Normalizes input text to make it easier for the model to say",
    )
    unit_normalization: bool = Field(
        default=False, description="Transforms units like 10KB to 10 kilobytes"
    )
    url_normalization: bool = Field(
        default=True,
        description="Changes urls so they can be properly pronounced by kokoro",
    )
    email_normalization: bool = Field(
        default=True,
        description="Changes emails so they can be properly pronouced by kokoro",
    )
    optional_pluralization_normalization: bool = Field(
        default=True,
        description="Replaces (s) with s so some words get pronounced correctly",
    )
    phone_normalization: bool = Field(
        default=True,
        description="Changes phone numbers so they can be properly pronouced by kokoro",
    )
    replace_remaining_symbols: bool = Field(
        default=True,
        description="Replaces the remaining symbols after normalization with their words",
    )


# shared with text_processor's VOICE_TAG_PATTERN so a rendered [voice:...] round-trips as one tag
VOICE_NAME_BODY = r"[a-zA-Z0-9_][a-zA-Z0-9_+\-(). ]*"
TURN_VOICE_PATTERN = re.compile(VOICE_NAME_BODY)


class VoiceAlias(BaseModel):
    """Alias target that can carry a natural pace for the voice"""

    voice: str
    rate: Optional[Rate] = Field(
        default=None,
        description="Multiplier on the request speed applied whenever this alias speaks",
    )


# the shape voice_aliases takes everywhere it is passed around, Mapping so plain str dicts fit
AliasMap = Mapping[str, Union[str, VoiceAlias]]


def _reject_tag_breaking_aliases(aliases: Optional[AliasMap]) -> Optional[AliasMap]:
    if aliases:
        for target in aliases.values():
            mix = target if isinstance(target, str) else target.voice
            if not TURN_VOICE_PATTERN.fullmatch(mix.strip()):
                raise ValueError(
                    f"Voice alias '{mix}' contains characters that cannot appear in a voice name"
                )
    return aliases


class VoiceAliasesMixin(BaseModel):
    """Shared voice_aliases field for request models that accept them"""

    voice_aliases: Optional[AliasMap] = Field(
        default=None,
        description="Optional short names for voices, e.g. {'narrator': 'af_bella(2)+af_sky'}. A value may also be {'voice': ..., 'rate': 0.8} to give the alias a natural pace. Usable wherever a voice name is given and in [voice:name] tags, matched case-insensitively.",
    )

    @field_validator("voice_aliases")
    @classmethod
    def _validate_voice_aliases(cls, value):
        return _reject_tag_breaking_aliases(value)


class OpenAISpeechRequest(VoiceAliasesMixin):
    """Request schema for OpenAI-compatible speech endpoint"""

    model: str = Field(
        default="kokoro",
        description="The model to use for generation. Supported models: tts-1, tts-1-hd, kokoro",
    )
    input: InputText = Field(..., description="The text to generate audio for")
    voice: str = Field(
        default="af_heart",
        description="The voice to use for generation. Can be a base voice or a combined voice name.",
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="The format to return audio in. Supported formats: mp3, opus, aac, flac, wav, pcm. PCM format returns raw 16-bit samples without headers.",
    )
    download_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]] = (
        Field(
            default=None,
            description="Optional different format for the final download. If not provided, uses response_format.",
        )
    )
    speed: Rate = Field(default=1.0, description="The speed of the generated audio")
    stream: bool = Field(
        default=True,  # Default to streaming for OpenAI compatibility
        description="If true (default), audio will be streamed as it's generated. Each chunk will be a complete sentence.",
    )
    return_download_link: bool = Field(
        default=False,
        description="If true, returns a download link in X-Download-Path header after streaming completes",
    )
    return_timing: bool = Field(
        default=False,
        description="If true with return_download_link, writes per-chunk timing JSON next to the download file and returns its path in the X-Timing-Path header",
    )
    lang_code: Optional[LangCode] = Field(
        default=None,
        description="Optional language code to use for text processing. If not provided, will use first letter of voice name.",
    )
    volume_multiplier: Optional[Volume] = Field(
        default=1.0, description="A volume multiplier to multiply the output audio by."
    )
    normalization_options: Optional[NormalizationOptions] = Field(
        default=NormalizationOptions(),
        description="Options for the normalization system",
    )
    allow_voice_tags: bool = Field(
        default=False,
        description="If true, [voice:name] in the input switches speaker. Off by default so bracketed text is spoken as written.",
    )
    ssml: bool = Field(
        default=False,
        description="If true, the input is translated from SSML before synthesis. Requires allow_voice_tags, since the translation emits [voice:] and [rate:] spans.",
    )


class DialogueTurn(BaseModel):
    """A single speaker turn in a dialogue request"""

    model_config = ConfigDict(populate_by_name=True)

    voice: str = Field(
        ...,
        validation_alias=AliasChoices("voice", "voice_id"),
        description="The voice for this turn. Can be a base voice or a combined voice name. Accepts voice_id as an alias.",
    )
    text: InputText = Field(..., min_length=1, description="The text this speaker says")

    @field_validator("voice")
    @classmethod
    def _reject_tag_breaking_voice(cls, value: str) -> str:
        value = value.strip()
        if not TURN_VOICE_PATTERN.fullmatch(value):
            raise ValueError(
                "Voice contains characters that cannot appear in a voice name"
            )
        return value

    @field_validator("text")
    @classmethod
    def _reject_blank_text(cls, value: str) -> str:
        # a blank turn renders to a tag with nothing to say, which fails deep in generation
        if not value.strip():
            raise ValueError("Turn text cannot be empty")
        return value


class DialogueRequest(VoiceAliasesMixin):
    """Request schema for the multi-speaker dialogue endpoint"""

    model_config = ConfigDict(populate_by_name=True)

    model: str = Field(
        default="kokoro",
        description="The model to use for generation. Supported models: tts-1, tts-1-hd, kokoro",
    )
    turns: List[DialogueTurn] = Field(
        ...,
        min_length=1,
        validation_alias=AliasChoices("turns", "inputs"),
        description="Speaker turns rendered in order. Consecutive turns sharing a voice are merged. Accepts inputs as an alias.",
    )
    pause_between_turns: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Seconds of silence inserted between turns.",
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="The format to return audio in. Supported formats: mp3, opus, aac, flac, wav, pcm. PCM format returns raw 16-bit samples without headers.",
    )
    download_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]] = (
        Field(
            default=None,
            description="Optional different format for the final download. If not provided, uses response_format.",
        )
    )
    speed: Rate = Field(default=1.0, description="The speed of the generated audio")
    stream: bool = Field(
        default=True,
        description="If true (default), audio will be streamed as it's generated.",
    )
    return_download_link: bool = Field(
        default=False,
        description="If true, returns a download link in X-Download-Path header after streaming completes",
    )
    lang_code: Optional[LangCode] = Field(
        default=None,
        description="Optional language code override. If not provided, each turn uses the language implied by its own voice.",
    )
    volume_multiplier: Optional[Volume] = Field(
        default=1.0, description="A volume multiplier to multiply the output audio by."
    )
    normalization_options: Optional[NormalizationOptions] = Field(
        default=NormalizationOptions(),
        description="Options for the normalization system",
    )
    return_timing: bool = Field(
        default=False,
        description="If true and return_download_link is set, writes a timing sidecar and returns its path in X-Timing-Path.",
    )

    def to_tagged_input(self) -> str:
        """Render turns as the inline [voice:...] form the text pipeline consumes."""
        # plain float repr can be sci-notation, which the pause tag parser won't match
        pause = f"{self.pause_between_turns:.3f}".rstrip("0").rstrip(".")
        separator = f" [pause:{pause}s] " if float(pause) > 0 else " "
        return separator.join(
            f"[voice:{turn.voice}] {turn.text.strip()}" for turn in self.turns
        )

    @model_validator(mode="after")
    def _rendered_within_limit(self) -> "DialogueRequest":
        # the rendered string is what the speech endpoint actually receives
        _within_input_limit(self.to_tagged_input())
        return self


class CaptionedSpeechRequest(VoiceAliasesMixin):
    """Request schema for captioned speech endpoint"""

    model: str = Field(
        default="kokoro",
        description="The model to use for generation. Supported models: tts-1, tts-1-hd, kokoro",
    )
    input: InputText = Field(..., description="The text to generate audio for")
    voice: str = Field(
        default="af_heart",
        description="The voice to use for generation. Can be a base voice or a combined voice name.",
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="The format to return audio in. Supported formats: mp3, opus, aac, flac, wav, pcm. PCM format returns raw 16-bit samples without headers.",
    )
    speed: Rate = Field(default=1.0, description="The speed of the generated audio")
    stream: bool = Field(
        default=True,  # Default to streaming for OpenAI compatibility
        description="If true (default), audio will be streamed as it's generated. Each chunk will be a complete sentence.",
    )
    return_timestamps: bool = Field(
        default=True,
        description="If true (default), returns word-level timestamps in the response",
    )
    return_download_link: bool = Field(
        default=False,
        description="If true, returns a download link in X-Download-Path header after streaming completes",
    )
    lang_code: Optional[LangCode] = Field(
        default=None,
        description="Optional language code to use for text processing. If not provided, will use first letter of voice name.",
    )
    volume_multiplier: Optional[Volume] = Field(
        default=1.0, description="A volume multiplier to multiply the output audio by."
    )
    normalization_options: Optional[NormalizationOptions] = Field(
        default=NormalizationOptions(),
        description="Options for the normalization system",
    )
    allow_voice_tags: bool = Field(
        default=False,
        description="If true, [voice:name] in the input switches speaker. Off by default so bracketed text is spoken as written.",
    )
    ssml: bool = Field(
        default=False,
        description="If true, the input is translated from SSML before synthesis. Requires allow_voice_tags, since the translation emits [voice:] and [rate:] spans.",
    )
