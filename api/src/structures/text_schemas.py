from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class PhonemeRequest(BaseModel):
    text: str
    language: str = "a"  # Default to American English


class PhonemeResponse(BaseModel):
    phonemes: str
    tokens: list[int]


class StitchOptions(BaseModel):
    """Options for stitching audio chunks together"""

    gap_method: str = Field(
        default="static_trim",
        description="Method to handle gaps between chunks. Currently only 'static_trim' supported.",
    )
    trim_ms: int = Field(
        default=0,
        ge=0,
        description="Milliseconds to trim from chunk boundaries when using static_trim",
    )

    @field_validator("gap_method")
    @classmethod
    def validate_gap_method(cls, v: str) -> str:
        if v != "static_trim":
            raise ValueError("Currently only 'static_trim' gap method is supported")
        return v


class GenerateFromPhonemesRequest(BaseModel):
    """Simple request for phoneme-to-speech generation"""

    phonemes: str = Field(..., description="Phoneme string to synthesize")
    voice: str = Field(..., description="Voice ID to use for generation")


class SsmlRequest(BaseModel):
    """Request to translate SSML into native inline control tokens"""

    text: str = Field(..., description="SSML document starting with <speak>")
    voice: str = Field(
        default="",
        description="Voice the speech request will use. Enables [voice:]/[rate:] emission with reverts; when empty those elements are stripped and their content kept.",
    )


class SsmlResponse(BaseModel):
    text: str = Field(..., description="Native-tag text ready for the speech endpoints")
