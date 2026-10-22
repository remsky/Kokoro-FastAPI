"""Per-language text normalizers. CONTRIBUTING.md covers adding a language."""

from ....structures.schemas import NormalizationOptions
from .base import Normalizer
from .english import EnglishNormalizer

ENGLISH = EnglishNormalizer()

LANGUAGES: tuple[Normalizer, ...] = (ENGLISH,)

NORMALIZERS: dict[str, Normalizer] = {
    code: normalizer for normalizer in LANGUAGES for code in normalizer.lang_codes
}


def get_normalizer(lang_code: str) -> Normalizer | None:
    return NORMALIZERS.get(lang_code)


def normalize_text(text: str, normalization_options: NormalizationOptions) -> str:
    """Normalize English text. The per-language entry point is get_normalizer."""
    return ENGLISH.normalize(text, normalization_options)
