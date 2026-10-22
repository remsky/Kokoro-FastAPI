"""Language-neutral normalizer and the contract each language implements."""

import re
from collections.abc import Callable

from ....structures.schemas import NormalizationOptions

CJK_PUNCTUATION = dict(zip("、。！，：；？–", ",.!,:;?-"))


class Normalizer:
    """Base normalizer, one subclass per language.

    - lang_codes: the pipeline codes the subclass serves.
    - pass_order: money and units before numbers, symbols after, acronyms
      last. Override it, not normalize, to change the order.
    - Hooks: text in, text out, no-op by default. Override as needed.
    """

    lang_codes: tuple[str, ...] = ()

    def pass_order(
        self, options: NormalizationOptions
    ) -> tuple[Callable[[str], str] | None, ...]:
        return (
            self.caps_words if options.caps_normalization else None,
            self.contractions,
            self.emails if options.email_normalization else None,
            self.urls if options.url_normalization else None,
            self.units if options.unit_normalization else None,
            self.optional_plurals
            if options.optional_pluralization_normalization
            else None,
            self.phone_numbers if options.phone_normalization else None,
            self.quotes,
            self.cjk_punctuation,
            self.times,
            self.whitespace,
            self.abbreviations,
            self.common_words,
            self.numbers,
            self.symbols if options.replace_remaining_symbols else None,
            self.acronyms,
        )

    def normalize(self, text: str, options: NormalizationOptions) -> str:
        for step in self.pass_order(options):
            if step is not None:
                text = step(text)
        return re.sub(r"\s{2,}", " ", text)

    # neutral passes
    def quotes(self, text: str) -> str:
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u00ab", "\u201c").replace("\u00bb", "\u201d")
        return text.replace("\u201c", '"').replace("\u201d", '"')

    def cjk_punctuation(self, text: str) -> str:
        for a, b in CJK_PUNCTUATION.items():
            text = text.replace(a, b + " ")
        return text

    def whitespace(self, text: str) -> str:
        text = re.sub(r"[^\S ]", " ", text)
        return re.sub(r"  +", " ", text)

    # language hooks, no-op until overridden
    def caps_words(self, text: str) -> str:
        return text

    def contractions(self, text: str) -> str:
        return text

    def emails(self, text: str) -> str:
        return text

    def urls(self, text: str) -> str:
        return text

    def units(self, text: str) -> str:
        return text

    def optional_plurals(self, text: str) -> str:
        return text

    def phone_numbers(self, text: str) -> str:
        return text

    def times(self, text: str) -> str:
        return text

    def abbreviations(self, text: str) -> str:
        return text

    def common_words(self, text: str) -> str:
        return text

    def numbers(self, text: str) -> str:
        return text

    def symbols(self, text: str) -> str:
        return text

    def acronyms(self, text: str) -> str:
        return text
