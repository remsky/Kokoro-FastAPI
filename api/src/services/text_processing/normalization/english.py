"""English text normalization: URLs, emails, numbers, money, time, units."""

import functools
import math
import re

import inflect

from .base import Normalizer
from .english_data import MONEY_UNITS, SYMBOL_REPLACEMENTS, VALID_TLDS, VALID_UNITS

INFLECT_ENGINE = inflect.engine()


def conditional_int(number: float, threshold: float = 0.00001):
    if abs(round(number) - number) < threshold:
        return int(round(number))
    return number


def translate_multiplier(multiplier: str) -> str:
    """Translate multiplier abrevations to words"""

    multiplier_translation = {
        "k": "thousand",
        "m": "million",
        "b": "billion",
        "t": "trillion",
    }
    if multiplier.lower() in multiplier_translation:
        return multiplier_translation[multiplier.lower()]
    return multiplier.strip()


def split_four_digit(number: float):
    digits = str(conditional_int(number))
    return f"{INFLECT_ENGINE.number_to_words(digits[:2])} {INFLECT_ENGINE.number_to_words(digits[2:])}"


def unchanged_on_error(handler):
    """Leave the token as is when the number is beyond what inflect or int() accept."""

    @functools.wraps(handler)
    def wrapped(m: re.Match[str]) -> str:
        try:
            return handler(m)
        except (inflect.NumOutOfRangeError, IndexError, OverflowError, ValueError):
            return m.group()

    return wrapped


class EnglishNormalizer(Normalizer):
    lang_codes = ("a", "b", "en-us", "en-gb")

    def contractions(self, text: str) -> str:
        # Expand the "'re" contractions that espeak mis-phonemizes with a spurious
        # /ɹeɪ/ ("-ray") ending, e.g. "how're" -> /haʊɹeɪ/ which sounds like "harry".
        # Only the wh-words and there/these/those break this way; "you're", "we're"
        # and "they're" already phonemize correctly and are left untouched.
        text = re.sub(
            r"\b(how|what|where|who|when|why|there|these|those)['’]re\b",
            r"\1 are",
            text,
            flags=re.IGNORECASE,
        )
        # apostrophe-less variants ("howre", "theyre"); word list differs from above
        # because some forms like "were" and "whore" are real words, not contractions
        return re.sub(
            r"\b(how|what|where|when|why|there|these|those|you|they)re\b",
            r"\1 are",
            text,
            flags=re.IGNORECASE,
        )

    # Lengths bounded to RFC limits (64/253) to prevent O(n^2) backtracking on floods.
    EMAIL_PATTERN = re.compile(
        r"\b[a-zA-Z0-9._%+-]{1,64}@[a-zA-Z0-9.-]{1,253}\.[a-z]{2,}\b", re.IGNORECASE
    )

    def emails(self, text: str) -> str:
        def handle(m: re.Match[str]) -> str:
            email = m.group(0)
            parts = email.split("@")
            if len(parts) == 2:
                user, domain = parts
                domain = domain.replace(".", " dot ")
                return f"{user} at {domain}"
            return email

        return self.EMAIL_PATTERN.sub(handle, text)

    URL_PATTERN = re.compile(
        # Token-start lookbehind + 253-char domain bound: without them the domain
        # branch rescans from every position of a long word run (O(n^2)).
        r"(?<![a-zA-Z0-9.-])"
        # Two optionals, not (https?://|www\.|)+. '+' over an empty alternative is a ReDoS smell.
        r"(?:https?://)?(?:www\.)?(localhost|[a-zA-Z0-9.-]{1,253}(\.(?:"
        + "|".join(VALID_TLDS)
        + r"))+|[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})(:[0-9]+)?([/?][^\s]*)?",
        re.IGNORECASE,
    )

    def urls(self, text: str) -> str:
        def handle(u: re.Match[str]) -> str:
            url = u.group(0).strip()

            # Handle protocol first
            url = re.sub(
                r"^https?://",
                lambda a: "https " if "https" in a.group() else "http ",
                url,
                flags=re.IGNORECASE,
            )
            url = re.sub(r"^www\.", "www ", url, flags=re.IGNORECASE)

            # Handle port numbers before other replacements
            url = re.sub(r":(\d+)(?=/|$)", lambda m: f" colon {m.group(1)}", url)

            # Split into domain and path
            domain, _, path = url.partition("/")

            # Handle dots in domain
            domain = domain.replace(".", " dot ")

            # Reconstruct URL
            if path:
                url = f"{domain} slash {path}"
            else:
                url = domain

            # Replace remaining symbols with words
            url = url.replace("-", " dash ")
            url = url.replace("_", " underscore ")
            url = url.replace("?", " question-mark ")
            url = url.replace("=", " equals ")
            url = url.replace("&", " ampersand ")
            url = url.replace("%", " percent ")
            url = url.replace(":", " colon ")  # Handle any remaining colons
            url = url.replace("/", " slash ")  # Handle any remaining slashes

            # Clean up extra spaces
            return re.sub(r"\s+", " ", url).strip()

        return self.URL_PATTERN.sub(handle, text)

    UNIT_PATTERN = re.compile(
        r"((?<!\w)(?<!\w[.,])([+-]?)(\d{1,3}(,\d{3})*|\d+)(\.\d+)?)\s*("
        + "|".join(sorted(list(VALID_UNITS.keys()), reverse=True))
        + r"""){1}(?=[^\w\d]{1}|\b)""",
        re.IGNORECASE,
    )

    def units(self, text: str) -> str:
        def handle(u: re.Match[str]) -> str:
            unit_string = u.group(6).strip()
            unit = unit_string

            if unit_string.lower() in VALID_UNITS:
                unit = VALID_UNITS[unit_string.lower()].split(" ")

                # Handles the B vs b case
                if unit[0].endswith("bit"):
                    b_case = unit_string[min(1, len(unit_string) - 1)]
                    if b_case == "B":
                        unit[0] = unit[0][:-3] + "byte"

                number = u.group(1).strip()
                unit[0] = INFLECT_ENGINE.no(unit[0], number)
            return " ".join(unit)

        return self.UNIT_PATTERN.sub(handle, text)

    def optional_plurals(self, text: str) -> str:
        return re.sub(r"\(s\)", "s", text)

    PHONE_PATTERN = re.compile(
        r"(\+?\d{1,2})?([ .-]?)(\(?\d{3}\)?)[\s.-](\d{3})[\s.-](\d{4})"
    )

    def phone_numbers(self, text: str) -> str:
        def handle(p: re.Match[str]) -> str:
            p = list(p.groups())

            country_code = ""
            if p[0] is not None:
                p[0] = p[0].replace("+", "")
                country_code += INFLECT_ENGINE.number_to_words(p[0])

            area_code = INFLECT_ENGINE.number_to_words(
                p[2].replace("(", "").replace(")", ""), group=1, comma=""
            )

            telephone_prefix = INFLECT_ENGINE.number_to_words(p[3], group=1, comma="")

            line_number = INFLECT_ENGINE.number_to_words(p[4], group=1, comma="")

            return ",".join([country_code, area_code, telephone_prefix, line_number])

        return self.PHONE_PATTERN.sub(handle, text)

    TIME_PATTERN = re.compile(
        r"([0-9]{1,2} ?: ?[0-9]{2}( ?: ?[0-9]{2})?)( ?(pm|am)\b)?", re.IGNORECASE
    )

    def times(self, text: str) -> str:
        def handle(t: re.Match[str]) -> str:
            time_parts = t.group(1).split(":")
            minute = int(time_parts[1])
            half = t.group(4)

            numbers = [INFLECT_ENGINE.number_to_words(time_parts[0].strip())]
            minute_number = INFLECT_ENGINE.number_to_words(time_parts[1].strip())
            if 0 < minute < 10:
                numbers.append(f"oh {minute_number}")
            elif minute >= 10:
                numbers.append(minute_number)

            if len(time_parts) > 2:
                seconds = int(time_parts[2])
                seconds_number = INFLECT_ENGINE.number_to_words(time_parts[2].strip())
                second_word = INFLECT_ENGINE.plural("second", seconds)
                numbers.append(f"and {seconds_number} {second_word}")
            elif half is None and minute == 0:
                numbers.append("o'clock")

            if half:
                numbers.append(half)
            return " ".join(numbers)

        return self.TIME_PATTERN.sub(handle, text)

    def abbreviations(self, text: str) -> str:
        text = re.sub(r"\bD[Rr]\.(?= [A-Z])", "Doctor", text)
        text = re.sub(r"\b(?:Mr\.|MR\.(?= [A-Z]))", "Mister", text)
        text = re.sub(r"\b(?:Ms\.|MS\.(?= [A-Z]))", "Miss", text)
        text = re.sub(r"\b(?:Mrs\.|MRS\.(?= [A-Z]))", "Mrs", text)
        return re.sub(r"\betc\.(?! [A-Z])", "etc", text)

    def common_words(self, text: str) -> str:
        return re.sub(r"(?i)\b(y)eah?\b", r"\1e'a", text)

    def numbers(self, text: str) -> str:
        text = re.sub(r"(?<=\d)S", " S", text)
        text = self._comma_groups(text)
        text = re.sub(r"(?<=\d)-(?=\d)", " to ", text)
        # version-like sequences (2.0.1, 10.3.2) before NUMBER_PATTERN eats the first decimal
        text = self._versions(text)
        text = self._money(text)
        return self._number_words(text)

    COMMA_RUN_PATTERN = re.compile(r"(?<!\d)\d+(?:,\d+)+")
    THOUSANDS_PATTERN = re.compile(r"\d{1,3}(?:,\d{3})+")

    def _comma_groups(self, text: str) -> str:
        def handle(m: re.Match[str]) -> str:
            if self.THOUSANDS_PATTERN.fullmatch(m.group()):
                return m.group()
            return m.group().replace(",", "")

        return self.COMMA_RUN_PATTERN.sub(handle, text)

    VERSION_PATTERN = re.compile(r"(?<!\d)\d+(?:\.\d+){2,}")

    def _versions(self, text: str) -> str:
        @unchanged_on_error
        def handle(m: re.Match[str]) -> str:
            parts = m.group().split(".")
            return " point ".join(INFLECT_ENGINE.number_to_words(int(p)) for p in parts)

        return self.VERSION_PATTERN.sub(handle, text)

    NOT_GLUED_BEFORE = r"(?<![^\W_])(?<![^\W_][.,])"
    NOT_GLUED_AFTER = r"(?![^\W_]|\.\d|,\d)"
    NUMBER_BODY = r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    NUMBER_MULTIPLIER = r"((?: hundred| thousand| (?:[bm]|tr|quadr)illion|k|m|b)*)"
    MONEY_MULTIPLIER = r"((?: hundred| thousand| (?:[bm]|tr|quadr)illion|k|m|b|t)*)"

    MONEY_PATTERN = re.compile(
        r"(-?)(["
        + "".join(MONEY_UNITS.keys())
        + r"])"
        + NUMBER_BODY
        + MONEY_MULTIPLIER
        + NOT_GLUED_AFTER,
        re.IGNORECASE,
    )

    def _money(self, text: str) -> str:
        @unchanged_on_error
        def handle(m: re.Match[str]) -> str:
            bill, coin = MONEY_UNITS[m.group(2)]
            number = float(m.group(3).replace(",", ""))

            if m.group(1) == "-":
                number *= -1

            multiplier = translate_multiplier(m.group(4))

            if multiplier != "":
                multiplier = f" {multiplier}"

            bills = INFLECT_ENGINE.plural(bill, count=number)
            if number % 1 == 0 or multiplier != "":
                whole = INFLECT_ENGINE.number_to_words(conditional_int(number))
                return f"{whole}{multiplier} {bills}"

            whole = INFLECT_ENGINE.number_to_words(int(math.floor(number)))
            fraction = int(str(number).split(".")[-1].ljust(2, "0"))
            coins = INFLECT_ENGINE.plural(coin, count=fraction)
            return f"{whole} {bills} and {INFLECT_ENGINE.number_to_words(fraction)} {coins}"

        return self.MONEY_PATTERN.sub(handle, text)

    NUMBER_PATTERN = re.compile(
        NOT_GLUED_BEFORE + r"(-?)" + NUMBER_BODY + NUMBER_MULTIPLIER + NOT_GLUED_AFTER,
        re.IGNORECASE,
    )

    def _number_words(self, text: str) -> str:
        @unchanged_on_error
        def handle(n: re.Match[str]) -> str:
            raw = n.group(2)
            number = float(raw.replace(",", ""))

            if n.group(1) == "-":
                number *= -1

            multiplier = translate_multiplier(n.group(3))

            number = conditional_int(number)
            if multiplier != "":
                multiplier = f" {multiplier}"
            elif (
                "," not in raw
                and number % 1 == 0
                and len(str(number)) == 4
                and number > 1500
                and number % 1000 > 9
            ):
                return split_four_digit(number)

            return f"{INFLECT_ENGINE.number_to_words(number)}{multiplier}"

        return self.NUMBER_PATTERN.sub(handle, text)

    def symbols(self, text: str) -> str:
        text = re.sub(r" ?-{2,} ?", " — ", text)
        for symbol, replacement in SYMBOL_REPLACEMENTS.items():
            text = text.replace(symbol, replacement)
        return text

    def possessives(self, text: str) -> str:
        text = re.sub(r"(?<=[BCDFGHJ-NP-TV-Z])'?s\b", "'S", text)
        return re.sub(r"(?<=X')S\b", "s", text)

    def acronyms(self, text: str) -> str:
        text = re.sub(
            # {2,12} not {2,}: real acronyms are short; unbounded backtracks O(n^2) on floods.
            r"(?:[A-Za-z]\.){2,12} [a-z]",
            lambda m: m.group().replace(".", "-"),
            text,
        )
        return re.sub(r"(?i)(?<=[A-Z])\.(?=[A-Z])", "-", text)
