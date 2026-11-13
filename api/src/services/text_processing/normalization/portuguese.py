"""Brazilian Portuguese text normalization."""

import re
from decimal import Decimal

from num2words import num2words

from .base import Normalizer


class PortugueseNormalizer(Normalizer):
    lang_codes = ("p",)

    # Portuguese uses a comma for decimals and a period for thousands. Keep the
    # groups bounded so long untrusted digit runs are left for the phonemizer.
    NUMBER_PATTERN = re.compile(
        r"(?<![\w:])(?P<sign>[+-]?)(?P<whole>\d{1,3}(?:\.\d{3}){1,9}|\d{1,30})"
        r"(?P<fraction>,\d{1,18})?(?![\w:])"
    )
    CURRENCY_PATTERN = re.compile(
        r"(?P<sign>(?<!\w)[+-])?(?P<symbol>R\$|US\$|[$€£])\s*"
        r"(?P<amount>\d{1,3}(?:\.\d{3}){1,9}(?:,\d{1,2})?"
        r"|\d{1,30}(?:,\d{1,2}|(?P<dot_cents>\.\d{2}))?)"
        r"(?!\d|[.,]\d)",
        re.IGNORECASE,
    )
    PERCENT_PATTERN = re.compile(r"(?<=\d)\s?%")
    TIME_PATTERN = re.compile(
        r"(?<![\w:])(?P<hour>\d{1,2})(?P<sep>[:h])(?P<minute>\d{2})(?:\s?(?:h|min|m)\b)?(?![\w:])",
        re.IGNORECASE,
    )
    QUANTITY = (
        r"(?P<sign>(?<!\w)[+-])?(?<![\w:])"
        r"(?P<number>(?:\d{1,3}(?:\.\d{3}){1,9}|\d{1,30})(?:,\d{1,18})?)\s*"
    )
    HOUR_PATTERN = re.compile(QUANTITY + r"(?P<unit>h)\b")
    UNIT_PATTERN = re.compile(
        QUANTITY + r"(?P<unit>km/h|m/s|km|cm|mm|kg|mg|g|ml|l|min|s|°c|°f)\b",
        re.IGNORECASE,
    )
    UNITS = {
        "km/h": ("quilômetro por hora", "quilômetros por hora"),
        "m/s": ("metro por segundo", "metros por segundo"),
        "km": ("quilômetro", "quilômetros"),
        "cm": ("centímetro", "centímetros"),
        "mm": ("milímetro", "milímetros"),
        "kg": ("quilograma", "quilogramas"),
        "mg": ("miligrama", "miligramas"),
        "g": ("grama", "gramas"),
        "ml": ("mililitro", "mililitros"),
        "l": ("litro", "litros"),
        "h": ("hora", "horas"),
        "min": ("minuto", "minutos"),
        "s": ("segundo", "segundos"),
        "°c": ("grau Celsius", "graus Celsius"),
        "°f": ("grau Fahrenheit", "graus Fahrenheit"),
    }
    FEMININE = {
        "um": "uma",
        "dois": "duas",
        "duzentos": "duzentas",
        "trezentos": "trezentas",
        "quatrocentos": "quatrocentas",
        "quinhentos": "quinhentas",
        "seiscentos": "seiscentas",
        "setecentos": "setecentas",
        "oitocentos": "oitocentas",
        "novecentos": "novecentas",
    }

    def units(self, text: str) -> str:
        return self.UNIT_PATTERN.sub(self._unit, text)

    def times(self, text: str) -> str:
        text = self.HOUR_PATTERN.sub(self._unit, text)

        def spell(match: re.Match[str]) -> str:
            hour, minute = int(match.group("hour")), int(match.group("minute"))
            if minute > 59 or (hour > 23 and match.group("sep") == ":"):
                return match.group()
            hour_words = self._feminine(num2words(hour, lang="pt_BR"))
            parts = [hour_words, "hora" if hour == 1 else "horas"]
            if minute:
                parts.extend(
                    (
                        "e",
                        num2words(minute, lang="pt_BR"),
                        "minuto" if minute == 1 else "minutos",
                    )
                )
            return " ".join(parts)

        return self.TIME_PATTERN.sub(spell, text)

    def numbers(self, text: str) -> str:
        text = self.CURRENCY_PATTERN.sub(self._currency, text)
        text = self.PERCENT_PATTERN.sub(" por cento", text)

        def spell(match: re.Match[str]) -> str:
            whole = int(match.group("whole").replace(".", ""))
            try:
                parts = [num2words(whole, lang="pt_BR")]
            except (NotImplementedError, OverflowError, ValueError):
                return match.group()
            if match.group("fraction"):
                digits = match.group("fraction")[1:].rstrip("0") or "0"
                parts.append("vírgula")
                parts.extend(num2words(int(digit), lang="pt_BR") for digit in digits)
            if match.group("sign") == "-":
                parts.insert(0, "menos")
            return " ".join(parts)

        return self.NUMBER_PATTERN.sub(spell, text)

    def _unit(self, match: re.Match[str]) -> str:
        amount = self._parse_amount(match.group("number"))
        unit = self.UNITS[match.group("unit").lower()]
        name = unit[0] if amount == 1 else unit[1]
        if unit[0] == "hora" and amount == int(amount):
            try:
                words = self._feminine(num2words(int(amount), lang="pt_BR"))
                sign = "menos " if match.group("sign") == "-" else ""
                return f"{sign}{words} {name}"
            except (NotImplementedError, OverflowError, ValueError):
                pass
        return f"{match.group('sign') or ''}{match.group('number')} {name}"

    def _currency(self, match: re.Match[str]) -> str:
        raw = match.group("amount")
        if match.group("dot_cents"):
            raw = raw.replace(".", ",")
        amount = self._parse_amount(raw)
        symbol = match.group("symbol").upper()
        currency = {
            "R$": ("real", "reais", "centavo", "centavos"),
            "US$": ("dólar", "dólares", "centavo", "centavos"),
            "$": ("dólar", "dólares", "centavo", "centavos"),
            "€": ("euro", "euros", "centavo", "centavos"),
            "£": ("libra", "libras", "centavo", "centavos"),
        }.get(symbol)
        if currency is None:
            return match.group()

        integer_part = int(amount)
        cents = int((amount - integer_part) * 100)
        try:
            words = num2words(integer_part, lang="pt_BR")
            if currency[0] == "libra":
                words = self._feminine(words)
            if words.endswith(("ão", "ões")):
                words += " de"
            parts = [words, currency[0] if integer_part == 1 else currency[1]]
            if cents:
                parts.extend(
                    (
                        "e",
                        num2words(cents, lang="pt_BR"),
                        currency[2] if cents == 1 else currency[3],
                    )
                )
            if match.group("sign") == "-":
                parts.insert(0, "menos")
            return " ".join(parts)
        except (NotImplementedError, OverflowError, ValueError):
            return match.group()

    def _feminine(self, words: str) -> str:
        tokens = words.split()
        start = max(
            (
                i + 1
                for i, token in enumerate(tokens)
                if token.rstrip(",").endswith(("ão", "ões"))
            ),
            default=0,
        )
        tokens[start:] = [self.FEMININE.get(token, token) for token in tokens[start:]]
        return " ".join(tokens)

    @staticmethod
    def _parse_amount(raw: str) -> Decimal:
        return Decimal(raw.replace(".", "").replace(",", "."))
