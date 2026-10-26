"""Brazilian Portuguese text normalization."""

import re
from decimal import Decimal, InvalidOperation

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
        r"(?P<symbol>R\$|US\$|[$€£])\s*"
        r"(?P<amount>\d{1,3}(?:\.\d{3}){1,9}(?:,\d{1,2})?|\d{1,30}(?:,\d{1,2})?)",
        re.IGNORECASE,
    )
    TIME_PATTERN = re.compile(r"(?<!\d)(?P<hour>\d{1,2}):(?P<minute>\d{2})(?!\d)")
    UNIT_PATTERN = re.compile(
        r"(?P<number>(?:\d{1,3}(?:\.\d{3}){1,9}|\d{1,30})(?:,\d{1,18})?)\s*"
        r"(?P<unit>km/h|m/s|km|cm|mm|kg|mg|g|ml|l|h|min|s|°c|°f)\b",
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

    def units(self, text: str) -> str:
        def spell(match: re.Match[str]) -> str:
            amount = self._parse_amount(match.group("number"))
            unit = self.UNITS[match.group("unit").lower()]
            name = unit[0] if amount == 1 else unit[1]
            return f"{match.group('number')} {name}"

        return self.UNIT_PATTERN.sub(spell, text)

    def times(self, text: str) -> str:
        def spell(match: re.Match[str]) -> str:
            hour, minute = int(match.group("hour")), int(match.group("minute"))
            if hour > 23 or minute > 59:
                return match.group()
            parts = [num2words(hour, lang="pt_BR"), "hora" if hour == 1 else "horas"]
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

        def spell(match: re.Match[str]) -> str:
            raw = match.group("whole").replace(".", "")
            if match.group("fraction"):
                raw += "." + match.group("fraction")[1:]
            try:
                value = Decimal(raw)
                if match.group("sign") == "-":
                    value = -value
                return num2words(value, lang="pt_BR")
            except (InvalidOperation, NotImplementedError, OverflowError, ValueError):
                return match.group()

        return self.NUMBER_PATTERN.sub(spell, text)

    def _currency(self, match: re.Match[str]) -> str:
        amount = self._parse_amount(match.group("amount"))
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
            parts = [num2words(integer_part, lang="pt_BR")]
            parts.append(currency[0] if integer_part == 1 else currency[1])
            if cents:
                parts.extend(
                    (
                        "e",
                        num2words(cents, lang="pt_BR"),
                        currency[2] if cents == 1 else currency[3],
                    )
                )
            return " ".join(parts)
        except (NotImplementedError, OverflowError, ValueError):
            return match.group()

    @staticmethod
    def _parse_amount(raw: str) -> Decimal:
        return Decimal(raw.replace(".", "").replace(",", "."))
