"""
Text normalization module for TTS processing.
Handles various text formats including URLs, emails, numbers, money, and special characters.
Converts them into a format suitable for text-to-speech processing.
"""

import math
import re
from functools import lru_cache
from typing import List, Optional, Union

import inflect
from numpy import number
# from text_to_num import text2num
from torch import mul

from ...structures.schemas import NormalizationOptions

# Constants
VALID_TLDS = [
    "com",
    "org",
    "net",
    "edu",
    "gov",
    "mil",
    "int",
    "biz",
    "info",
    "name",
    "pro",
    "coop",
    "museum",
    "travel",
    "jobs",
    "mobi",
    "tel",
    "asia",
    "cat",
    "xxx",
    "aero",
    "arpa",
    "bg",
    "br",
    "ca",
    "cn",
    "de",
    "es",
    "eu",
    "fr",
    "in",
    "it",
    "jp",
    "mx",
    "nl",
    "ru",
    "uk",
    "us",
    "io",
    "co",
]

VALID_UNITS = {
    "m": "meter",
    "cm": "centimeter",
    "mm": "millimeter",
    "km": "kilometer",
    "in": "inch",
    "ft": "foot",
    "yd": "yard",
    "mi": "mile",  # Length
    "g": "gram",
    "kg": "kilogram",
    "mg": "milligram",  # Mass
    "s": "second",
    "ms": "millisecond",
    "min": "minutes",
    "h": "hour",  # Time
    "l": "liter",
    "ml": "mililiter",
    "cl": "centiliter",
    "dl": "deciliter",  # Volume
    "kph": "kilometer per hour",
    "mph": "mile per hour",
    "mi/h": "mile per hour",
    "m/s": "meter per second",
    "km/h": "kilometer per hour",
    "mm/s": "milimeter per second",
    "cm/s": "centimeter per second",
    "ft/s": "feet per second",
    "cm/h": "centimeter per day",  # Speed
    "°c": "degree celsius",
    "c": "degree celsius",
    "°f": "degree fahrenheit",
    "f": "degree fahrenheit",
    "k": "kelvin",  # Temperature
    "pa": "pascal",
    "kpa": "kilopascal",
    "mpa": "megapascal",
    "atm": "atmosphere",  # Pressure
    "hz": "hertz",
    "khz": "kilohertz",
    "mhz": "megahertz",
    "ghz": "gigahertz",  # Frequency
    "v": "volt",
    "kv": "kilovolt",
    "mv": "mergavolt",  # Voltage
    "a": "amp",
    "ma": "megaamp",
    "ka": "kiloamp",  # Current
    "w": "watt",
    "kw": "kilowatt",
    "mw": "megawatt",  # Power
    "j": "joule",
    "kj": "kilojoule",
    "mj": "megajoule",  # Energy
    "Ω": "ohm",
    "kΩ": "kiloohm",
    "mΩ": "megaohm",  # Resistance (Ohm)
    "f": "farad",
    "µf": "microfarad",
    "nf": "nanofarad",
    "pf": "picofarad",  # Capacitance
    "b": "bit",
    "kb": "kilobit",
    "mb": "megabit",
    "gb": "gigabit",
    "tb": "terabit",
    "pb": "petabit",  # Data size
    "kbps": "kilobit per second",
    "mbps": "megabit per second",
    "gbps": "gigabit per second",
    "tbps": "terabit per second",
    "px": "pixel",  # CSS units
}

SYMBOL_REPLACEMENTS = {
    '~': ' ',
    '@': ' at ',
    '#': ' number ',
    '$': ' dollar ',
    '%': ' percent ',
    '^': ' ',
    '&': ' and ',
    '*': ' ',
    '_': ' ',
    '|': ' ',
    '\\': ' ',
    '/': ' slash ',
    '=': ' equals ',
    '+': ' plus ',
}

MONEY_UNITS = {"$": ("dollar", "cent"), "£": ("pound", "pence"), "€": ("euro", "cent")}

# Pre-compiled regex patterns for performance
EMAIL_PATTERN = re.compile(
    r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}\b", re.IGNORECASE
)
URL_PATTERN = re.compile(
    r"(https?://|www\.|)+(localhost|[a-zA-Z0-9.-]+(\.(?:"
    + "|".join(VALID_TLDS)
    + "))+|[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})(:[0-9]+)?([/?][^\s]*)?",
    re.IGNORECASE,
)

UNIT_PATTERN = re.compile(
    r"((?<!\w)([+-]?)(\d{1,3}(,\d{3})*|\d+)(\.\d+)?)\s*("
    + "|".join(sorted(list(VALID_UNITS.keys()), reverse=True))
    + r"""){1}(?=[^\w\d]{1}|\b)""",
    re.IGNORECASE,
)

TIME_PATTERN = re.compile(
    r"([0-9]{1,2} ?: ?[0-9]{2}( ?: ?[0-9]{2})?)( ?(pm|am)\b)?", re.IGNORECASE
)

MONEY_PATTERN = re.compile(
    r"(-?)(["
    + "".join(MONEY_UNITS.keys())
    + r"])(\d+(?:\.\d+)?)((?: hundred| thousand| (?:[bm]|tr|quadr)illion|k|m|b|t)*)\b",
    re.IGNORECASE,
)

NUMBER_PATTERN = re.compile(
    r"(-?)(\d+(?:\.\d+)?)((?: hundred| thousand| (?:[bm]|tr|quadr)illion|k|m|b)*)\b",
    re.IGNORECASE,
)

INFLECT_ENGINE = inflect.engine()


def handle_units(u: re.Match[str]) -> str:
    """Converts units to their full form"""
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
    part1 = str(conditional_int(number))[:2]
    part2 = str(conditional_int(number))[2:]
    return f"{INFLECT_ENGINE.number_to_words(part1)} {INFLECT_ENGINE.number_to_words(part2)}"


def handle_numbers(n: re.Match[str]) -> str:
    number = n.group(2)

    try:
        number = float(number)
    except:
        return n.group()

    if n.group(1) == "-":
        number *= -1

    multiplier = translate_multiplier(n.group(3))

    number = conditional_int(number)
    if multiplier != "":
        multiplier = f" {multiplier}"
    else:
        if (
            number % 1 == 0
            and len(str(number)) == 4
            and number > 1500
            and number % 1000 > 9
        ):
            return split_four_digit(number)

    return f"{INFLECT_ENGINE.number_to_words(number)}{multiplier}"


def handle_money(m: re.Match[str]) -> str:
    """Convert money expressions to spoken form"""

    bill, coin = MONEY_UNITS[m.group(2)]

    number = m.group(3)

    try:
        number = float(number)
    except:
        return m.group()

    if m.group(1) == "-":
        number *= -1

    multiplier = translate_multiplier(m.group(4))

    if multiplier != "":
        multiplier = f" {multiplier}"

    if number % 1 == 0 or multiplier != "":
        text_number = f"{INFLECT_ENGINE.number_to_words(conditional_int(number))}{multiplier} {INFLECT_ENGINE.plural(bill, count=number)}"
    else:
        sub_number = int(str(number).split(".")[-1].ljust(2, "0"))

        text_number = f"{INFLECT_ENGINE.number_to_words(int(math.floor(number)))} {INFLECT_ENGINE.plural(bill, count=number)} and {INFLECT_ENGINE.number_to_words(sub_number)} {INFLECT_ENGINE.plural(coin, count=sub_number)}"

    return text_number


def handle_decimal(num: re.Match[str]) -> str:
    """Convert decimal numbers to spoken form"""
    a, b = num.group().split(".")
    return " point ".join([a, " ".join(b)])


def handle_email(m: re.Match[str]) -> str:
    """Convert email addresses into speakable format"""
    email = m.group(0)
    parts = email.split("@")
    if len(parts) == 2:
        user, domain = parts
        domain = domain.replace(".", " dot ")
        return f"{user} at {domain}"
    return email


def handle_url(u: re.Match[str]) -> str:
    """Make URLs speakable by converting special characters to spoken words"""
    if not u:
        return ""

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
    parts = url.split("/", 1)
    domain = parts[0]
    path = parts[1] if len(parts) > 1 else ""

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


def handle_phone_number(p: re.Match[str]) -> str:
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


def handle_time(t: re.Match[str]) -> str:
    t = t.groups()

    time_parts = t[0].split(":")

    numbers = []
    numbers.append(INFLECT_ENGINE.number_to_words(time_parts[0].strip()))

    minute_number = INFLECT_ENGINE.number_to_words(time_parts[1].strip())
    if int(time_parts[1]) < 10:
        if int(time_parts[1]) != 0:
            numbers.append(f"oh {minute_number}")
    else:
        numbers.append(minute_number)

    half = ""
    if len(time_parts) > 2:
        seconds_number = INFLECT_ENGINE.number_to_words(time_parts[2].strip())
        second_word = INFLECT_ENGINE.plural("second", int(time_parts[2].strip()))
        numbers.append(f"and {seconds_number} {second_word}")
    else:
        if t[2] is not None:
            half = " " + t[2].strip()
        else:
            if int(time_parts[1]) == 0:
                numbers.append("o'clock")

    return " ".join(numbers) + half


def normalize_text(text: str, normalization_options: NormalizationOptions) -> str:
    """Normalize text for TTS processing"""
    
    # Handle email addresses first if enabled
    if normalization_options.email_normalization:
        text = EMAIL_PATTERN.sub(handle_email, text)

    # Handle URLs if enabled
    if normalization_options.url_normalization:
        text = URL_PATTERN.sub(handle_url, text)

    # Pre-process numbers with units if enabled
    if normalization_options.unit_normalization:
        text = UNIT_PATTERN.sub(handle_units, text)

    # Replace optional pluralization
    if normalization_options.optional_pluralization_normalization:
        text = re.sub(r"\(s\)", "s", text)

    # Replace phone numbers:
    if normalization_options.phone_normalization:
        text = re.sub(
            r"(\+?\d{1,2})?([ .-]?)(\(?\d{3}\)?)[\s.-](\d{3})[\s.-](\d{4})",
            handle_phone_number,
            text,
        )

    # Replace quotes and brackets (additional cleanup)
    text = text.replace(chr(8216), "'").replace(chr(8217), "'")
    text = text.replace("«", chr(8220)).replace("»", chr(8221))
    text = text.replace(chr(8220), '"').replace(chr(8221), '"')

    # Handle CJK punctuation and some non standard chars
    for a, b in zip("、。！，：；？–", ",.!,:;?-"):
        text = text.replace(a, b + " ")

    # Handle simple time in the format of HH:MM:SS (am/pm)
    text = TIME_PATTERN.sub(
        handle_time,
        text,
    )

    # Clean up whitespace
    text = re.sub(r"[^\S \n]", " ", text)
    text = re.sub(r"  +", " ", text)
    text = re.sub(r"(?<=\n) +(?=\n)", "", text)

    # Handle special characters that might cause audio artifacts first
    # Replace newlines with spaces (or pauses if needed)
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    
    # Handle titles and abbreviations
    text = re.sub(r"\bD[Rr]\.(?= [A-Z])", "Doctor", text)
    text = re.sub(r"\b(?:Mr\.|MR\.(?= [A-Z]))", "Mister", text)
    text = re.sub(r"\b(?:Ms\.|MS\.(?= [A-Z]))", "Miss", text)
    text = re.sub(r"\b(?:Mrs\.|MRS\.(?= [A-Z]))", "Mrs", text)
    text = re.sub(r"\betc\.(?! [A-Z])", "etc", text)

    # Handle common words
    text = re.sub(r"(?i)\b(y)eah?\b", r"\1e'a", text)

    # Handle numbers and money BEFORE replacing special characters
    text = re.sub(r"(?<=\d),(?=\d)", "", text)

    text = MONEY_PATTERN.sub(
        handle_money,
        text,
    )

    text = NUMBER_PATTERN.sub(handle_numbers, text)

    text = re.sub(r"\d*\.\d+", handle_decimal, text)

    # Handle other problematic symbols AFTER money/number processing
    if normalization_options.replace_remaining_symbols:
        for symbol, replacement in SYMBOL_REPLACEMENTS.items():
            text = text.replace(symbol, replacement)

    # Handle various formatting
    text = re.sub(r"(?<=\d)-(?=\d)", " to ", text)
    text = re.sub(r"(?<=\d)S", " S", text)
    text = re.sub(r"(?<=[BCDFGHJ-NP-TV-Z])'?s\b", "'S", text)
    text = re.sub(r"(?<=X')S\b", "s", text)
    text = re.sub(
        r"(?:[A-Za-z]\.){2,} [a-z]", lambda m: m.group().replace(".", "-"), text
    )
    text = re.sub(r"(?i)(?<=[A-Z])\.(?=[A-Z])", "-", text)

    text = re.sub(r"\s{2,}", " ", text)

    return text
