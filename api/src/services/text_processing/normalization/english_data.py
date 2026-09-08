"""Tables for the English normalizer: TLDs, units, symbol readings, currencies."""

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
    "~": " ",
    "@": " at ",
    "#": " number ",
    "$": " dollar ",
    "%": " percent ",
    "^": " ",
    "&": " and ",
    "*": " ",
    "_": " ",
    "|": " ",
    "\\": " ",
    "/": " slash ",
    "=": " equals ",
    "+": " plus ",
}

MONEY_UNITS = {"$": ("dollar", "cent"), "£": ("pound", "pence"), "€": ("euro", "cent")}
