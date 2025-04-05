import re

import inflect
from text_to_num import text2num
from torch import mul

INFLECT_ENGINE = inflect.engine()


def conditional_int(number: float, threshold: float = 0.00001):
    if abs(round(number) - number) < threshold:
        return int(round(number))
    return number


def handle_money(m: re.Match[str]) -> str:
    """Convert money expressions to spoken form"""

    bill = "dollar" if m.group(2) == "$" else "pound"
    coin = "cent" if m.group(2) == "$" else "pence"
    number = m.group(3)

    multiplier = m.group(4)
    try:
        number = float(number)
    except:
        return m.group()

    if m.group(1) == "-":
        number *= -1

    if number % 1 == 0 or multiplier != "":
        text_number = f"{INFLECT_ENGINE.number_to_words(conditional_int(number))}{multiplier} {INFLECT_ENGINE.plural(bill, count=number)}"
    else:
        sub_number = int(str(number).split(".")[-1].ljust(2, "0"))

        text_number = f"{INFLECT_ENGINE.number_to_words(int(round(number)))} {INFLECT_ENGINE.plural(bill, count=number)} and {INFLECT_ENGINE.number_to_words(sub_number)} {INFLECT_ENGINE.plural(coin, count=sub_number)}"

    return text_number


text = re.sub(
    r"(?i)(-?)([$£])(\d+(?:\.\d+)?)((?: hundred| thousand| (?:[bm]|tr|quadr)illion)*)\b",
    handle_money,
    "he administration has offered up a platter of repression for more than a year and is still slated to lose -$5.3 billion",
)
print(text)
