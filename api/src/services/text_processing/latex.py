"""LaTeX math to spoken English.

Converts math inside explicit delimiters ($$..$$, \\[..\\], \\(..\\), $..$)
into words, so normalize_text can run before money and symbol replacement
mangle the backslashes. Anything outside a delimiter is left untouched, and a
span that fails to parse is returned verbatim.
"""

import re
import unicodedata
from typing import Optional

from pylatexenc import latexwalker, macrospec
from pylatexenc.latex2text import (
    LatexNodes2Text,
    MacroTextSpec,
    get_default_latex_context_db,
)

MAX_SPAN = 600

MATH_SPAN = re.compile(
    r"\$\$.{1," + str(MAX_SPAN) + r"}?\$\$"
    r"|\\\[.{1," + str(MAX_SPAN) + r"}?\\\]"
    r"|\\\(.{1," + str(MAX_SPAN) + r"}?\\\)"
    r"|\$(?=[^$\n]*[\\^_{])[^$\n]{1,200}?\$",
    re.DOTALL,
)

BIG_OPS = "∫∑∏⋃⋂∮"

BIG_OP_LIMITS = re.compile(r"(?<=[" + BIG_OPS + r"])_([^\s^_]+)(?:\^([^\s^_]+))?")
SUPERSCRIPT = re.compile(r"\^([^\s^_]+)")
SUBSCRIPT = re.compile(r"_([^\s^_]+)")
COMBINING = re.compile(r"[\u0300-\u036f]")

MATH_SYMBOLS = {
    "≤": " less than or equal to ",
    "≥": " greater than or equal to ",
    "≪": " much less than ",
    "≫": " much greater than ",
    "≠": " not equal to ",
    "≈": " approximately ",
    "≡": " identical to ",
    "∼": " similar to ",
    "∝": " proportional to ",
    "±": " plus or minus ",
    "∓": " minus or plus ",
    "×": " times ",
    "⋅": " times ",
    "·": " times ",
    "÷": " divided by ",
    "∞": " infinity ",
    "∫": " the integral ",
    "∮": " the contour integral ",
    "∑": " the sum ",
    "∏": " the product ",
    "√": " the square root of ",
    "∂": " partial ",
    "∇": " del ",
    "∈": " in ",
    "∉": " not in ",
    "⊂": " subset of ",
    "⊆": " subset of or equal to ",
    "∪": " union ",
    "∩": " intersect ",
    "∅": " the empty set ",
    "∀": " for all ",
    "∃": " there exists ",
    "¬": " not ",
    "∧": " and ",
    "∨": " or ",
    "→": " goes to ",
    "←": " from ",
    "⇒": " implies ",
    "⇔": " if and only if ",
    "°": " degrees ",
    "′": " prime ",
    "″": " double prime ",
    "…": " and so on ",
    "⋯": " and so on ",
}

ORDINAL_ROOTS = {2: "square", 3: "cube"}


def _sqrt(node, l2tobj) -> str:
    args = node.nodeargd.argnlist
    body = l2tobj.nodelist_to_text([args[1]])
    if args[0] is None:
        return f" the square root of {body} "
    degree = l2tobj.nodelist_to_text([args[0]]).strip()
    root = ORDINAL_ROOTS.get(_as_int(degree), "")
    if root:
        return f" the {root} root of {body} "
    return f" the {degree}th root of {body} "


def _as_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except ValueError:
        return None


SPOKEN_MACROS = [
    MacroTextSpec("frac", simplify_repl="%s over %s"),
    MacroTextSpec("dfrac", simplify_repl="%s over %s"),
    MacroTextSpec("tfrac", simplify_repl="%s over %s"),
    MacroTextSpec("binom", simplify_repl="%s choose %s"),
    MacroTextSpec("sqrt", simplify_repl=_sqrt),
]

# a spoken form only fires if the walker also knows the macro's arg signature
EXTRA_MACRO_SIGNATURES = [macrospec.MacroSpec("binom", "{{")]

_CONTEXT = get_default_latex_context_db()
_CONTEXT.add_context_category("kokoro-tts", macros=SPOKEN_MACROS, prepend=True)
_CONVERTER = LatexNodes2Text(latex_context=_CONTEXT, math_mode="text")

_WALKER_CONTEXT = latexwalker.get_default_latex_context_db()
_WALKER_CONTEXT.add_context_category(
    "kokoro-tts", macros=EXTRA_MACRO_SIGNATURES, prepend=True
)


def _speak_char(char: str) -> str:
    if char in MATH_SYMBOLS:
        return MATH_SYMBOLS[char]
    name = unicodedata.name(char, "")
    if name.startswith("GREEK"):
        return f" {name.split()[-1].lower()} "
    return char


def _handle_big_op_limits(m: re.Match[str]) -> str:
    lower, upper = m.group(1), m.group(2)
    if upper is None:
        return f" over {lower} "
    return f" from {lower} to {upper} "


def _handle_superscript(m: re.Match[str]) -> str:
    exponent = m.group(1)
    if exponent == "2":
        return " squared "
    if exponent == "3":
        return " cubed "
    return f" to the power of {exponent} "


def _speak_math(latex: str) -> str:
    text = _CONVERTER.latex_to_text(latex, latex_context=_WALKER_CONTEXT)
    text = BIG_OP_LIMITS.sub(_handle_big_op_limits, text)
    text = SUPERSCRIPT.sub(_handle_superscript, text)
    text = SUBSCRIPT.sub(lambda m: f" sub {m.group(1)} ", text)
    text = COMBINING.sub("", unicodedata.normalize("NFKD", text))
    return "".join(_speak_char(c) for c in text)


def normalize_latex(text: str) -> str:
    """Replace delimited LaTeX math with a spoken rendering."""

    def replace(m: re.Match[str]) -> str:
        try:
            spoken = _speak_math(m.group(0))
        except Exception:
            return m.group(0)
        # an arg-count mismatch leaves the raw %s placeholder behind
        if "%s" in spoken:
            return m.group(0)
        spoken = re.sub(r"\s+", " ", spoken).strip()
        return f" {spoken} " if spoken else " "

    return re.sub(r"  +", " ", MATH_SPAN.sub(replace, text))
