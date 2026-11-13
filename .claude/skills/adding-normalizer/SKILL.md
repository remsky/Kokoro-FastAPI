---
name: adding-normalizer
description: "Adding a language normalizer to Kokoro-FastAPI: the Normalizer contract, registry, and test expectations. Use when adding or changing text normalization under api/src/services/text_processing/normalization/."
---

Read [AGENTS.md](../../../AGENTS.md) and [CONTRIBUTING.md](../../../CONTRIBUTING.md) first, house rules live there.

# Adding a language normalizer

Text normalization (numbers, money, time, URLs, etc. spelled out before phonemization) is per language. English and Brazilian Portuguese are implemented; other languages skip normalization and rely on the phonemizer.

## Where things live

`api/src/services/text_processing/normalization/`:

- `base.py`: `Normalizer`, the contract. `pass_order()` is the pass order, `normalize()` runs it. The language-neutral passes (quotes, CJK punctuation, whitespace) are implemented; every other pass is a hook that returns its input unchanged until overridden. Override `pass_order()` if a language needs a different order.
- `english.py`: `EnglishNormalizer`, the reference implementation.
- `english_data.py`: the English tables, TLDs, units, symbol readings, currencies.
- `portuguese.py`: `PortugueseNormalizer`, a small second language (numbers, currency, times, units). The closer template for a new one.
- `__init__.py`: `LANGUAGES`, the registry. Lang codes come from `lang_codes` on each class.

## Adding one

1. Create `normalization/<language>.py` with a `Normalizer` subclass. Set `lang_codes` to the Kokoro pipeline codes it serves (`p` for Portuguese, `e` for Spanish, etc.). Override the hooks you need, `numbers` first (money is spelled out there too); espeak already reads bare digits, so the value is in currency, time, and units. `num2words` is in the lock and covers cardinals for most languages.
2. Add an instance to `LANGUAGES` in `__init__.py`.
3. Add `api/tests/test_normalizer_<language>.py` with a `CASES` table of `(input, expected, options)` rows, mirroring `test_normalizer_portuguese.py`. `test_normalizer_contract.py` runs the never-raises and idempotence properties on every registered language automatically.

## Gotchas

- Passes run in order (`units` only with `unit_normalization`, then `times`, then `numbers`), and an early pass hides text from the later ones. Each pattern that takes digits owns its sign and its left boundary (`(?<![\w:])`), or `-1 h` keeps a bare `-` and `abc2h` / `12:30h` get read as units.
- Reject malformed input whole (a trailing lookahead like `(?!\d|[.,]\d)`), so it falls through to the plain number pass. A half match reads worse than no match.
- Bound every digit quantifier (`\d{1,30}`, `(?:\.\d{3}){1,9}`). It keeps the regex linear, and long runs are left for the phonemizer.
- `num2words` raises `OverflowError` on large values (from 10^18 in `pt_BR`, limits vary by language). Catch it and return the digits unchanged, in every pass that calls it.
- Pass `num2words` an int. Its Decimal path goes through float, losing precision and the sign between -1 and 0. Spell the fraction digit by digit and add the sign word yourself.
- `num2words` has no grammatical gender and punctuates scale words its own way (`dois milhões, duzentos`). Agreement (`duas horas`, `duzentas libras`) is post-processing on its output, so strip punctuation when matching tokens.
- Decide what happens to the other locale's format (`$1.50` in Portuguese text) and pin it in a test.

## Testing

- Keep regexes linear. `test_normalizer_redos.py` shows the flood inputs that have bitten before. Add a flood unit there for each new pattern, with the option that enables its pass.
- The contract properties run with default options, so option-gated passes (units) are only covered by rows in the language's own table. For each pattern include a sign, a huge number, and a neighbor that must stay untouched (clock, identifier).
- `uv run pytest api/tests/test_normalizer*.py`, then `uv run --only-group lint ruff format .` and `uv run --only-group lint ruff check . --fix` before staging.
