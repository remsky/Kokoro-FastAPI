---
name: adding-normalizer
description: "Adding a language normalizer to Kokoro-FastAPI: the Normalizer contract, registry, and test expectations. Use when adding or changing text normalization under api/src/services/text_processing/normalization/."
---

Read [AGENTS.md](../../../AGENTS.md) and [CONTRIBUTING.md](../../../CONTRIBUTING.md) first, house rules live there.

# Adding a language normalizer

Text normalization (numbers, money, time, URLs, etc. spelled out before phonemization) is per language. Only English is implemented; other languages skip normalization and rely on the phonemizer.

## Where things live

`api/src/services/text_processing/normalization/`:

- `base.py`: `Normalizer`, the contract. `pass_order()` is the pass order, `normalize()` runs it. The language-neutral passes (quotes, CJK punctuation, whitespace) are implemented; every other pass is a hook that returns its input unchanged until overridden. Override `pass_order()` if a language needs a different order.
- `english.py`: `EnglishNormalizer`, the reference implementation.
- `english_data.py`: the English tables, TLDs, units, symbol readings, currencies.
- `__init__.py`: `LANGUAGES`, the registry. Lang codes come from `lang_codes` on each class.

## Adding one

1. Create `normalization/<language>.py` with a `Normalizer` subclass. Set `lang_codes` to the Kokoro pipeline codes it serves (`p` for Portuguese, `e` for Spanish, etc.). Override the hooks you need, `numbers` first (money is spelled out there too); espeak already reads bare digits, so the value is in currency, time, and units. `num2words` is in the lock and covers cardinals for most languages.
2. Add an instance to `LANGUAGES` in `__init__.py`.
3. Add `api/tests/test_normalizer_<language>.py` with a `CASES` table of `(input, expected)` rows, mirroring `test_normalizer.py`. `test_normalizer_contract.py` runs the never-raises and idempotence properties on every registered language automatically.

## Testing

- Keep regexes linear. `test_normalizer_redos.py` shows the flood inputs that have bitten before.
- `uv run pytest api/tests/test_normalizer*.py`, then `ruff format .` and `ruff check . --fix` before staging.
