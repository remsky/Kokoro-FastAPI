"""Tests for Brazilian Portuguese text normalization."""

import pytest

from api.src.services.text_processing.normalization import get_normalizer
from api.src.structures.schemas import NormalizationOptions

NORMALIZER = get_normalizer("p")
OPTIONS = NormalizationOptions()
UNIT_OPTIONS = NormalizationOptions(unit_normalization=True)

CASES = [
    ("Tenho 2 livros", "Tenho dois livros", OPTIONS),
    (
        "A viagem custa R$ 12,50",
        "A viagem custa doze reais e cinquenta centavos",
        OPTIONS,
    ),
    ("O valor é US$ 12,50", "O valor é doze dólares e cinquenta centavos", OPTIONS),
    ("São 13:05", "São treze horas e cinco minutos", OPTIONS),
    ("Às 9:00", "Às nove horas", OPTIONS),
    ("Às 1:00", "Às uma hora", OPTIONS),
    ("São 21:00", "São vinte e uma horas", OPTIONS),
    ("Timestamp: 12:34:56", "Timestamp: 12:34:56", OPTIONS),
    ("São 24:75", "São 24:75", OPTIONS),
    ("Fica a 2 km", "Fica a dois km", OPTIONS),
    ("Fica a 2 km", "Fica a dois quilômetros", UNIT_OPTIONS),
    ("Pesa 1,5 kg", "Pesa um vírgula cinco quilogramas", UNIT_OPTIONS),
    (
        "O código 1234567890123456789012345678901234567890",
        "O código 1234567890123456789012345678901234567890",
        OPTIONS,
    ),
]


@pytest.mark.parametrize(("text", "expected", "options"), CASES)
def test_portuguese_normalization(text, expected, options):
    assert NORMALIZER.normalize(text, options) == expected
