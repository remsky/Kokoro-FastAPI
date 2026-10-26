"""Tests for Brazilian Portuguese text normalization."""

import pytest

from api.src.services.text_processing.normalization import get_normalizer
from api.src.structures.schemas import NormalizationOptions

NORMALIZER = get_normalizer("p")
OPTIONS = NormalizationOptions()

CASES = [
    ("Tenho 2 livros", "Tenho dois livros"),
    ("A viagem custa R$ 12,50", "A viagem custa doze reais e cinquenta centavos"),
    ("O valor é US$ 12,50", "O valor é doze dólares e cinquenta centavos"),
    ("São 13:05", "São treze horas e cinco minutos"),
    ("Às 9:00", "Às nove horas"),
    ("São 24:75", "São 24:75"),
    ("Fica a 2 km", "Fica a dois quilômetros"),
    ("Pesa 1,5 kg", "Pesa um vírgula cinco quilogramas"),
    (
        "O código 1234567890123456789012345678901234567890",
        "O código 1234567890123456789012345678901234567890",
    ),
]


@pytest.mark.parametrize(("text", "expected"), CASES)
def test_portuguese_normalization(text, expected):
    assert NORMALIZER.normalize(text, OPTIONS) == expected
