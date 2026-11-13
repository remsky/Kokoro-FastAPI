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
    ("São 2:00", "São duas horas", OPTIONS),
    ("São 22:30", "São vinte e duas horas e trinta minutos", OPTIONS),
    ("Durou 1 h", "Durou uma hora", UNIT_OPTIONS),
    ("Durou 2 h", "Durou duas horas", UNIT_OPTIONS),
    ("Custa £2", "Custa duas libras", OPTIONS),
    ("Custa £200", "Custa duzentas libras", OPTIONS),
    ("Custa £2.000.000", "Custa dois milhões de libras", OPTIONS),
    (
        "Custa £2.234.201",
        "Custa dois milhões, duzentas e trinta e quatro mil, duzentas e uma libras",
        OPTIONS,
    ),
    ("Custa R$ 1.000.000,00", "Custa um milhão de reais", OPTIONS),
    ("Custa R$ 2.500.000", "Custa dois milhões e quinhentos mil reais", OPTIONS),
    ("Custa R$ 5, mas", "Custa cinco reais, mas", OPTIONS),
    ("Custa $1.50", "Custa um dólar e cinquenta centavos", OPTIONS),
    ("Custa R$ 1.500", "Custa mil e quinhentos reais", OPTIONS),
    ("Custa R$ 12,505", "Custa R$ doze vírgula cinco zero cinco", OPTIONS),
    ("Saldo -R$ 1,50", "Saldo menos um real e cinquenta centavos", OPTIONS),
    ("Subiu 50%", "Subiu cinquenta por cento", OPTIONS),
    ("Caiu -0,5", "Caiu menos zero vírgula cinco", OPTIONS),
    (
        "Vale 1,000000000000000001",
        "Vale um vírgula " + "zero " * 17 + "um",
        OPTIONS,
    ),
    ("Às 12:30h", "Às doze horas e trinta minutos", UNIT_OPTIONS),
    ("Às 12:30 h", "Às doze horas e trinta minutos", OPTIONS),
    ("Modelo abc2h", "Modelo abc2h", UNIT_OPTIONS),
    ("Atraso de -1 h", "Atraso de menos uma hora", UNIT_OPTIONS),
    ("Adiantou +1 h", "Adiantou uma hora", UNIT_OPTIONS),
    ("Saldo +R$ 5", "Saldo cinco reais", OPTIONS),
    ("Desceu -2 km", "Desceu menos dois quilômetros", UNIT_OPTIONS),
    ("Das 10-2 h", "Das dez-duas horas", UNIT_OPTIONS),
    ("Durou 1000000000000000000 h", "Durou 1000000000000000000 horas", UNIT_OPTIONS),
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
