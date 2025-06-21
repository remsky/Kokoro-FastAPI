"""Tests for text normalization service"""

import pytest

from api.src.services.text_processing.normalizer import normalize_text
from api.src.structures.schemas import NormalizationOptions


def test_url_protocols():
    """Test URL protocol handling"""
    assert (
        normalize_text(
            "Check out https://example.com",
            normalization_options=NormalizationOptions(),
        )
        == "Check out https example dot com"
    )
    assert (
        normalize_text(
            "Visit http://site.com", normalization_options=NormalizationOptions()
        )
        == "Visit http site dot com"
    )
    assert (
        normalize_text(
            "Go to https://test.org/path", normalization_options=NormalizationOptions()
        )
        == "Go to https test dot org slash path"
    )


def test_url_www():
    """Test www prefix handling"""
    assert (
        normalize_text(
            "Go to www.example.com", normalization_options=NormalizationOptions()
        )
        == "Go to www example dot com"
    )
    assert (
        normalize_text(
            "Visit www.test.org/docs", normalization_options=NormalizationOptions()
        )
        == "Visit www test dot org slash docs"
    )
    assert (
        normalize_text(
            "Check www.site.com?q=test", normalization_options=NormalizationOptions()
        )
        == "Check www site dot com question-mark q equals test"
    )


def test_url_localhost():
    """Test localhost URL handling"""
    assert (
        normalize_text(
            "Running on localhost:7860", normalization_options=NormalizationOptions()
        )
        == "Running on localhost colon seventy-eight sixty"
    )
    assert (
        normalize_text(
            "Server at localhost:8080/api", normalization_options=NormalizationOptions()
        )
        == "Server at localhost colon eighty eighty slash api"
    )
    assert (
        normalize_text(
            "Test localhost:3000/test?v=1", normalization_options=NormalizationOptions()
        )
        == "Test localhost colon three thousand slash test question-mark v equals one"
    )


def test_url_ip_addresses():
    """Test IP address URL handling"""
    assert (
        normalize_text(
            "Access 0.0.0.0:9090/test", normalization_options=NormalizationOptions()
        )
        == "Access zero dot zero dot zero dot zero colon ninety ninety slash test"
    )
    assert (
        normalize_text(
            "API at 192.168.1.1:8000", normalization_options=NormalizationOptions()
        )
        == "API at one hundred and ninety-two dot one hundred and sixty-eight dot one dot one colon eight thousand"
    )
    assert (
        normalize_text("Server 127.0.0.1", normalization_options=NormalizationOptions())
        == "Server one hundred and twenty-seven dot zero dot zero dot one"
    )


def test_url_raw_domains():
    """Test raw domain handling"""
    assert (
        normalize_text(
            "Visit google.com/search", normalization_options=NormalizationOptions()
        )
        == "Visit google dot com slash search"
    )
    assert (
        normalize_text(
            "Go to example.com/path?q=test",
            normalization_options=NormalizationOptions(),
        )
        == "Go to example dot com slash path question-mark q equals test"
    )
    assert (
        normalize_text(
            "Check docs.test.com", normalization_options=NormalizationOptions()
        )
        == "Check docs dot test dot com"
    )


def test_url_email_addresses():
    """Test email address handling"""
    assert (
        normalize_text(
            "Email me at user@example.com", normalization_options=NormalizationOptions()
        )
        == "Email me at user at example dot com"
    )
    assert (
        normalize_text(
            "Contact admin@test.org", normalization_options=NormalizationOptions()
        )
        == "Contact admin at test dot org"
    )
    assert (
        normalize_text(
            "Send to test.user@site.com", normalization_options=NormalizationOptions()
        )
        == "Send to test dot user at site dot com"
    )


def test_money():
    """Test that money text is normalized correctly"""
    assert (
        normalize_text(
            "He lost $5.3 thousand.", normalization_options=NormalizationOptions()
        )
        == "He lost five point three thousand dollars."
    )

    assert (
        normalize_text(
            "He went gambling and lost about $25.05k.",
            normalization_options=NormalizationOptions(),
        )
        == "He went gambling and lost about twenty-five point zero five thousand dollars."
    )

    assert (
        normalize_text(
            "To put it weirdly -$6.9 million",
            normalization_options=NormalizationOptions(),
        )
        == "To put it weirdly minus six point nine million dollars"
    )

    assert (
        normalize_text("It costs $50.3.", normalization_options=NormalizationOptions())
        == "It costs fifty dollars and thirty cents."
    )

    assert (
        normalize_text(
            "The plant cost $200,000.8.", normalization_options=NormalizationOptions()
        )
        == "The plant cost two hundred thousand dollars and eighty cents."
    )

    assert (
        normalize_text(
            "Your shopping spree cost $674.03!", normalization_options=NormalizationOptions()
        )
        == "Your shopping spree cost six hundred and seventy-four dollars and three cents!"
    )

    assert (
        normalize_text(
            "€30.2 is in euros", normalization_options=NormalizationOptions()
        )
        == "thirty euros and twenty cents is in euros"
    )


def test_time():
    """Test time normalization"""

    assert (
        normalize_text(
            "Your flight leaves at 10:35 pm",
            normalization_options=NormalizationOptions(),
        )
        == "Your flight leaves at ten thirty-five pm"
    )

    assert (
        normalize_text(
            "He departed for london around 5:03 am.",
            normalization_options=NormalizationOptions(),
        )
        == "He departed for london around five oh three am."
    )

    assert (
        normalize_text(
            "Only the 13:42 and 15:12 slots are available.",
            normalization_options=NormalizationOptions(),
        )
        == "Only the thirteen forty-two and fifteen twelve slots are available."
    )

    assert (
        normalize_text(
            "It is currently 1:00 pm", normalization_options=NormalizationOptions()
        )
        == "It is currently one pm"
    )

    assert (
        normalize_text(
            "It is currently 3:00", normalization_options=NormalizationOptions()
        )
        == "It is currently three o'clock"
    )

    assert (
        normalize_text(
            "12:00 am is midnight", normalization_options=NormalizationOptions()
        )
        == "twelve am is midnight"
    )


def test_number():
    """Test number normalization"""

    assert (
        normalize_text(
            "I bought 1035 cans of soda", normalization_options=NormalizationOptions()
        )
        == "I bought one thousand and thirty-five cans of soda"
    )

    assert (
        normalize_text(
            "The bus has a maximum capacity of 62 people",
            normalization_options=NormalizationOptions(),
        )
        == "The bus has a maximum capacity of sixty-two people"
    )

    assert (
        normalize_text(
            "There are 1300 products left in stock",
            normalization_options=NormalizationOptions(),
        )
        == "There are one thousand, three hundred products left in stock"
    )

    assert (
        normalize_text(
            "The population is 7,890,000 people.",
            normalization_options=NormalizationOptions(),
        )
        == "The population is seven million, eight hundred and ninety thousand people."
    )

    assert (
        normalize_text(
            "He looked around but only found 1.6k of the 10k bricks",
            normalization_options=NormalizationOptions(),
        )
        == "He looked around but only found one point six thousand of the ten thousand bricks"
    )

    assert (
        normalize_text(
            "The book has 342 pages.", normalization_options=NormalizationOptions()
        )
        == "The book has three hundred and forty-two pages."
    )

    assert (
        normalize_text(
            "He made -50 sales today.", normalization_options=NormalizationOptions()
        )
        == "He made minus fifty sales today."
    )

    assert (
        normalize_text(
            "56.789 to the power of 1.35 million",
            normalization_options=NormalizationOptions(),
        )
        == "fifty-six point seven eight nine to the power of one point three five million"
    )


def test_non_url_text():
    """Test that non-URL text is unaffected"""
    assert (
        normalize_text(
            "This is not.a.url text", normalization_options=NormalizationOptions()
        )
        == "This is not-a-url text"
    )
    assert (
        normalize_text(
            "Hello, how are you today?", normalization_options=NormalizationOptions()
        )
        == "Hello, how are you today?"
    )
    assert (
        normalize_text("It costs $50.", normalization_options=NormalizationOptions())
        == "It costs fifty dollars."
    )

def test_remaining_symbol():
    """Test that remaining symbols are replaced"""
    assert (
        normalize_text(
            "I love buying products @ good store here & @ other store", normalization_options=NormalizationOptions()
        )
        == "I love buying products at good store here and at other store"
    )
