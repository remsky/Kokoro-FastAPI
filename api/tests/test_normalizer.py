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
        == "Running on localhost colon 78 60"
    )
    assert (
        normalize_text(
            "Server at localhost:8080/api", normalization_options=NormalizationOptions()
        )
        == "Server at localhost colon 80 80 slash api"
    )
    assert (
        normalize_text(
            "Test localhost:3000/test?v=1", normalization_options=NormalizationOptions()
        )
        == "Test localhost colon 3000 slash test question-mark v equals 1"
    )


def test_url_ip_addresses():
    """Test IP address URL handling"""
    assert (
        normalize_text(
            "Access 0.0.0.0:9090/test", normalization_options=NormalizationOptions()
        )
        == "Access 0 dot 0 dot 0 dot 0 colon 90 90 slash test"
    )
    assert (
        normalize_text(
            "API at 192.168.1.1:8000", normalization_options=NormalizationOptions()
        )
        == "API at 192 dot 168 dot 1 dot 1 colon 8000"
    )
    assert (
        normalize_text("Server 127.0.0.1", normalization_options=NormalizationOptions())
        == "Server 127 dot 0 dot 0 dot 1"
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
            "To put it weirdly -$6.9 million",
            normalization_options=NormalizationOptions(),
        )
        == "To put it weirdly minus six point nine million dollars"
    )
    assert (
        normalize_text("It costs $50.3.", normalization_options=NormalizationOptions())
        == "It costs fifty dollars and thirty cents."
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
