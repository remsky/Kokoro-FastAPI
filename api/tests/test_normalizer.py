"""Tests for text normalization service"""

import pytest

from api.src.services.text_processing.normalizer import normalize_text


def test_url_protocols():
    """Test URL protocol handling"""
    assert (
        normalize_text("Check out https://example.com")
        == "Check out https example dot com"
    )
    assert normalize_text("Visit http://site.com") == "Visit http site dot com"
    assert (
        normalize_text("Go to https://test.org/path")
        == "Go to https test dot org slash path"
    )


def test_url_www():
    """Test www prefix handling"""
    assert normalize_text("Go to www.example.com") == "Go to www example dot com"
    assert (
        normalize_text("Visit www.test.org/docs") == "Visit www test dot org slash docs"
    )
    assert (
        normalize_text("Check www.site.com?q=test")
        == "Check www site dot com question-mark q equals test"
    )


def test_url_localhost():
    """Test localhost URL handling"""
    assert (
        normalize_text("Running on localhost:7860")
        == "Running on localhost colon 78 60"
    )
    assert (
        normalize_text("Server at localhost:8080/api")
        == "Server at localhost colon 80 80 slash api"
    )
    assert (
        normalize_text("Test localhost:3000/test?v=1")
        == "Test localhost colon 3000 slash test question-mark v equals 1"
    )


def test_url_ip_addresses():
    """Test IP address URL handling"""
    assert (
        normalize_text("Access 0.0.0.0:9090/test")
        == "Access 0 dot 0 dot 0 dot 0 colon 90 90 slash test"
    )
    assert (
        normalize_text("API at 192.168.1.1:8000")
        == "API at 192 dot 168 dot 1 dot 1 colon 8000"
    )
    assert normalize_text("Server 127.0.0.1") == "Server 127 dot 0 dot 0 dot 1"


def test_url_raw_domains():
    """Test raw domain handling"""
    assert (
        normalize_text("Visit google.com/search") == "Visit google dot com slash search"
    )
    assert (
        normalize_text("Go to example.com/path?q=test")
        == "Go to example dot com slash path question-mark q equals test"
    )
    assert normalize_text("Check docs.test.com") == "Check docs dot test dot com"


def test_url_email_addresses():
    """Test email address handling"""
    assert (
        normalize_text("Email me at user@example.com")
        == "Email me at user at example dot com"
    )
    assert normalize_text("Contact admin@test.org") == "Contact admin at test dot org"
    assert (
        normalize_text("Send to test.user@site.com")
        == "Send to test dot user at site dot com"
    )


def test_non_url_text():
    """Test that non-URL text is unaffected"""
    assert normalize_text("This is not.a.url text") == "This is not-a-url text"
    assert normalize_text("Hello, how are you today?") == "Hello, how are you today?"
    assert normalize_text("It costs $50.") == "It costs 50 dollars."
