"""
Lazy import for the Configuration class to avoid loading the entire openapi_support package.
"""

from ..config.openapi_configuration import Configuration

__all__ = ["Configuration"]
