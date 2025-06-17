from typing import Optional, Dict
import logging
import json
import os
from .config import ConfigBuilder, Config

logger = logging.getLogger(__name__)
""" @private """

DEFAULT_CONTROLLER_HOST = "https://api.pinecone.io"


class PineconeConfig:
    @staticmethod
    def build(
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        additional_headers: Optional[Dict[str, str]] = {},
        **kwargs,
    ) -> Config:
        host = (
            host
            or kwargs.get("host")
            or os.getenv("PINECONE_CONTROLLER_HOST")
            or DEFAULT_CONTROLLER_HOST
        )
        headers_json = os.getenv("PINECONE_ADDITIONAL_HEADERS")
        if headers_json:
            try:
                headers = json.loads(headers_json)
                additional_headers = additional_headers or headers
            except Exception as e:
                logger.warn(f"Ignoring PINECONE_ADDITIONAL_HEADERS: {e}")

        return ConfigBuilder.build(
            api_key=api_key, host=host, additional_headers=additional_headers, **kwargs
        )
