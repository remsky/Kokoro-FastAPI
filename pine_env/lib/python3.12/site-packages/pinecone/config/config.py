from typing import NamedTuple, Optional, Dict, TYPE_CHECKING
import os

from pinecone.exceptions import PineconeConfigurationError
from pinecone.config.openapi_config_factory import OpenApiConfigFactory

if TYPE_CHECKING:
    from pinecone.config.openapi_configuration import Configuration as OpenApiConfiguration


# Duplicated this util to help resolve circular imports
def normalize_host(host: Optional[str]) -> str:
    if host is None:
        return ""
    if host.startswith("https://"):
        return host
    if host.startswith("http://"):
        return host
    return "https://" + host


class Config(NamedTuple):
    api_key: str = ""
    host: str = ""
    proxy_url: Optional[str] = None
    proxy_headers: Optional[Dict[str, str]] = None
    ssl_ca_certs: Optional[str] = None
    ssl_verify: Optional[bool] = None
    additional_headers: Optional[Dict[str, str]] = {}
    source_tag: Optional[str] = None


class ConfigBuilder:
    """

    Configurations are resolved in the following order:

    - configs passed as keyword parameters
    - configs specified in environment variables
    - default values (if applicable)
    """

    """Initializes the Pinecone client.

    :param api_key: Required if not set in config file or by environment variable ``PINECONE_API_KEY``.
    :param host: Optional. Controller host.
    :param openapi_config: Optional. Set OpenAPI client configuration.
    """

    @staticmethod
    def build(
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        proxy_url: Optional[str] = None,
        proxy_headers: Optional[Dict[str, str]] = None,
        ssl_ca_certs: Optional[str] = None,
        ssl_verify: Optional[bool] = None,
        additional_headers: Optional[Dict[str, str]] = {},
        **kwargs,
    ) -> Config:
        api_key = api_key or kwargs.pop("api_key", None) or os.getenv("PINECONE_API_KEY")
        host = host or kwargs.pop("host", None)
        host = normalize_host(host)
        source_tag = kwargs.pop("source_tag", None)

        if not api_key:
            raise PineconeConfigurationError(
                "You haven't specified an API key. Please either set the PINECONE_API_KEY environment variable or pass the 'api_key' keyword argument to the Pinecone client constructor."
            )
        if not host:
            raise PineconeConfigurationError("You haven't specified a host.")

        return Config(
            api_key,
            host,
            proxy_url,
            proxy_headers,
            ssl_ca_certs,
            ssl_verify,
            additional_headers,
            source_tag,
        )

    @staticmethod
    def build_openapi_config(
        config: Config, openapi_config: Optional["OpenApiConfiguration"] = None, **kwargs
    ) -> "OpenApiConfiguration":
        if openapi_config:
            openapi_config = OpenApiConfigFactory.copy(
                openapi_config=openapi_config, api_key=config.api_key, host=config.host
            )
        elif openapi_config is None:
            openapi_config = OpenApiConfigFactory.build(api_key=config.api_key, host=config.host)

        # Check if value passed before overriding any values present
        # in the openapi_config. This means if the user has passed
        # an openapi_config object and a kwarg for the same setting,
        # the kwarg will take precedence.
        if config.proxy_url:
            openapi_config.proxy = config.proxy_url
        if config.proxy_headers:
            openapi_config.proxy_headers = config.proxy_headers
        if config.ssl_ca_certs:
            openapi_config.ssl_ca_cert = config.ssl_ca_certs
        if config.ssl_verify is not None:
            openapi_config.verify_ssl = config.ssl_verify

        return openapi_config
