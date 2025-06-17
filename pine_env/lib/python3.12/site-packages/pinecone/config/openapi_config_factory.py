import sys
from typing import List, Optional, Tuple

import certifi
import socket
import copy

from pinecone.config.openapi_configuration import Configuration as OpenApiConfiguration

TCP_KEEPINTVL = 60  # Sec
TCP_KEEPIDLE = 300  # Sec
TCP_KEEPCNT = 4


class OpenApiConfigFactory:
    @classmethod
    def build(cls, api_key: str, host: Optional[str] = None, **kwargs):
        openapi_config = OpenApiConfiguration()
        openapi_config.api_key = {"ApiKeyAuth": api_key}
        openapi_config.host = host
        openapi_config.ssl_ca_cert = certifi.where()
        openapi_config.socket_options = cls._get_socket_options()
        openapi_config.discard_unknown_keys = True

        return openapi_config

    @classmethod
    def copy(
        cls, openapi_config: OpenApiConfiguration, api_key: str, host: str
    ) -> OpenApiConfiguration:
        """
        Copy a user-supplied openapi configuration and update it with the user's api key and host.
        If they have not specified other socket configuration, we will use the default values.
        We expect these objects are being passed mainly a vehicle for proxy configuration, so
        we don't modify those settings.
        """
        copied = copy.deepcopy(openapi_config)

        copied.api_key = {"ApiKeyAuth": api_key}
        copied.host = host

        # Set sensible defaults if the user hasn't set them
        if not copied.socket_options:
            copied.socket_options = cls._get_socket_options()

        # We specifically do not modify the user's ssl_ca_cert or proxy settings, as
        # they may have set them intentionally. This is the main reason somebody would
        # pass an openapi_config in the first place.

        return copied

    @classmethod
    def _get_socket_options(
        self,
        do_keep_alive: bool = True,
        keep_alive_idle_sec: int = TCP_KEEPIDLE,
        keep_alive_interval_sec: int = TCP_KEEPINTVL,
        keep_alive_tries: int = TCP_KEEPCNT,
    ) -> List[Tuple[int, int, int]]:
        """
        Returns the socket options to pass to OpenAPI's Rest client
        Args:
            do_keep_alive: Whether to enable TCP keep alive mechanism
            keep_alive_idle_sec: Time in seconds of connection idleness before starting to send keep alive probes
            keep_alive_interval_sec: Interval time in seconds between keep alive probe messages
            keep_alive_tries: Number of failed keep alive tries (unanswered KA messages) before terminating the connection

        Returns:
            A list of socket options for the Rest client's connection pool
        """
        # Source: https://www.finbourne.com/blog/the-mysterious-hanging-client-tcp-keep-alives

        # urllib3.connection.HTTPConnection.default_socket_options
        socket_params = [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]
        if not do_keep_alive:
            return socket_params

        socket_params += [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]

        # TCP Keep Alive Probes for different platforms
        platform = sys.platform
        # TCP Keep Alive Probes for Linux
        if (
            platform == "linux"
            and hasattr(socket, "TCP_KEEPIDLE")
            and hasattr(socket, "TCP_KEEPINTVL")
            and hasattr(socket, "TCP_KEEPCNT")
        ):
            socket_params += [(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, keep_alive_idle_sec)]
            socket_params += [(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, keep_alive_interval_sec)]
            socket_params += [(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, keep_alive_tries)]

        # TCP Keep Alive Probes for Windows OS
        # NOTE: Changing TCP KA params on windows is done via a different mechanism which OpenAPI's Rest client doesn't expose.
        # Since the default values work well, it seems setting `(socket.SO_KEEPALIVE, 1)` is sufficient.
        # Leaving this code here for future reference.
        # elif platform == 'win32' and hasattr(socket, "SIO_KEEPALIVE_VALS"):
        #     socket.ioctl((socket.SIO_KEEPALIVE_VALS, (1, keep_alive_idle_sec * 1000, keep_alive_interval_sec * 1000)))

        # TCP Keep Alive Probes for Mac OS
        elif platform == "darwin":
            TCP_KEEPALIVE = 0x10
            socket_params += [(socket.IPPROTO_TCP, TCP_KEEPALIVE, keep_alive_interval_sec)]

        return socket_params
