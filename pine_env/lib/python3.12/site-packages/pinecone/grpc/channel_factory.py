import logging
from typing import Optional

import certifi
import grpc
import json

from pinecone import Config
from .config import GRPCClientConfig
from pinecone.utils.constants import MAX_MSG_SIZE
from pinecone.utils.user_agent import get_user_agent_grpc

_logger = logging.getLogger(__name__)
""" @private """


class GrpcChannelFactory:
    def __init__(
        self,
        config: Config,
        grpc_client_config: GRPCClientConfig,
        use_asyncio: Optional[bool] = False,
    ):
        self.config = config
        self.grpc_client_config = grpc_client_config
        self.use_asyncio = use_asyncio

    def _get_service_config(self):
        # https://github.com/grpc/grpc-proto/blob/master/grpc/service_config/service_config.proto
        return json.dumps(
            {
                "methodConfig": [
                    {
                        "name": [{"service": "VectorService.Upsert"}],
                        "retryPolicy": {
                            "maxAttempts": 5,
                            "initialBackoff": "0.1s",
                            "maxBackoff": "1s",
                            "backoffMultiplier": 2,
                            "retryableStatusCodes": ["UNAVAILABLE"],
                        },
                    },
                    {
                        "name": [{"service": "VectorService"}],
                        "retryPolicy": {
                            "maxAttempts": 5,
                            "initialBackoff": "0.1s",
                            "maxBackoff": "1s",
                            "backoffMultiplier": 2,
                            "retryableStatusCodes": ["UNAVAILABLE"],
                        },
                    },
                ]
            }
        )

    def _build_options(self, target):
        # For property definitions, see https://github.com/grpc/grpc/blob/v1.43.x/include/grpc/impl/codegen/grpc_types.h
        options = {
            "grpc.max_send_message_length": MAX_MSG_SIZE,
            "grpc.max_receive_message_length": MAX_MSG_SIZE,
            "grpc.service_config": self._get_service_config(),
            "grpc.enable_retries": True,
            "grpc.per_rpc_retry_buffer_size": MAX_MSG_SIZE,
            "grpc.primary_user_agent": get_user_agent_grpc(self.config),
        }
        if self.grpc_client_config.secure:
            options["grpc.ssl_target_name_override"] = target.split(":")[0]
        if self.config.proxy_url:
            options["grpc.http_proxy"] = self.config.proxy_url

        options_tuple = tuple((k, v) for k, v in options.items())
        return options_tuple

    def _build_channel_credentials(self):
        ca_certs = self.config.ssl_ca_certs if self.config.ssl_ca_certs else certifi.where()
        root_cas = open(ca_certs, "rb").read()
        channel_creds = grpc.ssl_channel_credentials(root_certificates=root_cas)
        return channel_creds

    def create_channel(self, endpoint):
        options_tuple = self._build_options(endpoint)

        _logger.debug(
            "Creating new channel with endpoint %s options %s and config %s",
            endpoint,
            options_tuple,
            self.grpc_client_config,
        )

        if not self.grpc_client_config.secure:
            create_channel_fn = (
                grpc.aio.insecure_channel if self.use_asyncio else grpc.insecure_channel
            )
            channel = create_channel_fn(endpoint, options=options_tuple)
        else:
            channel_creds = self._build_channel_credentials()
            create_channel_fn = grpc.aio.secure_channel if self.use_asyncio else grpc.secure_channel
            channel = create_channel_fn(endpoint, credentials=channel_creds, options=options_tuple)

        return channel
