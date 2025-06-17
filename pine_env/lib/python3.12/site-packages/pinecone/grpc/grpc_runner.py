from functools import wraps
from typing import Dict, Tuple, Optional

from grpc._channel import _InactiveRpcError

from pinecone import Config
from .utils import _generate_request_id
from .config import GRPCClientConfig
from pinecone.utils.constants import REQUEST_ID, CLIENT_VERSION
from pinecone.exceptions.exceptions import PineconeException
from grpc import CallCredentials, Compression
from google.protobuf.message import Message


class GrpcRunner:
    def __init__(self, index_name: str, config: Config, grpc_config: GRPCClientConfig):
        self.config = config
        self.grpc_client_config = grpc_config

        self.fixed_metadata = {
            "api-key": config.api_key,
            "service-name": index_name,
            "client-version": CLIENT_VERSION,
        }
        if self.grpc_client_config.additional_metadata:
            self.fixed_metadata.update(self.grpc_client_config.additional_metadata)

    def run(
        self,
        func,
        request: Message,
        timeout: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
        credentials: Optional[CallCredentials] = None,
        wait_for_ready: Optional[bool] = None,
        compression: Optional[Compression] = None,
    ):
        @wraps(func)
        def wrapped():
            user_provided_metadata = metadata or {}
            _metadata = self._prepare_metadata(user_provided_metadata)
            try:
                return func(
                    request,
                    timeout=timeout,
                    metadata=_metadata,
                    credentials=credentials,
                    wait_for_ready=wait_for_ready,
                    compression=compression,
                )
            except _InactiveRpcError as e:
                raise PineconeException(e._state.debug_error_string) from e

        return wrapped()

    async def run_asyncio(
        self,
        func,
        request: Message,
        timeout: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
        credentials: Optional[CallCredentials] = None,
        wait_for_ready: Optional[bool] = None,
        compression: Optional[Compression] = None,
    ):
        @wraps(func)
        async def wrapped():
            user_provided_metadata = metadata or {}
            _metadata = self._prepare_metadata(user_provided_metadata)
            try:
                return await func(
                    request,
                    timeout=timeout,
                    metadata=_metadata,
                    credentials=credentials,
                    wait_for_ready=wait_for_ready,
                    compression=compression,
                )
            except _InactiveRpcError as e:
                raise PineconeException(e._state.debug_error_string) from e

        return await wrapped()

    def _prepare_metadata(
        self, user_provided_metadata: Dict[str, str]
    ) -> Tuple[Tuple[str, str], ...]:
        return tuple(
            (k, v)
            for k, v in {
                **self.fixed_metadata,
                **self._request_metadata(),
                **user_provided_metadata,
            }.items()
        )

    def _request_metadata(self) -> Dict[str, str]:
        return {REQUEST_ID: _generate_request_id()}
