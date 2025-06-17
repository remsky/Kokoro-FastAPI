from abc import ABC, abstractmethod
from typing import Optional

import grpc
from grpc._channel import Channel

from .channel_factory import GrpcChannelFactory

from pinecone import Config
from .config import GRPCClientConfig
from .grpc_runner import GrpcRunner
from concurrent.futures import ThreadPoolExecutor


class GRPCIndexBase(ABC):
    """
    Base class for grpc-based interaction with Pinecone indexes
    """

    _pool = None

    def __init__(
        self,
        index_name: str,
        config: Config,
        channel: Optional[Channel] = None,
        grpc_config: Optional[GRPCClientConfig] = None,
        pool_threads: Optional[int] = None,
        _endpoint_override: Optional[str] = None,
    ):
        self.config = config
        self.grpc_client_config = grpc_config or GRPCClientConfig()
        self.pool_threads = pool_threads

        self._endpoint_override = _endpoint_override

        self.runner = GrpcRunner(
            index_name=index_name, config=config, grpc_config=self.grpc_client_config
        )
        self.channel_factory = GrpcChannelFactory(
            config=self.config, grpc_client_config=self.grpc_client_config, use_asyncio=False
        )
        self._channel = channel or self._gen_channel()
        self.stub = self.stub_class(self._channel)

    @property
    def threadpool_executor(self):
        if self._pool is None:
            pt = self.pool_threads or 10
            self._pool = ThreadPoolExecutor(max_workers=pt)
        return self._pool

    @property
    @abstractmethod
    def stub_class(self):
        pass

    def _endpoint(self):
        grpc_host = self.config.host.replace("https://", "")
        if ":" not in grpc_host:
            grpc_host = f"{grpc_host}:443"
        return self._endpoint_override if self._endpoint_override else grpc_host

    def _gen_channel(self):
        return self.channel_factory.create_channel(self._endpoint())

    @property
    def channel(self):
        """Creates GRPC channel."""
        if self.grpc_client_config.reuse_channel and self._channel and self.grpc_server_on():
            return self._channel
        self._channel = self._gen_channel()
        return self._channel

    def grpc_server_on(self) -> bool:
        try:
            grpc.channel_ready_future(self._channel).result(
                timeout=self.grpc_client_config.conn_timeout
            )
            return True
        except grpc.FutureTimeoutError:
            return False

    def close(self):
        """Closes the connection to the index."""
        try:
            self._channel.close()
        except TypeError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
