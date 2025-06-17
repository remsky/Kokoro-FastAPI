from concurrent.futures import Future as ConcurrentFuture
from typing import Optional
from grpc import Future as GrpcFuture, RpcError
from pinecone.exceptions.exceptions import PineconeException


class PineconeGrpcFuture(ConcurrentFuture):
    def __init__(
        self, grpc_future: GrpcFuture, timeout: Optional[int] = None, result_transformer=None
    ):
        super().__init__()
        self._grpc_future = grpc_future
        self._result_transformer = result_transformer
        if timeout is not None:
            self._default_timeout = timeout  # seconds
        else:
            self._default_timeout = 5  # seconds

        # Sync initial state, in case the gRPC future is already done
        self._sync_state(self._grpc_future)

        # Add callback to subscribe to updates from the gRPC future
        self._grpc_future.add_done_callback(self._sync_state)

    @property
    def grpc_future(self):
        return self._grpc_future

    def _sync_state(self, grpc_future):
        if self.done():
            return

        if grpc_future.running() and not self.running():
            if not self.set_running_or_notify_cancel():
                grpc_future.cancel()
        elif grpc_future.cancelled():
            self.cancel()
        elif grpc_future.done():
            try:
                result = grpc_future.result(timeout=self._default_timeout)
                self.set_result(result)
            except Exception as e:
                self.set_exception(e)

    def set_result(self, result):
        if self._result_transformer:
            result = self._result_transformer(result)
        return super().set_result(result)

    def cancel(self):
        self._grpc_future.cancel()
        return super().cancel()

    def exception(self, timeout=None):
        exception = super().exception(timeout=self._timeout(timeout))
        if isinstance(exception, RpcError):
            return self._wrap_rpc_exception(exception)
        return exception

    def traceback(self, timeout=None):
        # This is not part of the ConcurrentFuture interface, but keeping it for
        # backward compatibility
        return self._grpc_future.traceback(timeout=self._timeout(timeout))

    def result(self, timeout=None):
        try:
            return super().result(timeout=self._timeout(timeout))
        except RpcError as e:
            raise self._wrap_rpc_exception(e) from e

    def _timeout(self, timeout: Optional[int] = None) -> int:
        if timeout is not None:
            return timeout
        else:
            return self._default_timeout

    def _wrap_rpc_exception(self, e):
        if e._state and e._state.debug_error_string:
            return PineconeException(e._state.debug_error_string)
        else:
            return PineconeException("Unknown GRPC error")

    def __del__(self):
        self._grpc_future.cancel()
        self = None  # release the reference to the grpc future
