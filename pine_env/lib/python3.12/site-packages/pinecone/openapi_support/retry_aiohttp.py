import random
from typing import Optional
from aiohttp_retry import RetryOptionsBase, EvaluateResponseCallbackType, ClientResponse
import logging

logger = logging.getLogger(__name__)


class JitterRetry(RetryOptionsBase):
    """https://github.com/inyutin/aiohttp_retry/issues/44."""

    def __init__(
        self,
        attempts: int = 3,  # How many times we should retry
        start_timeout: float = 0.1,  # Base timeout time, then it exponentially grow
        max_timeout: float = 5.0,  # Max possible timeout between tries
        statuses: Optional[set[int]] = None,  # On which statuses we should retry
        exceptions: Optional[set[type[Exception]]] = None,  # On which exceptions we should retry
        methods: Optional[set[str]] = None,  # On which HTTP methods we should retry
        retry_all_server_errors: bool = True,
        evaluate_response_callback: Optional[EvaluateResponseCallbackType] = None,
    ) -> None:
        super().__init__(
            attempts=attempts,
            statuses=statuses,
            exceptions=exceptions,
            methods=methods,
            retry_all_server_errors=retry_all_server_errors,
            evaluate_response_callback=evaluate_response_callback,
        )

        self._start_timeout: float = start_timeout
        self._max_timeout: float = max_timeout

    def get_timeout(
        self,
        attempt: int,
        response: Optional[ClientResponse] = None,  # noqa: ARG002
    ) -> float:
        logger.debug(f"JitterRetry get_timeout: attempt={attempt}, response={response}")
        """Return timeout with exponential backoff."""
        jitter = random.uniform(0, 0.1)
        timeout = self._start_timeout * (2 ** (attempt - 1))
        return min(timeout + jitter, self._max_timeout)
