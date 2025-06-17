import random
from urllib3.util.retry import Retry
import logging

logger = logging.getLogger(__name__)


class JitterRetry(Retry):
    """
    Retry with exponential back‑off with jitter.

    The Retry class is being extended as built-in support for jitter was added only in urllib3 2.0.0.
    Jitter logic is following the official implementation with a constant jitter factor: https://github.com/urllib3/urllib3/blob/main/src/urllib3/util/retry.py
    """

    def get_backoff_time(self) -> float:
        backoff_value = super().get_backoff_time()
        jitter = random.random() * 0.25
        backoff_value += jitter
        logger.debug(f"Calculating retry backoff: {backoff_value} (jitter: {jitter})")
        return backoff_value
