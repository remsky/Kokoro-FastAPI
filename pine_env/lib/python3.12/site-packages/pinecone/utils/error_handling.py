import inspect
from functools import wraps


class ProtocolError(Exception):
    """Raised when there is a protocol error in the connection."""

    pass


def validate_and_convert_errors(func):
    @wraps(func)
    def inner_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Lazy import of urllib3 exceptions
            from urllib3.exceptions import MaxRetryError, ProtocolError as Urllib3ProtocolError

            if isinstance(e, MaxRetryError):
                if isinstance(e.reason, Urllib3ProtocolError):
                    raise ProtocolError(f"Failed to connect to {e.url}") from e
                else:
                    raise e from e
            elif isinstance(e, Urllib3ProtocolError):
                raise ProtocolError(
                    "Connection failed. Please verify that the index host is correct and accessible."
                ) from e
            else:
                raise e from e

    # Override signature
    sig = inspect.signature(func)
    inner_func.__signature__ = sig
    return inner_func
