import warnings

__all__ = ["tqdm"]

try:
    # Suppress the specific tqdm warning about IProgress
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="tqdm")
        from tqdm.auto import tqdm
except ImportError:
    # Fallback: define a dummy tqdm that supports the same interface.
    class tqdm:  # type: ignore
        def __init__(self, iterable=None, total=None, desc="", **kwargs):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            # You can store additional kwargs if needed

        def __iter__(self):
            # Just iterate over the underlying iterable
            for item in self.iterable:
                yield item

        def update(self, n=1):
            # No-op: This stub doesn't track progress
            pass

        def __enter__(self):
            # Allow use as a context manager
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            # Nothing to cleanup
            pass
