import functools
import inspect


def require_kwargs(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) > 1:  # First arg is self
            param_names = list(inspect.signature(func).parameters.keys())[1:]  # Skip self
            raise TypeError(
                f"{func.__name__}() requires keyword arguments. "
                f"Please use {func.__name__}({', '.join(f'{name}=value' for name in param_names)})"
            )
        return func(*args, **kwargs)

    return wrapper
