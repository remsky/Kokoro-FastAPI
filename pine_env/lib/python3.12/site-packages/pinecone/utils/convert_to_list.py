from ..exceptions.exceptions import ListConversionException


def convert_to_list(obj):
    class_name = obj.__class__.__name__

    if class_name == "list":
        return obj
    elif hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
        return obj.tolist()
    elif obj is None or isinstance(obj, str) or isinstance(obj, dict):
        # The string and dictionary classes in python can be passed to list()
        # but they're not going to yield sensible results for our use case.
        raise ListConversionException(
            f"Expected a list or list-like data structure, but got: {obj}"
        )
    else:
        try:
            return list(obj)
        except Exception as e:
            raise ListConversionException(
                f"Expected a list or list-like data structure, but got: {obj}"
            ) from e
