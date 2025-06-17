import json
from datetime import datetime


def custom_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    try:
        # First try to get a dictionary representation if available
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        # Fall back to string representation
        return str(obj)
    except (TypeError, RecursionError):
        # If we hit any serialization issues, return a safe string representation
        return f"<{obj.__class__.__name__} object>"


def install_json_repr_override(klass):
    klass.__repr__ = lambda self: json.dumps(
        self.to_dict(), indent=4, sort_keys=False, default=custom_serializer
    )
