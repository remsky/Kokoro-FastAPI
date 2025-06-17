from ..utils.constants import REQUIRED_VECTOR_FIELDS, OPTIONAL_VECTOR_FIELDS


class VectorDictionaryMissingKeysError(ValueError):
    def __init__(self, item) -> None:
        message = f"Vector dictionary is missing required fields: {list(REQUIRED_VECTOR_FIELDS - set(item.keys()))}"
        super().__init__(message)


class VectorDictionaryExcessKeysError(ValueError):
    def __init__(self, item) -> None:
        invalid_keys = list(set(item.keys()) - (REQUIRED_VECTOR_FIELDS | OPTIONAL_VECTOR_FIELDS))
        message = f"Found excess keys in the vector dictionary: {invalid_keys}. The allowed keys are: {list(REQUIRED_VECTOR_FIELDS | OPTIONAL_VECTOR_FIELDS)}"
        super().__init__(message)


class VectorTupleLengthError(ValueError):
    def __init__(self, item) -> None:
        message = f"Found a tuple of length {len(item)} which is not supported. Vectors can be represented as tuples either the form (id, values, metadata) or (id, values). To pass sparse values please use either dicts or Vector objects as inputs."
        super().__init__(message)


class SparseValuesTypeError(ValueError, TypeError):
    def __init__(self) -> None:
        message = "Found unexpected data in column `sparse_values`. Expected format is `'sparse_values': {'indices': List[int], 'values': List[float]}`."
        super().__init__(message)


class SparseValuesMissingKeysError(ValueError):
    def __init__(self, sparse_values_dict) -> None:
        message = f"Missing required keys in data in column `sparse_values`. Expected format is `'sparse_values': {{'indices': List[int], 'values': List[float]}}`. Found keys {list(sparse_values_dict.keys())}"
        super().__init__(message)


class SparseValuesDictionaryExpectedError(ValueError, TypeError):
    def __init__(self, sparse_values_dict) -> None:
        message = f"Column `sparse_values` is expected to be a dictionary, found {type(sparse_values_dict)}"
        super().__init__(message)


class MetadataDictionaryExpectedError(ValueError, TypeError):
    def __init__(self, item) -> None:
        message = (
            f"Column `metadata` is expected to be a dictionary, found {type(item['metadata'])}"
        )
        super().__init__(message)
