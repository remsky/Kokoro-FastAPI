import numbers

from collections.abc import Iterable, Mapping
from typing import Union, Tuple

from ..utils import fix_tuple_length, convert_to_list, parse_non_empty_args
from ..utils.constants import REQUIRED_VECTOR_FIELDS, OPTIONAL_VECTOR_FIELDS

from .sparse_values_factory import SparseValuesFactory

from pinecone.core.openapi.db_data.models import (
    Vector as OpenApiVector,
    SparseValues as OpenApiSparseValues,
)
from .dataclasses import Vector

from .errors import (
    VectorDictionaryMissingKeysError,
    VectorDictionaryExcessKeysError,
    VectorTupleLengthError,
    MetadataDictionaryExpectedError,
)

from .types import VectorTuple, VectorTypedDict


class VectorFactory:
    """VectorFactory is used to convert various types of input into vector objects used in generated request code."""

    @staticmethod
    def build(
        item: Union[OpenApiVector, VectorTuple, VectorTypedDict], check_type: bool = True
    ) -> OpenApiVector:
        if isinstance(item, OpenApiVector):
            return item
        elif isinstance(item, Vector):
            args = parse_non_empty_args(
                [
                    ("id", item.id),
                    ("values", item.values),
                    ("metadata", item.metadata),
                    ("sparse_values", SparseValuesFactory.build(item.sparse_values)),
                ]
            )

            return OpenApiVector(**args)
        elif isinstance(item, tuple):
            return VectorFactory._tuple_to_vector(item, check_type)
        elif isinstance(item, Mapping):
            return VectorFactory._dict_to_vector(item, check_type)
        else:
            raise ValueError(f"Invalid vector value passed: cannot interpret type {type(item)}")

    @staticmethod
    def _tuple_to_vector(item: Tuple, check_type: bool) -> OpenApiVector:
        if len(item) < 2 or len(item) > 3:
            raise VectorTupleLengthError(item)
        id, values, metadata = fix_tuple_length(item, 3)
        if isinstance(values, OpenApiSparseValues):
            raise ValueError(
                "Sparse values are not supported in tuples. Please use either dicts or OpenApiVector objects as inputs."
            )
        else:
            return OpenApiVector(
                id=id,
                values=convert_to_list(values),
                metadata=metadata or {},
                _check_type=check_type,
            )

    @staticmethod
    def _dict_to_vector(item, check_type: bool) -> OpenApiVector:
        item_keys = set(item.keys())
        if not item_keys.issuperset(REQUIRED_VECTOR_FIELDS):
            raise VectorDictionaryMissingKeysError(item)

        if "sparse_values" not in item_keys and "values" not in item_keys:
            raise ValueError(
                "At least one of 'values' or 'sparse_values' must be provided in the vector dictionary."
            )

        excessive_keys = item_keys - (REQUIRED_VECTOR_FIELDS | OPTIONAL_VECTOR_FIELDS)
        if len(excessive_keys) > 0:
            raise VectorDictionaryExcessKeysError(item)

        values = item.get("values")
        if "values" in item:
            item["values"] = convert_to_list(values)
        else:
            item["values"] = []

        sparse_values = item.get("sparse_values")
        if sparse_values is None:
            item.pop("sparse_values", None)
        else:
            item["sparse_values"] = SparseValuesFactory.build(sparse_values)

        metadata = item.get("metadata")
        if metadata and not isinstance(metadata, Mapping):
            raise MetadataDictionaryExpectedError(item)

        try:
            return OpenApiVector(**item, _check_type=check_type)
        except TypeError as e:
            if not isinstance(item["values"], Iterable) or not isinstance(
                item["values"].__iter__().__next__(), numbers.Real
            ):
                raise TypeError("Column `values` is expected to be a list of floats")
            raise e
