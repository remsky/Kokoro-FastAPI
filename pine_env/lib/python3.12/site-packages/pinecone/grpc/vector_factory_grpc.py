import numbers

from collections.abc import Iterable, Mapping
from typing import Union

from google.protobuf.struct_pb2 import Struct

from .utils import dict_to_proto_struct
from ..utils import fix_tuple_length, convert_to_list
from ..utils.constants import REQUIRED_VECTOR_FIELDS, OPTIONAL_VECTOR_FIELDS
from ..db_data import (
    VectorDictionaryMissingKeysError,
    VectorDictionaryExcessKeysError,
    VectorTupleLengthError,
    MetadataDictionaryExpectedError,
)
from ..db_data.types import VectorTuple, VectorTypedDict
from .sparse_values_factory import SparseValuesFactory

from pinecone.core.grpc.protos.db_data_2025_01_pb2 import (
    Vector as GRPCVector,
    SparseValues as GRPCSparseValues,
)
from pinecone.core.openapi.db_data.models import (
    SparseValues as OpenApiSparseValues,
    Vector as OpenApiVector,
)
from pinecone import Vector, SparseValues


class VectorFactoryGRPC:
    """This class is responsible for building GRPCVector objects from various input types."""

    @staticmethod
    def build(item: Union[Vector, GRPCVector, Vector, VectorTuple, VectorTypedDict]) -> GRPCVector:
        if isinstance(item, GRPCVector):
            return item
        elif isinstance(item, Vector) or isinstance(item, OpenApiVector):
            if item.sparse_values:
                sv = SparseValuesFactory.build(item.sparse_values)
                return GRPCVector(
                    id=item.id,
                    values=item.values,
                    metadata=dict_to_proto_struct(item.metadata or {}),
                    sparse_values=sv,
                )
            else:
                return GRPCVector(
                    id=item.id,
                    values=item.values,
                    metadata=dict_to_proto_struct(item.metadata or {}),
                )
        elif isinstance(item, tuple):
            return VectorFactoryGRPC._tuple_to_vector(item)
        elif isinstance(item, Mapping):
            return VectorFactoryGRPC._dict_to_vector(item)
        else:
            raise ValueError(f"Invalid vector value passed: cannot interpret type {type(item)}")

    @staticmethod
    def _tuple_to_vector(item) -> GRPCVector:
        if len(item) < 2 or len(item) > 3:
            raise VectorTupleLengthError(item)
        id, values, metadata = fix_tuple_length(item, 3)
        if (
            isinstance(values, GRPCSparseValues)
            or isinstance(values, SparseValues)
            or isinstance(values, OpenApiSparseValues)
        ):
            raise ValueError(
                "Sparse values are not supported in tuples. Please use either dicts or Vector objects as inputs."
            )
        else:
            return GRPCVector(
                id=id, values=convert_to_list(values), metadata=dict_to_proto_struct(metadata or {})
            )

    @staticmethod
    def _dict_to_vector(item) -> GRPCVector:
        item_keys = set(item.keys())
        if not item_keys.issuperset(REQUIRED_VECTOR_FIELDS):
            raise VectorDictionaryMissingKeysError(item)

        excessive_keys = item_keys - (REQUIRED_VECTOR_FIELDS | OPTIONAL_VECTOR_FIELDS)
        if len(excessive_keys) > 0:
            raise VectorDictionaryExcessKeysError(item)

        if "values" not in item and "sparse_values" not in item:
            raise ValueError(
                "At least one of 'values' or 'sparse_values' must be provided in the vector dictionary."
            )

        values = item.get("values")
        if "values" in item:
            try:
                item["values"] = convert_to_list(values)
            except TypeError as e:
                raise TypeError("Column `values` is expected to be a list of floats") from e

        sparse_values = item.get("sparse_values")
        if sparse_values is not None:
            item["sparse_values"] = SparseValuesFactory.build(sparse_values)

        metadata = item.get("metadata")
        if metadata:
            if isinstance(metadata, dict):
                item["metadata"] = dict_to_proto_struct(metadata)
            elif not isinstance(metadata, Struct):
                raise MetadataDictionaryExpectedError(item)
        else:
            item["metadata"] = dict_to_proto_struct({})

        try:
            return GRPCVector(**item)
        except TypeError as e:
            # Where possible raise a more specific error to the user.
            vid = item.get("id")
            if not isinstance(vid, bytes) and not isinstance(vid, str):
                raise TypeError(
                    f"Cannot set Vector.id to {vid}: {vid} has type {type(vid)}, "
                    "but expected one of: (<class 'bytes'>, <class 'str'>) for field Vector.id"
                )
            if not isinstance(item["values"], Iterable) or not isinstance(
                item["values"].__iter__().__next__(), numbers.Real
            ):
                raise TypeError("Column `values` is expected to be a list of floats")
            raise e
