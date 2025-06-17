from collections.abc import Mapping
from typing import Union, Optional

from ..utils import convert_to_list

from ..db_data import SparseValuesTypeError, SparseValuesMissingKeysError
from ..db_data.types import SparseVectorTypedDict

from pinecone.core.grpc.protos.db_data_2025_01_pb2 import SparseValues as GRPCSparseValues
from pinecone.core.openapi.db_data.models import SparseValues as OpenApiSparseValues
from pinecone import SparseValues


class SparseValuesFactory:
    @staticmethod
    def build(
        input: Optional[
            Union[SparseVectorTypedDict, SparseValues, GRPCSparseValues, OpenApiSparseValues]
        ],
    ) -> Optional[GRPCSparseValues]:
        if input is None:
            return input
        if isinstance(input, GRPCSparseValues):
            return input
        if isinstance(input, SparseValues) or isinstance(input, OpenApiSparseValues):
            return GRPCSparseValues(
                indices=SparseValuesFactory._convert_to_list(input.indices, int),
                values=SparseValuesFactory._convert_to_list(input.values, float),
            )
        if isinstance(input, Mapping):
            if not {"indices", "values"}.issubset(input):
                raise SparseValuesMissingKeysError(input)

            indices = SparseValuesFactory._convert_to_list(input.get("indices"), int)
            values = SparseValuesFactory._convert_to_list(input.get("values"), float)

            if len(indices) != len(values):
                raise ValueError("Sparse values indices and values must have the same length")

            try:
                return GRPCSparseValues(indices=indices, values=values)
            except TypeError as e:
                raise SparseValuesTypeError() from e
        raise ValueError(
            "SparseValuesFactory does not know how to handle input type {}".format(type(input))
        )

    @staticmethod
    def _convert_to_list(input, expected_type):
        try:
            converted = convert_to_list(input)
        except TypeError as e:
            raise SparseValuesTypeError() from e

        SparseValuesFactory._validate_list_items_type(converted, expected_type)
        return converted

    @staticmethod
    def _validate_list_items_type(input, expected_type):
        if len(input) > 0 and not isinstance(input[0], expected_type):
            raise SparseValuesTypeError()
