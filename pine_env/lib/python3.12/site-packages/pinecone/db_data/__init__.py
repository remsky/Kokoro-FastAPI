from .index import (
    Index as _Index,
    FetchResponse,
    QueryResponse,
    DescribeIndexStatsResponse,
    UpsertResponse,
    SparseValues,
    Vector,
)
from .dataclasses import *
from .import_error import (
    Index,
    IndexClientInstantiationError,
    Inference,
    InferenceInstantiationError,
)
from .index_asyncio import *
from .errors import (
    VectorDictionaryMissingKeysError,
    VectorDictionaryExcessKeysError,
    VectorTupleLengthError,
    SparseValuesTypeError,
    SparseValuesMissingKeysError,
    SparseValuesDictionaryExpectedError,
    MetadataDictionaryExpectedError,
)

from .resources.sync.bulk_import import ImportErrorMode

__all__ = [
    "_Index",
    "_IndexAsyncio",
    "DescribeIndexStatsResponse",
    "FetchResponse",
    "ImportErrorMode",
    "Index",
    "IndexClientInstantiationError",
    "Inference",
    "InferenceInstantiationError",
    "MetadataDictionaryExpectedError",
    "QueryResponse",
    "SearchQuery",
    "SearchQueryVector",
    "SearchRerank",
    "SparseValues",
    "SparseValuesDictionaryExpectedError",
    "SparseValuesMissingKeysError",
    "SparseValuesTypeError",
    "UpsertResponse",
    "Vector",
    "VectorDictionaryExcessKeysError",
    "VectorDictionaryMissingKeysError",
    "VectorTupleLengthError",
]

import warnings


def _get_deprecated_import(name, from_module, to_module):
    warnings.warn(
        f"The import of `{name}` from `{from_module}` has moved to `{to_module}`. "
        f"Please update your imports from `from {from_module} import {name}` "
        f"to `from {to_module} import {name}`. "
        "This warning will become an error in a future version of the Pinecone Python SDK.",
        DeprecationWarning,
    )
    # Import from the new location
    from pinecone.inference import (
        Inference as _Inference, # noqa: F401
        AsyncioInference as _AsyncioInference, # noqa: F401
        RerankModel, # noqa: F401
        EmbedModel, # noqa: F401
    )

    return locals()[name]


moved = ["_Inference", "_AsyncioInference", "RerankModel", "EmbedModel"]


def __getattr__(name):
    if name in locals():
        return locals()[name]
    elif name in moved:
        return _get_deprecated_import(name, "pinecone.data", "pinecone.inference")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
