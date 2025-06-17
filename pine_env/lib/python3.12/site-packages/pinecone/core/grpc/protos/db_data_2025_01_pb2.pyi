from google.protobuf import struct_pb2 as _struct_pb2
from google.api import annotations_pb2 as _annotations_pb2
from google.api import field_behavior_pb2 as _field_behavior_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SparseValues(_message.Message):
    __slots__ = ("indices", "values")
    INDICES_FIELD_NUMBER: _ClassVar[int]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    indices: _containers.RepeatedScalarFieldContainer[int]
    values: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, indices: _Optional[_Iterable[int]] = ..., values: _Optional[_Iterable[float]] = ...) -> None: ...

class Vector(_message.Message):
    __slots__ = ("id", "values", "sparse_values", "metadata")
    ID_FIELD_NUMBER: _ClassVar[int]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    SPARSE_VALUES_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    id: str
    values: _containers.RepeatedScalarFieldContainer[float]
    sparse_values: SparseValues
    metadata: _struct_pb2.Struct
    def __init__(self, id: _Optional[str] = ..., values: _Optional[_Iterable[float]] = ..., sparse_values: _Optional[_Union[SparseValues, _Mapping]] = ..., metadata: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class ScoredVector(_message.Message):
    __slots__ = ("id", "score", "values", "sparse_values", "metadata")
    ID_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    SPARSE_VALUES_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    id: str
    score: float
    values: _containers.RepeatedScalarFieldContainer[float]
    sparse_values: SparseValues
    metadata: _struct_pb2.Struct
    def __init__(self, id: _Optional[str] = ..., score: _Optional[float] = ..., values: _Optional[_Iterable[float]] = ..., sparse_values: _Optional[_Union[SparseValues, _Mapping]] = ..., metadata: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class RequestUnion(_message.Message):
    __slots__ = ("upsert", "delete", "update")
    UPSERT_FIELD_NUMBER: _ClassVar[int]
    DELETE_FIELD_NUMBER: _ClassVar[int]
    UPDATE_FIELD_NUMBER: _ClassVar[int]
    upsert: UpsertRequest
    delete: DeleteRequest
    update: UpdateRequest
    def __init__(self, upsert: _Optional[_Union[UpsertRequest, _Mapping]] = ..., delete: _Optional[_Union[DeleteRequest, _Mapping]] = ..., update: _Optional[_Union[UpdateRequest, _Mapping]] = ...) -> None: ...

class UpsertRequest(_message.Message):
    __slots__ = ("vectors", "namespace")
    VECTORS_FIELD_NUMBER: _ClassVar[int]
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    vectors: _containers.RepeatedCompositeFieldContainer[Vector]
    namespace: str
    def __init__(self, vectors: _Optional[_Iterable[_Union[Vector, _Mapping]]] = ..., namespace: _Optional[str] = ...) -> None: ...

class UpsertResponse(_message.Message):
    __slots__ = ("upserted_count",)
    UPSERTED_COUNT_FIELD_NUMBER: _ClassVar[int]
    upserted_count: int
    def __init__(self, upserted_count: _Optional[int] = ...) -> None: ...

class DeleteRequest(_message.Message):
    __slots__ = ("ids", "delete_all", "namespace", "filter")
    IDS_FIELD_NUMBER: _ClassVar[int]
    DELETE_ALL_FIELD_NUMBER: _ClassVar[int]
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    ids: _containers.RepeatedScalarFieldContainer[str]
    delete_all: bool
    namespace: str
    filter: _struct_pb2.Struct
    def __init__(self, ids: _Optional[_Iterable[str]] = ..., delete_all: bool = ..., namespace: _Optional[str] = ..., filter: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class DeleteResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class FetchRequest(_message.Message):
    __slots__ = ("ids", "namespace")
    IDS_FIELD_NUMBER: _ClassVar[int]
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    ids: _containers.RepeatedScalarFieldContainer[str]
    namespace: str
    def __init__(self, ids: _Optional[_Iterable[str]] = ..., namespace: _Optional[str] = ...) -> None: ...

class FetchResponse(_message.Message):
    __slots__ = ("vectors", "namespace", "usage")
    class VectorsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Vector
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Vector, _Mapping]] = ...) -> None: ...
    VECTORS_FIELD_NUMBER: _ClassVar[int]
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    USAGE_FIELD_NUMBER: _ClassVar[int]
    vectors: _containers.MessageMap[str, Vector]
    namespace: str
    usage: Usage
    def __init__(self, vectors: _Optional[_Mapping[str, Vector]] = ..., namespace: _Optional[str] = ..., usage: _Optional[_Union[Usage, _Mapping]] = ...) -> None: ...

class ListRequest(_message.Message):
    __slots__ = ("prefix", "limit", "pagination_token", "namespace")
    PREFIX_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    PAGINATION_TOKEN_FIELD_NUMBER: _ClassVar[int]
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    prefix: str
    limit: int
    pagination_token: str
    namespace: str
    def __init__(self, prefix: _Optional[str] = ..., limit: _Optional[int] = ..., pagination_token: _Optional[str] = ..., namespace: _Optional[str] = ...) -> None: ...

class Pagination(_message.Message):
    __slots__ = ("next",)
    NEXT_FIELD_NUMBER: _ClassVar[int]
    next: str
    def __init__(self, next: _Optional[str] = ...) -> None: ...

class ListItem(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    def __init__(self, id: _Optional[str] = ...) -> None: ...

class ListResponse(_message.Message):
    __slots__ = ("vectors", "pagination", "namespace", "usage")
    VECTORS_FIELD_NUMBER: _ClassVar[int]
    PAGINATION_FIELD_NUMBER: _ClassVar[int]
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    USAGE_FIELD_NUMBER: _ClassVar[int]
    vectors: _containers.RepeatedCompositeFieldContainer[ListItem]
    pagination: Pagination
    namespace: str
    usage: Usage
    def __init__(self, vectors: _Optional[_Iterable[_Union[ListItem, _Mapping]]] = ..., pagination: _Optional[_Union[Pagination, _Mapping]] = ..., namespace: _Optional[str] = ..., usage: _Optional[_Union[Usage, _Mapping]] = ...) -> None: ...

class QueryVector(_message.Message):
    __slots__ = ("values", "sparse_values", "top_k", "namespace", "filter")
    VALUES_FIELD_NUMBER: _ClassVar[int]
    SPARSE_VALUES_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[float]
    sparse_values: SparseValues
    top_k: int
    namespace: str
    filter: _struct_pb2.Struct
    def __init__(self, values: _Optional[_Iterable[float]] = ..., sparse_values: _Optional[_Union[SparseValues, _Mapping]] = ..., top_k: _Optional[int] = ..., namespace: _Optional[str] = ..., filter: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class QueryRequest(_message.Message):
    __slots__ = ("namespace", "top_k", "filter", "include_values", "include_metadata", "queries", "vector", "sparse_vector", "id")
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_VALUES_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_METADATA_FIELD_NUMBER: _ClassVar[int]
    QUERIES_FIELD_NUMBER: _ClassVar[int]
    VECTOR_FIELD_NUMBER: _ClassVar[int]
    SPARSE_VECTOR_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    namespace: str
    top_k: int
    filter: _struct_pb2.Struct
    include_values: bool
    include_metadata: bool
    queries: _containers.RepeatedCompositeFieldContainer[QueryVector]
    vector: _containers.RepeatedScalarFieldContainer[float]
    sparse_vector: SparseValues
    id: str
    def __init__(self, namespace: _Optional[str] = ..., top_k: _Optional[int] = ..., filter: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., include_values: bool = ..., include_metadata: bool = ..., queries: _Optional[_Iterable[_Union[QueryVector, _Mapping]]] = ..., vector: _Optional[_Iterable[float]] = ..., sparse_vector: _Optional[_Union[SparseValues, _Mapping]] = ..., id: _Optional[str] = ...) -> None: ...

class SingleQueryResults(_message.Message):
    __slots__ = ("matches", "namespace")
    MATCHES_FIELD_NUMBER: _ClassVar[int]
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    matches: _containers.RepeatedCompositeFieldContainer[ScoredVector]
    namespace: str
    def __init__(self, matches: _Optional[_Iterable[_Union[ScoredVector, _Mapping]]] = ..., namespace: _Optional[str] = ...) -> None: ...

class QueryResponse(_message.Message):
    __slots__ = ("results", "matches", "namespace", "usage")
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    MATCHES_FIELD_NUMBER: _ClassVar[int]
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    USAGE_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[SingleQueryResults]
    matches: _containers.RepeatedCompositeFieldContainer[ScoredVector]
    namespace: str
    usage: Usage
    def __init__(self, results: _Optional[_Iterable[_Union[SingleQueryResults, _Mapping]]] = ..., matches: _Optional[_Iterable[_Union[ScoredVector, _Mapping]]] = ..., namespace: _Optional[str] = ..., usage: _Optional[_Union[Usage, _Mapping]] = ...) -> None: ...

class Usage(_message.Message):
    __slots__ = ("read_units",)
    READ_UNITS_FIELD_NUMBER: _ClassVar[int]
    read_units: int
    def __init__(self, read_units: _Optional[int] = ...) -> None: ...

class UpdateRequest(_message.Message):
    __slots__ = ("id", "values", "sparse_values", "set_metadata", "namespace")
    ID_FIELD_NUMBER: _ClassVar[int]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    SPARSE_VALUES_FIELD_NUMBER: _ClassVar[int]
    SET_METADATA_FIELD_NUMBER: _ClassVar[int]
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    id: str
    values: _containers.RepeatedScalarFieldContainer[float]
    sparse_values: SparseValues
    set_metadata: _struct_pb2.Struct
    namespace: str
    def __init__(self, id: _Optional[str] = ..., values: _Optional[_Iterable[float]] = ..., sparse_values: _Optional[_Union[SparseValues, _Mapping]] = ..., set_metadata: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., namespace: _Optional[str] = ...) -> None: ...

class UpdateResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class DescribeIndexStatsRequest(_message.Message):
    __slots__ = ("filter",)
    FILTER_FIELD_NUMBER: _ClassVar[int]
    filter: _struct_pb2.Struct
    def __init__(self, filter: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class NamespaceSummary(_message.Message):
    __slots__ = ("vector_count",)
    VECTOR_COUNT_FIELD_NUMBER: _ClassVar[int]
    vector_count: int
    def __init__(self, vector_count: _Optional[int] = ...) -> None: ...

class DescribeIndexStatsResponse(_message.Message):
    __slots__ = ("namespaces", "dimension", "index_fullness", "total_vector_count", "metric", "vector_type")
    class NamespacesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: NamespaceSummary
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[NamespaceSummary, _Mapping]] = ...) -> None: ...
    NAMESPACES_FIELD_NUMBER: _ClassVar[int]
    DIMENSION_FIELD_NUMBER: _ClassVar[int]
    INDEX_FULLNESS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_VECTOR_COUNT_FIELD_NUMBER: _ClassVar[int]
    METRIC_FIELD_NUMBER: _ClassVar[int]
    VECTOR_TYPE_FIELD_NUMBER: _ClassVar[int]
    namespaces: _containers.MessageMap[str, NamespaceSummary]
    dimension: int
    index_fullness: float
    total_vector_count: int
    metric: str
    vector_type: str
    def __init__(self, namespaces: _Optional[_Mapping[str, NamespaceSummary]] = ..., dimension: _Optional[int] = ..., index_fullness: _Optional[float] = ..., total_vector_count: _Optional[int] = ..., metric: _Optional[str] = ..., vector_type: _Optional[str] = ...) -> None: ...
