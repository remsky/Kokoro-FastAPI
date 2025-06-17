import logging
from typing import Union, List, Optional, Dict, Any, cast

from pinecone.core.openapi.db_data.models import (
    QueryRequest,
    UpsertRequest,
    DeleteRequest,
    UpdateRequest,
    DescribeIndexStatsRequest,
    SearchRecordsRequest,
    SearchRecordsRequestQuery,
    SearchRecordsRequestRerank,
    VectorValues,
    SearchRecordsVector,
    UpsertRecord,
)
from ..utils import parse_non_empty_args, convert_enum_to_string
from .vector_factory import VectorFactory
from .sparse_values_factory import SparseValuesFactory
from pinecone.openapi_support import OPENAPI_ENDPOINT_PARAMS
from .types import (
    VectorTypedDict,
    SparseVectorTypedDict,
    VectorMetadataTypedDict,
    VectorTuple,
    VectorTupleWithMetadata,
    FilterTypedDict,
    SearchQueryTypedDict,
    SearchRerankTypedDict,
    SearchQueryVectorTypedDict,
)

from .dataclasses import Vector, SparseValues, SearchQuery, SearchRerank, SearchQueryVector

logger = logging.getLogger(__name__)
""" @private """


def non_openapi_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in kwargs.items() if k not in OPENAPI_ENDPOINT_PARAMS}


class IndexRequestFactory:
    @staticmethod
    def query_request(
        top_k: int,
        vector: Optional[List[float]] = None,
        id: Optional[str] = None,
        namespace: Optional[str] = None,
        filter: Optional[FilterTypedDict] = None,
        include_values: Optional[bool] = None,
        include_metadata: Optional[bool] = None,
        sparse_vector: Optional[Union[SparseValues, SparseVectorTypedDict]] = None,
        **kwargs,
    ) -> QueryRequest:
        if vector is not None and id is not None:
            raise ValueError("Cannot specify both `id` and `vector`")

        sparse_vector_normalized = SparseValuesFactory.build(sparse_vector)
        args_dict = parse_non_empty_args(
            [
                ("vector", vector),
                ("id", id),
                ("queries", None),
                ("top_k", top_k),
                ("namespace", namespace),
                ("filter", filter),
                ("include_values", include_values),
                ("include_metadata", include_metadata),
                ("sparse_vector", sparse_vector_normalized),
            ]
        )

        return QueryRequest(
            **args_dict, _check_type=kwargs.pop("_check_type", False), **non_openapi_kwargs(kwargs)
        )

    @staticmethod
    def upsert_request(
        vectors: Union[
            List[Vector], List[VectorTuple], List[VectorTupleWithMetadata], List[VectorTypedDict]
        ],
        namespace: Optional[str],
        _check_type: bool,
        **kwargs,
    ) -> UpsertRequest:
        args_dict = parse_non_empty_args([("namespace", namespace)])

        def vec_builder(v):
            return VectorFactory.build(v, check_type=_check_type)

        return UpsertRequest(
            vectors=list(map(vec_builder, vectors)),
            **args_dict,
            _check_type=_check_type,
            **non_openapi_kwargs(kwargs),
        )

    @staticmethod
    def delete_request(
        ids: Optional[List[str]] = None,
        delete_all: Optional[bool] = None,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = None,
        **kwargs,
    ) -> DeleteRequest:
        _check_type = kwargs.pop("_check_type", False)
        args_dict = parse_non_empty_args(
            [("ids", ids), ("delete_all", delete_all), ("namespace", namespace), ("filter", filter)]
        )
        return DeleteRequest(**args_dict, **non_openapi_kwargs(kwargs), _check_type=_check_type)

    @staticmethod
    def update_request(
        id: str,
        values: Optional[List[float]] = None,
        set_metadata: Optional[VectorMetadataTypedDict] = None,
        namespace: Optional[str] = None,
        sparse_values: Optional[Union[SparseValues, SparseVectorTypedDict]] = None,
        **kwargs,
    ) -> UpdateRequest:
        _check_type = kwargs.pop("_check_type", False)
        sparse_values_normalized = SparseValuesFactory.build(sparse_values)
        args_dict = parse_non_empty_args(
            [
                ("values", values),
                ("set_metadata", set_metadata),
                ("namespace", namespace),
                ("sparse_values", sparse_values_normalized),
            ]
        )

        return UpdateRequest(
            id=id, **args_dict, _check_type=_check_type, **non_openapi_kwargs(kwargs)
        )

    @staticmethod
    def describe_index_stats_request(
        filter: Optional[FilterTypedDict] = None, **kwargs
    ) -> DescribeIndexStatsRequest:
        _check_type = kwargs.pop("_check_type", False)
        args_dict = parse_non_empty_args([("filter", filter)])

        return DescribeIndexStatsRequest(
            **args_dict, **non_openapi_kwargs(kwargs), _check_type=_check_type
        )

    @staticmethod
    def list_paginated_args(
        prefix: Optional[str] = None,
        limit: Optional[int] = None,
        pagination_token: Optional[str] = None,
        namespace: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        return parse_non_empty_args(
            [
                ("prefix", prefix),
                ("limit", limit),
                ("namespace", namespace),
                ("pagination_token", pagination_token),
            ]
        )

    @staticmethod
    def search_request(
        query: Union[SearchQueryTypedDict, SearchQuery],
        rerank: Optional[Union[SearchRerankTypedDict, SearchRerank]] = None,
        fields: Optional[List[str]] = ["*"],  # Default to returning all fields
    ) -> SearchRecordsRequest:
        request_args = parse_non_empty_args(
            [
                ("query", IndexRequestFactory._parse_search_query(query)),
                ("fields", fields),
                ("rerank", IndexRequestFactory._parse_search_rerank(rerank)),
            ]
        )

        return SearchRecordsRequest(**request_args)

    @staticmethod
    def _parse_search_query(
        query: Union[SearchQueryTypedDict, SearchQuery],
    ) -> SearchRecordsRequestQuery:
        if isinstance(query, SearchQuery):
            query_dict = query.as_dict()
        else:
            query_dict = cast(dict[str, Any], query)

        required_fields = {"top_k"}
        for key in required_fields:
            if query_dict.get(key, None) is None:
                raise ValueError(f"Missing required field '{key}' in search query.")

        # User-provided dict could contain object that need conversion
        if isinstance(query_dict.get("vector", None), SearchQueryVector):
            query_dict["vector"] = query_dict["vector"].as_dict()

        srrq = SearchRecordsRequestQuery(
            **{k: v for k, v in query_dict.items() if k not in {"vector"}}
        )
        if query_dict.get("vector", None) is not None:
            srrq.vector = IndexRequestFactory._parse_search_vector(query_dict["vector"])
        return srrq

    @staticmethod
    def _parse_search_vector(
        vector: Optional[Union[SearchQueryVectorTypedDict, SearchQueryVector]],
    ):
        if vector is None:
            return None

        if isinstance(vector, SearchQueryVector):
            if vector.values is None and vector.sparse_values is None:
                return None
            vector_dict = vector.as_dict()
        else:
            vector_dict = cast(dict[str, Any], vector)
            if (
                vector_dict.get("values", None) is None
                and vector_dict.get("sparse_values", None) is None
            ):
                return None

        srv = SearchRecordsVector(**{k: v for k, v in vector_dict.items() if k not in {"values"}})

        values = vector_dict.get("values", None)
        if values is not None:
            srv.values = VectorValues(value=values)

        return srv

    @staticmethod
    def _parse_search_rerank(rerank: Optional[Union[SearchRerankTypedDict, SearchRerank]] = None):
        if rerank is None:
            return None

        if isinstance(rerank, SearchRerank):
            rerank_dict = rerank.as_dict()
        else:
            rerank_dict = cast(dict[str, Any], rerank)

        required_fields = {"model", "rank_fields"}
        for key in required_fields:
            if rerank_dict.get(key, None) is None:
                raise ValueError(f"Missing required field '{key}' in rerank.")

        rerank_dict["model"] = convert_enum_to_string(rerank_dict["model"])

        return SearchRecordsRequestRerank(**rerank_dict)

    @staticmethod
    def upsert_records_args(namespace: str, records: List[Dict]):
        if namespace is None:
            raise ValueError("namespace is required when upserting records")
        if not records or len(records) == 0:
            raise ValueError("No records provided")

        records_to_upsert = []
        for record in records:
            if not record.get("_id") and not record.get("id"):
                raise ValueError("Each record must have an '_id' or 'id' value")

            records_to_upsert.append(
                UpsertRecord(
                    record.get("_id", record.get("id")),
                    **{k: v for k, v in record.items() if k not in {"_id", "id"}},
                )
            )

        return {"namespace": namespace, "upsert_record": records_to_upsert}
