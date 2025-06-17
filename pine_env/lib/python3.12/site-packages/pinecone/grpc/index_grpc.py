import logging
from typing import Optional, Dict, Union, List, Tuple, Any, Iterable, cast, Literal

from google.protobuf import json_format

from pinecone.utils.tqdm import tqdm
from concurrent.futures import as_completed, Future


from .utils import (
    dict_to_proto_struct,
    parse_fetch_response,
    parse_query_response,
    parse_stats_response,
)
from .vector_factory_grpc import VectorFactoryGRPC
from .sparse_values_factory import SparseValuesFactory

from pinecone.core.openapi.db_data.models import (
    FetchResponse,
    QueryResponse,
    IndexDescription as DescribeIndexStatsResponse,
)
from pinecone.db_control.models.list_response import ListResponse as SimpleListResponse, Pagination
from pinecone.core.grpc.protos.db_data_2025_01_pb2 import (
    Vector as GRPCVector,
    QueryVector as GRPCQueryVector,
    UpsertRequest,
    UpsertResponse,
    DeleteRequest,
    QueryRequest,
    FetchRequest,
    UpdateRequest,
    ListRequest,
    DescribeIndexStatsRequest,
    DeleteResponse,
    UpdateResponse,
    SparseValues as GRPCSparseValues,
)
from pinecone import Vector, SparseValues
from pinecone.db_data.query_results_aggregator import QueryNamespacesResults, QueryResultsAggregator
from pinecone.core.grpc.protos.db_data_2025_01_pb2_grpc import VectorServiceStub
from .base import GRPCIndexBase
from .future import PineconeGrpcFuture
from ..db_data.types import (
    SparseVectorTypedDict,
    VectorTypedDict,
    VectorTuple,
    FilterTypedDict,
    VectorMetadataTypedDict,
)


__all__ = ["GRPCIndex", "GRPCVector", "GRPCQueryVector", "GRPCSparseValues"]

_logger = logging.getLogger(__name__)
""" @private """


class GRPCIndex(GRPCIndexBase):
    """A client for interacting with a Pinecone index via GRPC API."""

    @property
    def stub_class(self):
        """@private"""
        return VectorServiceStub

    def upsert(
        self,
        vectors: Union[List[Vector], List[GRPCVector], List[VectorTuple], List[VectorTypedDict]],
        async_req: bool = False,
        namespace: Optional[str] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> Union[UpsertResponse, PineconeGrpcFuture]:
        """
        The upsert operation writes vectors into a namespace.
        If a new value is upserted for an existing vector id, it will overwrite the previous value.

        Examples:
            >>> index.upsert([('id1', [1.0, 2.0, 3.0], {'key': 'value'}),
                              ('id2', [1.0, 2.0, 3.0])
                              ],
                              namespace='ns1', async_req=True)
            >>> index.upsert([{'id': 'id1', 'values': [1.0, 2.0, 3.0], 'metadata': {'key': 'value'}},
                              {'id': 'id2',
                                        'values': [1.0, 2.0, 3.0],
                                        'sparse_values': {'indices': [1, 8], 'values': [0.2, 0.4]},
                              ])
            >>> index.upsert([GRPCVector(id='id1', values=[1.0, 2.0, 3.0], metadata={'key': 'value'}),
                              GRPCVector(id='id2', values=[1.0, 2.0, 3.0]),
                              GRPCVector(id='id3',
                                         values=[1.0, 2.0, 3.0],
                                         sparse_values=GRPCSparseValues(indices=[1, 2], values=[0.2, 0.4]))])

        Args:
            vectors (Union[List[Vector], List[Tuple]]): A list of vectors to upsert.

                     A vector can be represented by a 1) GRPCVector object, a 2) tuple or 3) a dictionary
                     1) if a tuple is used, it must be of the form (id, values, metadata) or (id, values).
                        where id is a string, vector is a list of floats, and metadata is a dict.
                        Examples: ('id1', [1.0, 2.0, 3.0], {'key': 'value'}), ('id2', [1.0, 2.0, 3.0])

                    2) if a GRPCVector object is used, a GRPCVector object must be of the form
                        GRPCVector(id, values, metadata), where metadata is an optional argument of type
                        Dict[str, Union[str, float, int, bool, List[int], List[float], List[str]]]
                       Examples: GRPCVector(id='id1', values=[1.0, 2.0, 3.0], metadata={'key': 'value'}),
                                 GRPCVector(id='id2', values=[1.0, 2.0, 3.0]),
                                 GRPCVector(id='id3',
                                            values=[1.0, 2.0, 3.0],
                                            sparse_values=GRPCSparseValues(indices=[1, 2], values=[0.2, 0.4]))

                    3) if a dictionary is used, it must be in the form
                       {'id': str, 'values': List[float], 'sparse_values': {'indices': List[int], 'values': List[float]},
                        'metadata': dict}

                    Note: the dimension of each vector must match the dimension of the index.
            async_req (bool): If True, the upsert operation will be performed asynchronously.
                              Cannot be used with batch_size.
                              Defaults to False. See: https://docs.pinecone.io/docs/performance-tuning [optional]
            namespace (str): The namespace to write to. If not specified, the default namespace is used. [optional]
            batch_size (int): The number of vectors to upsert in each batch.
                                Cannot be used with async_req=True.
                               If not specified, all vectors will be upserted in a single batch. [optional]
            show_progress (bool): Whether to show a progress bar using tqdm.
                                  Applied only if batch_size is provided. Default is True.

        Returns: UpsertResponse, contains the number of vectors upserted
        """
        if async_req and batch_size is not None:
            raise ValueError(
                "async_req is not supported when batch_size is provided."
                "To upsert in parallel, please follow: "
                "https://docs.pinecone.io/docs/performance-tuning"
            )

        timeout = kwargs.pop("timeout", None)

        vectors = list(map(VectorFactoryGRPC.build, vectors))
        if async_req:
            args_dict = self._parse_non_empty_args([("namespace", namespace)])
            request = UpsertRequest(vectors=vectors, **args_dict, **kwargs)
            future = self.runner.run(self.stub.Upsert.future, request, timeout=timeout)
            return PineconeGrpcFuture(future)

        if batch_size is None:
            return self._upsert_batch(vectors, namespace, timeout=timeout, **kwargs)

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        pbar = tqdm(total=len(vectors), disable=not show_progress, desc="Upserted vectors")
        total_upserted = 0
        for i in range(0, len(vectors), batch_size):
            batch_result = self._upsert_batch(
                vectors[i : i + batch_size], namespace, timeout=timeout, **kwargs
            )
            pbar.update(batch_result.upserted_count)
            # we can't use here pbar.n for the case show_progress=False
            total_upserted += batch_result.upserted_count

        return UpsertResponse(upserted_count=total_upserted)

    def _upsert_batch(
        self, vectors: List[GRPCVector], namespace: Optional[str], timeout: Optional[int], **kwargs
    ) -> UpsertResponse:
        args_dict = self._parse_non_empty_args([("namespace", namespace)])
        request = UpsertRequest(vectors=vectors, **args_dict)
        return self.runner.run(self.stub.Upsert, request, timeout=timeout, **kwargs)

    def upsert_from_dataframe(
        self,
        df,
        namespace: str = "",
        batch_size: int = 500,
        use_async_requests: bool = True,
        show_progress: bool = True,
    ) -> UpsertResponse:
        """Upserts a dataframe into the index.

        Args:
            df: A pandas dataframe with the following columns: id, values, sparse_values, and metadata.
            namespace: The namespace to upsert into.
            batch_size: The number of rows to upsert in a single batch.
            use_async_requests: Whether to upsert multiple requests at the same time using asynchronous request mechanism.
                                Set to `False`
            show_progress: Whether to show a progress bar.
        """
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError(
                "The `pandas` package is not installed. Please install pandas to use `upsert_from_dataframe()`"
            )

        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Only pandas dataframes are supported. Found: {type(df)}")

        pbar = tqdm(total=len(df), disable=not show_progress, desc="sending upsert requests")
        results = []
        for chunk in self._iter_dataframe(df, batch_size=batch_size):
            res = self.upsert(vectors=chunk, namespace=namespace, async_req=use_async_requests)
            pbar.update(len(chunk))
            results.append(res)

        if use_async_requests:
            cast_results = cast(List[PineconeGrpcFuture], results)
            results = [
                async_result.result()
                for async_result in tqdm(
                    iterable=cast_results,
                    disable=not show_progress,
                    desc="collecting async responses",
                )
            ]

        upserted_count = 0
        for res in results:
            if hasattr(res, "upserted_count") and isinstance(res.upserted_count, int):
                upserted_count += res.upserted_count

        return UpsertResponse(upserted_count=upserted_count)

    @staticmethod
    def _iter_dataframe(df, batch_size):
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i : i + batch_size].to_dict(orient="records")
            yield batch

    def delete(
        self,
        ids: Optional[List[str]] = None,
        delete_all: Optional[bool] = None,
        namespace: Optional[str] = None,
        filter: Optional[FilterTypedDict] = None,
        async_req: bool = False,
        **kwargs,
    ) -> Union[DeleteResponse, PineconeGrpcFuture]:
        """
        The Delete operation deletes vectors from the index, from a single namespace.
        No error raised if the vector id does not exist.
        Note: for any delete call, if namespace is not specified, the default namespace is used.

        Delete can occur in the following mutual exclusive ways:
        1. Delete by ids from a single namespace
        2. Delete all vectors from a single namespace by setting delete_all to True
        3. Delete all vectors from a single namespace by specifying a metadata filter
           (note that for this option delete all must be set to False)

        Examples:
            >>> index.delete(ids=['id1', 'id2'], namespace='my_namespace')
            >>> index.delete(delete_all=True, namespace='my_namespace')
            >>> index.delete(filter={'key': 'value'}, namespace='my_namespace', async_req=True)

        Args:
            ids (List[str]): Vector ids to delete [optional]
            delete_all (bool): This indicates that all vectors in the index namespace should be deleted.. [optional]
                               Default is False.
            namespace (str): The namespace to delete vectors from [optional]
                             If not specified, the default namespace is used.
            filter (FilterTypedDict):
                    If specified, the metadata filter here will be used to select the vectors to delete.
                    This is mutually exclusive with specifying ids to delete in the ids param or using delete_all=True.
                     See https://www.pinecone.io/docs/metadata-filtering/.. [optional]
            async_req (bool): If True, the delete operation will be performed asynchronously.
                              Defaults to False. [optional]

        Returns: DeleteResponse (contains no data) or a PineconeGrpcFuture object if async_req is True.
        """

        if filter is not None:
            filter_struct = dict_to_proto_struct(filter)
        else:
            filter_struct = None

        args_dict = self._parse_non_empty_args(
            [
                ("ids", ids),
                ("delete_all", delete_all),
                ("namespace", namespace),
                ("filter", filter_struct),
            ]
        )
        timeout = kwargs.pop("timeout", None)

        request = DeleteRequest(**args_dict, **kwargs)
        if async_req:
            future = self.runner.run(self.stub.Delete.future, request, timeout=timeout)
            return PineconeGrpcFuture(future)
        else:
            return self.runner.run(self.stub.Delete, request, timeout=timeout)

    def fetch(
        self,
        ids: Optional[List[str]],
        namespace: Optional[str] = None,
        async_req: Optional[bool] = False,
        **kwargs,
    ) -> Union[FetchResponse, PineconeGrpcFuture]:
        """
        The fetch operation looks up and returns vectors, by ID, from a single namespace.
        The returned vectors include the vector data and/or metadata.

        Examples:
            >>> index.fetch(ids=['id1', 'id2'], namespace='my_namespace')
            >>> index.fetch(ids=['id1', 'id2'])

        Args:
            ids (List[str]): The vector IDs to fetch.
            namespace (str): The namespace to fetch vectors from.
                             If not specified, the default namespace is used. [optional]

        Returns: FetchResponse object which contains the list of Vector objects, and namespace name.
        """
        timeout = kwargs.pop("timeout", None)

        args_dict = self._parse_non_empty_args([("namespace", namespace)])

        request = FetchRequest(ids=ids, **args_dict, **kwargs)

        if async_req:
            future = self.runner.run(self.stub.Fetch.future, request, timeout=timeout)
            return PineconeGrpcFuture(future, result_transformer=parse_fetch_response)
        else:
            response = self.runner.run(self.stub.Fetch, request, timeout=timeout)
            return parse_fetch_response(response)

    def query(
        self,
        vector: Optional[List[float]] = None,
        id: Optional[str] = None,
        namespace: Optional[str] = None,
        top_k: Optional[int] = None,
        filter: Optional[FilterTypedDict] = None,
        include_values: Optional[bool] = None,
        include_metadata: Optional[bool] = None,
        sparse_vector: Optional[
            Union[SparseValues, GRPCSparseValues, SparseVectorTypedDict]
        ] = None,
        async_req: Optional[bool] = False,
        **kwargs,
    ) -> Union[QueryResponse, PineconeGrpcFuture]:
        """
        The Query operation searches a namespace, using a query vector.
        It retrieves the ids of the most similar items in a namespace, along with their similarity scores.

        Examples:
            >>> index.query(vector=[1, 2, 3], top_k=10, namespace='my_namespace')
            >>> index.query(id='id1', top_k=10, namespace='my_namespace')
            >>> index.query(vector=[1, 2, 3], top_k=10, namespace='my_namespace', filter={'key': 'value'})
            >>> index.query(id='id1', top_k=10, namespace='my_namespace', include_metadata=True, include_values=True)
            >>> index.query(vector=[1, 2, 3], sparse_vector={'indices': [1, 2], 'values': [0.2, 0.4]},
            >>>             top_k=10, namespace='my_namespace')
            >>> index.query(vector=[1, 2, 3], sparse_vector=GRPCSparseValues([1, 2], [0.2, 0.4]),
            >>>             top_k=10, namespace='my_namespace')

        Args:
            vector (List[float]): The query vector. This should be the same length as the dimension of the index
                                  being queried. Each `query()` request can contain only one of the parameters
                                  `id` or `vector`.. [optional]
            id (str): The unique ID of the vector to be used as a query vector.
                      Each `query()` request can contain only one of the parameters
                      `vector` or  `id`.. [optional]
            top_k (int): The number of results to return for each query. Must be an integer greater than 1.
            namespace (str): The namespace to fetch vectors from.
                             If not specified, the default namespace is used. [optional]
            filter (Dict[str, Union[str, float, int, bool, List, dict]]):
                    The filter to apply. You can use vector metadata to limit your search.
                    See https://www.pinecone.io/docs/metadata-filtering/.. [optional]
            include_values (bool): Indicates whether vector values are included in the response.
                                   If omitted the server will use the default value of False [optional]
            include_metadata (bool): Indicates whether metadata is included in the response as well as the ids.
                                     If omitted the server will use the default value of False  [optional]
            sparse_vector: (Union[SparseValues, Dict[str, Union[List[float], List[int]]]]): sparse values of the query vector.
                            Expected to be either a SparseValues object or a dict of the form:
                             {'indices': List[int], 'values': List[float]}, where the lists each have the same length.

        Returns: QueryResponse object which contains the list of the closest vectors as ScoredVector objects,
                 and namespace name.
        """

        if vector is not None and id is not None:
            raise ValueError("Cannot specify both `id` and `vector`")

        if filter is not None:
            filter_struct = dict_to_proto_struct(filter)
        else:
            filter_struct = None

        sparse_vector = SparseValuesFactory.build(sparse_vector)
        args_dict = self._parse_non_empty_args(
            [
                ("vector", vector),
                ("id", id),
                ("namespace", namespace),
                ("top_k", top_k),
                ("filter", filter_struct),
                ("include_values", include_values),
                ("include_metadata", include_metadata),
                ("sparse_vector", sparse_vector),
            ]
        )

        request = QueryRequest(**args_dict)

        timeout = kwargs.pop("timeout", None)

        if async_req:
            future = self.runner.run(self.stub.Query.future, request, timeout=timeout)
            return PineconeGrpcFuture(future)
        else:
            response = self.runner.run(self.stub.Query, request, timeout=timeout)
            json_response = json_format.MessageToDict(response)
            return parse_query_response(json_response, _check_type=False)

    def query_namespaces(
        self,
        vector: List[float],
        namespaces: List[str],
        metric: Literal["cosine", "euclidean", "dotproduct"],
        top_k: Optional[int] = None,
        filter: Optional[FilterTypedDict] = None,
        include_values: Optional[bool] = None,
        include_metadata: Optional[bool] = None,
        sparse_vector: Optional[Union[GRPCSparseValues, SparseVectorTypedDict]] = None,
        **kwargs,
    ) -> QueryNamespacesResults:
        if namespaces is None or len(namespaces) == 0:
            raise ValueError("At least one namespace must be specified")
        if len(vector) == 0:
            raise ValueError("Query vector must not be empty")

        overall_topk = top_k if top_k is not None else 10
        aggregator = QueryResultsAggregator(top_k=overall_topk, metric=metric)

        target_namespaces = set(namespaces)  # dedup namespaces
        futures = [
            self.threadpool_executor.submit(
                self.query,
                vector=vector,
                namespace=ns,
                top_k=overall_topk,
                filter=filter,
                include_values=include_values,
                include_metadata=include_metadata,
                sparse_vector=sparse_vector,
                async_req=False,
                **kwargs,
            )
            for ns in target_namespaces
        ]

        only_futures = cast(Iterable[Future], futures)
        for response in as_completed(only_futures):
            aggregator.add_results(response.result())

        final_results = aggregator.get_results()
        return final_results

    def update(
        self,
        id: str,
        async_req: bool = False,
        values: Optional[List[float]] = None,
        set_metadata: Optional[VectorMetadataTypedDict] = None,
        namespace: Optional[str] = None,
        sparse_values: Optional[Union[GRPCSparseValues, SparseVectorTypedDict]] = None,
        **kwargs,
    ) -> Union[UpdateResponse, PineconeGrpcFuture]:
        """
        The Update operation updates vector in a namespace.
        If a value is included, it will overwrite the previous value.
        If a set_metadata is included,
        the values of the fields specified in it will be added or overwrite the previous value.

        Examples:
            >>> index.update(id='id1', values=[1, 2, 3], namespace='my_namespace')
            >>> index.update(id='id1', set_metadata={'key': 'value'}, namespace='my_namespace', async_req=True)
            >>> index.update(id='id1', values=[1, 2, 3], sparse_values={'indices': [1, 2], 'values': [0.2, 0.4]},
            >>>              namespace='my_namespace')
            >>> index.update(id='id1', values=[1, 2, 3], sparse_values=GRPCSparseValues(indices=[1, 2], values=[0.2, 0.4]),
            >>>              namespace='my_namespace')

        Args:
            id (str): Vector's unique id.
            async_req (bool): If True, the update operation will be performed asynchronously.
                              Defaults to False. [optional]
            values (List[float]): vector values to set. [optional]
            set_metadata (Dict[str, Union[str, float, int, bool, List[int], List[float], List[str]]]]):
                metadata to set for vector. [optional]
            namespace (str): Namespace name where to update the vector.. [optional]
            sparse_values: (Dict[str, Union[List[float], List[int]]]): sparse values to update for the vector.
                           Expected to be either a GRPCSparseValues object or a dict of the form:
                           {'indices': List[int], 'values': List[float]} where the lists each have the same length.


        Returns: UpdateResponse (contains no data) or a PineconeGrpcFuture object if async_req is True.
        """
        if set_metadata is not None:
            set_metadata_struct = dict_to_proto_struct(set_metadata)
        else:
            set_metadata_struct = None

        timeout = kwargs.pop("timeout", None)
        sparse_values = SparseValuesFactory.build(sparse_values)
        args_dict = self._parse_non_empty_args(
            [
                ("values", values),
                ("set_metadata", set_metadata_struct),
                ("namespace", namespace),
                ("sparse_values", sparse_values),
            ]
        )

        request = UpdateRequest(id=id, **args_dict)
        if async_req:
            future = self.runner.run(self.stub.Update.future, request, timeout=timeout)
            return PineconeGrpcFuture(future)
        else:
            return self.runner.run(self.stub.Update, request, timeout=timeout)

    def list_paginated(
        self,
        prefix: Optional[str] = None,
        limit: Optional[int] = None,
        pagination_token: Optional[str] = None,
        namespace: Optional[str] = None,
        **kwargs,
    ) -> SimpleListResponse:
        """
        The list_paginated operation finds vectors based on an id prefix within a single namespace.
        It returns matching ids in a paginated form, with a pagination token to fetch the next page of results.
        This id list can then be passed to fetch or delete operations, depending on your use case.

        Consider using the `list` method to avoid having to handle pagination tokens manually.

        Examples:
            >>> results = index.list_paginated(prefix='99', limit=5, namespace='my_namespace')
            >>> [v.id for v in results.vectors]
            ['99', '990', '991', '992', '993']
            >>> results.pagination.next
            eyJza2lwX3Bhc3QiOiI5OTMiLCJwcmVmaXgiOiI5OSJ9
            >>> next_results = index.list_paginated(prefix='99', limit=5, namespace='my_namespace', pagination_token=results.pagination.next)

        Args:
            prefix (Optional[str]): The id prefix to match. If unspecified, an empty string prefix will
                                    be used with the effect of listing all ids in a namespace [optional]
            limit (Optional[int]): The maximum number of ids to return. If unspecified, the server will use a default value. [optional]
            pagination_token (Optional[str]): A token needed to fetch the next page of results. This token is returned
                in the response if additional results are available. [optional]
            namespace (Optional[str]): The namespace to fetch vectors from. If not specified, the default namespace is used. [optional]

        Returns: SimpleListResponse object which contains the list of ids, the namespace name, pagination information, and usage showing the number of read_units consumed.
        """
        args_dict = self._parse_non_empty_args(
            [
                ("prefix", prefix),
                ("limit", limit),
                ("namespace", namespace),
                ("pagination_token", pagination_token),
            ]
        )
        request = ListRequest(**args_dict, **kwargs)
        timeout = kwargs.pop("timeout", None)
        response = self.runner.run(self.stub.List, request, timeout=timeout)

        if response.pagination and response.pagination.next != "":
            pagination = Pagination(next=response.pagination.next)
        else:
            pagination = None

        return SimpleListResponse(
            namespace=response.namespace, vectors=response.vectors, pagination=pagination
        )

    def list(self, **kwargs):
        """
        The list operation accepts all of the same arguments as list_paginated, and returns a generator that yields
        a list of the matching vector ids in each page of results. It automatically handles pagination tokens on your
        behalf.

        Examples:
            >>> for ids in index.list(prefix='99', limit=5, namespace='my_namespace'):
            >>>     print(ids)
            ['99', '990', '991', '992', '993']
            ['994', '995', '996', '997', '998']
            ['999']

        Args:
            prefix (Optional[str]): The id prefix to match. If unspecified, an empty string prefix will
                                    be used with the effect of listing all ids in a namespace [optional]
            limit (Optional[int]): The maximum number of ids to return. If unspecified, the server will use a default value. [optional]
            pagination_token (Optional[str]): A token needed to fetch the next page of results. This token is returned
                in the response if additional results are available. [optional]
            namespace (Optional[str]): The namespace to fetch vectors from. If not specified, the default namespace is used. [optional]
        """
        done = False
        while not done:
            try:
                results = self.list_paginated(**kwargs)
            except Exception as e:
                raise e

            if len(results.vectors) > 0:
                yield [v.id for v in results.vectors]

            if results.pagination and results.pagination.next:
                kwargs.update({"pagination_token": results.pagination.next})
            else:
                done = True

    def describe_index_stats(
        self, filter: Optional[FilterTypedDict] = None, **kwargs
    ) -> DescribeIndexStatsResponse:
        """
        The DescribeIndexStats operation returns statistics about the index's contents.
        For example: The vector count per namespace and the number of dimensions.

        Examples:
            >>> index.describe_index_stats()
            >>> index.describe_index_stats(filter={'key': 'value'})

        Args:
            filter (Dict[str, Union[str, float, int, bool, List, dict]]):
            If this parameter is present, the operation only returns statistics for vectors that satisfy the filter.
            See https://www.pinecone.io/docs/metadata-filtering/.. [optional]

        Returns: DescribeIndexStatsResponse object which contains stats about the index.
        """
        if filter is not None:
            filter_struct = dict_to_proto_struct(filter)
        else:
            filter_struct = None
        args_dict = self._parse_non_empty_args([("filter", filter_struct)])
        timeout = kwargs.pop("timeout", None)

        request = DescribeIndexStatsRequest(**args_dict)
        response = self.runner.run(self.stub.DescribeIndexStats, request, timeout=timeout)
        json_response = json_format.MessageToDict(response)
        return parse_stats_response(json_response)

    @staticmethod
    def _parse_non_empty_args(args: List[Tuple[str, Any]]) -> Dict[str, Any]:
        return {arg_name: val for arg_name, val in args if val is not None}
