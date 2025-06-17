from pinecone.utils.tqdm import tqdm
import warnings
import logging
import json
from typing import Union, List, Optional, Dict, Any, Literal, Iterator, TYPE_CHECKING

from pinecone.config import ConfigBuilder

from pinecone.openapi_support import ApiClient
from pinecone.core.openapi.db_data.api.vector_operations_api import VectorOperationsApi
from pinecone.core.openapi.db_data import API_VERSION
from pinecone.core.openapi.db_data.models import (
    QueryResponse,
    IndexDescription as DescribeIndexStatsResponse,
    UpsertResponse,
    ListResponse,
    SearchRecordsResponse,
)
from .dataclasses import Vector, SparseValues, FetchResponse, SearchQuery, SearchRerank
from .interfaces import IndexInterface
from .request_factory import IndexRequestFactory
from .types import (
    SparseVectorTypedDict,
    VectorTypedDict,
    VectorMetadataTypedDict,
    VectorTuple,
    VectorTupleWithMetadata,
    FilterTypedDict,
    SearchRerankTypedDict,
    SearchQueryTypedDict,
)
from ..utils import (
    setup_openapi_client,
    parse_non_empty_args,
    validate_and_convert_errors,
    filter_dict,
    PluginAware,
)
from .query_results_aggregator import QueryResultsAggregator, QueryNamespacesResults
from pinecone.openapi_support import OPENAPI_ENDPOINT_PARAMS

from multiprocessing.pool import ApplyResult
from multiprocessing import cpu_count
from concurrent.futures import as_completed


if TYPE_CHECKING:
    from pinecone.config import Config, OpenApiConfiguration
    from .resources.sync.bulk_import import BulkImportResource

    from pinecone.core.openapi.db_data.models import (
        StartImportResponse,
        ListImportsResponse,
        ImportModel,
    )

    from .resources.sync.bulk_import import ImportErrorMode

logger = logging.getLogger(__name__)
""" @private """


def parse_query_response(response: QueryResponse):
    """@private"""
    response._data_store.pop("results", None)
    return response


class Index(PluginAware, IndexInterface):
    """
    A client for interacting with a Pinecone index via REST API.
    For improved performance, use the Pinecone GRPC index client.
    """

    _bulk_import_resource: Optional["BulkImportResource"]
    """ @private """

    def __init__(
        self,
        api_key: str,
        host: str,
        pool_threads: Optional[int] = None,
        additional_headers: Optional[Dict[str, str]] = {},
        openapi_config=None,
        **kwargs,
    ):
        self._config = ConfigBuilder.build(
            api_key=api_key, host=host, additional_headers=additional_headers, **kwargs
        )
        """ @private """
        self._openapi_config = ConfigBuilder.build_openapi_config(self._config, openapi_config)
        """ @private """

        if pool_threads is None:
            self._pool_threads = 5 * cpu_count()
            """ @private """
        else:
            self._pool_threads = pool_threads
            """ @private """

        if kwargs.get("connection_pool_maxsize", None):
            self._openapi_config.connection_pool_maxsize = kwargs.get("connection_pool_maxsize")

        self._vector_api = setup_openapi_client(
            api_client_klass=ApiClient,
            api_klass=VectorOperationsApi,
            config=self._config,
            openapi_config=self._openapi_config,
            pool_threads=self._pool_threads,
            api_version=API_VERSION,
        )

        self._api_client = self._vector_api.api_client

        self._bulk_import_resource = None
        """ @private """

        # Pass the same api_client to the ImportFeatureMixin
        super().__init__(api_client=self._api_client)

    @property
    def config(self) -> "Config":
        """@private"""
        return self._config

    @property
    def openapi_config(self) -> "OpenApiConfiguration":
        """@private"""
        warnings.warn(
            "The `openapi_config` property has been renamed to `_openapi_config`. It is considered private and should not be used directly. This warning will become an error in a future version of the Pinecone Python SDK.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._openapi_config

    @property
    def pool_threads(self) -> int:
        """@private"""
        warnings.warn(
            "The `pool_threads` property has been renamed to `_pool_threads`. It is considered private and should not be used directly. This warning will become an error in a future version of the Pinecone Python SDK.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._pool_threads

    @property
    def bulk_import(self) -> "BulkImportResource":
        """@private"""
        if self._bulk_import_resource is None:
            from .resources.sync.bulk_import import BulkImportResource

            self._bulk_import_resource = BulkImportResource(api_client=self._api_client)
        return self._bulk_import_resource

    def _openapi_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return filter_dict(kwargs, OPENAPI_ENDPOINT_PARAMS)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._vector_api.api_client.close()

    def close(self):
        self._vector_api.api_client.close()

    @validate_and_convert_errors
    def upsert(
        self,
        vectors: Union[
            List[Vector], List[VectorTuple], List[VectorTupleWithMetadata], List[VectorTypedDict]
        ],
        namespace: Optional[str] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> UpsertResponse:
        _check_type = kwargs.pop("_check_type", True)

        if kwargs.get("async_req", False) and batch_size is not None:
            raise ValueError(
                "async_req is not supported when batch_size is provided."
                "To upsert in parallel, please follow: "
                "https://docs.pinecone.io/docs/insert-data#sending-upserts-in-parallel"
            )

        if batch_size is None:
            return self._upsert_batch(vectors, namespace, _check_type, **kwargs)

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        pbar = tqdm(total=len(vectors), disable=not show_progress, desc="Upserted vectors")
        total_upserted = 0
        for i in range(0, len(vectors), batch_size):
            batch_result = self._upsert_batch(
                vectors[i : i + batch_size], namespace, _check_type, **kwargs
            )
            pbar.update(batch_result.upserted_count)
            # we can't use here pbar.n for the case show_progress=False
            total_upserted += batch_result.upserted_count

        return UpsertResponse(upserted_count=total_upserted)

    def _upsert_batch(
        self,
        vectors: Union[
            List[Vector], List[VectorTuple], List[VectorTupleWithMetadata], List[VectorTypedDict]
        ],
        namespace: Optional[str],
        _check_type: bool,
        **kwargs,
    ) -> UpsertResponse:
        return self._vector_api.upsert_vectors(
            IndexRequestFactory.upsert_request(vectors, namespace, _check_type, **kwargs),
            **self._openapi_kwargs(kwargs),
        )

    @staticmethod
    def _iter_dataframe(df, batch_size):
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i : i + batch_size].to_dict(orient="records")
            yield batch

    @validate_and_convert_errors
    def upsert_from_dataframe(
        self, df, namespace: Optional[str] = None, batch_size: int = 500, show_progress: bool = True
    ) -> UpsertResponse:
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
            res = self.upsert(vectors=chunk, namespace=namespace)
            pbar.update(len(chunk))
            results.append(res)

        upserted_count = 0
        for res in results:
            upserted_count += res.upserted_count

        return UpsertResponse(upserted_count=upserted_count)

    def upsert_records(self, namespace: str, records: List[Dict]):
        args = IndexRequestFactory.upsert_records_args(namespace=namespace, records=records)
        self._vector_api.upsert_records_namespace(**args)

    @validate_and_convert_errors
    def search(
        self,
        namespace: str,
        query: Union[SearchQueryTypedDict, SearchQuery],
        rerank: Optional[Union[SearchRerankTypedDict, SearchRerank]] = None,
        fields: Optional[List[str]] = ["*"],  # Default to returning all fields
    ) -> SearchRecordsResponse:
        if namespace is None:
            raise Exception("Namespace is required when searching records")

        request = IndexRequestFactory.search_request(query=query, rerank=rerank, fields=fields)

        return self._vector_api.search_records_namespace(namespace, request)

    @validate_and_convert_errors
    def search_records(
        self,
        namespace: str,
        query: Union[SearchQueryTypedDict, SearchQuery],
        rerank: Optional[Union[SearchRerankTypedDict, SearchRerank]] = None,
        fields: Optional[List[str]] = ["*"],  # Default to returning all fields
    ) -> SearchRecordsResponse:
        return self.search(namespace, query=query, rerank=rerank, fields=fields)

    @validate_and_convert_errors
    def delete(
        self,
        ids: Optional[List[str]] = None,
        delete_all: Optional[bool] = None,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        return self._vector_api.delete_vectors(
            IndexRequestFactory.delete_request(
                ids=ids, delete_all=delete_all, namespace=namespace, filter=filter, **kwargs
            ),
            **self._openapi_kwargs(kwargs),
        )

    @validate_and_convert_errors
    def fetch(self, ids: List[str], namespace: Optional[str] = None, **kwargs) -> FetchResponse:
        args_dict = parse_non_empty_args([("namespace", namespace)])
        result = self._vector_api.fetch_vectors(ids=ids, **args_dict, **kwargs)
        return FetchResponse(
            namespace=result.namespace,
            vectors={k: Vector.from_dict(v) for k, v in result.vectors.items()},
            usage=result.usage,
        )

    @validate_and_convert_errors
    def query(
        self,
        *args,
        top_k: int,
        vector: Optional[List[float]] = None,
        id: Optional[str] = None,
        namespace: Optional[str] = None,
        filter: Optional[FilterTypedDict] = None,
        include_values: Optional[bool] = None,
        include_metadata: Optional[bool] = None,
        sparse_vector: Optional[Union[SparseValues, SparseVectorTypedDict]] = None,
        **kwargs,
    ) -> Union[QueryResponse, ApplyResult]:
        response = self._query(
            *args,
            top_k=top_k,
            vector=vector,
            id=id,
            namespace=namespace,
            filter=filter,
            include_values=include_values,
            include_metadata=include_metadata,
            sparse_vector=sparse_vector,
            **kwargs,
        )

        if kwargs.get("async_req", False) or kwargs.get("async_threadpool_executor", False):
            return response
        else:
            return parse_query_response(response)

    def _query(
        self,
        *args,
        top_k: int,
        vector: Optional[List[float]] = None,
        id: Optional[str] = None,
        namespace: Optional[str] = None,
        filter: Optional[FilterTypedDict] = None,
        include_values: Optional[bool] = None,
        include_metadata: Optional[bool] = None,
        sparse_vector: Optional[Union[SparseValues, SparseVectorTypedDict]] = None,
        **kwargs,
    ) -> QueryResponse:
        if len(args) > 0:
            raise ValueError(
                "The argument order for `query()` has changed; please use keyword arguments instead of positional arguments. Example: index.query(vector=[0.1, 0.2, 0.3], top_k=10, namespace='my_namespace')"
            )

        if top_k < 1:
            raise ValueError("top_k must be a positive integer")

        request = IndexRequestFactory.query_request(
            top_k=top_k,
            vector=vector,
            id=id,
            namespace=namespace,
            filter=filter,
            include_values=include_values,
            include_metadata=include_metadata,
            sparse_vector=sparse_vector,
            **kwargs,
        )
        return self._vector_api.query_vectors(request, **self._openapi_kwargs(kwargs))

    @validate_and_convert_errors
    def query_namespaces(
        self,
        vector: Optional[List[float]],
        namespaces: List[str],
        metric: Literal["cosine", "euclidean", "dotproduct"],
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = None,
        include_values: Optional[bool] = None,
        include_metadata: Optional[bool] = None,
        sparse_vector: Optional[
            Union[SparseValues, Dict[str, Union[List[float], List[int]]]]
        ] = None,
        **kwargs,
    ) -> QueryNamespacesResults:
        if namespaces is None or len(namespaces) == 0:
            raise ValueError("At least one namespace must be specified")
        if sparse_vector is None and vector is not None and len(vector) == 0:
            # If querying with a vector, it must not be empty
            raise ValueError("Query vector must not be empty")

        overall_topk = top_k if top_k is not None else 10
        aggregator = QueryResultsAggregator(top_k=overall_topk, metric=metric)

        target_namespaces = set(namespaces)  # dedup namespaces
        async_futures = [
            self.query(
                vector=vector,
                namespace=ns,
                top_k=overall_topk,
                filter=filter,
                include_values=include_values,
                include_metadata=include_metadata,
                sparse_vector=sparse_vector,
                async_threadpool_executor=True,
                _preload_content=False,
                **kwargs,
            )
            for ns in target_namespaces
        ]

        for result in as_completed(async_futures):
            raw_result = result.result()
            response = json.loads(raw_result.data.decode("utf-8"))
            aggregator.add_results(response)

        final_results = aggregator.get_results()
        return final_results

    @validate_and_convert_errors
    def update(
        self,
        id: str,
        values: Optional[List[float]] = None,
        set_metadata: Optional[VectorMetadataTypedDict] = None,
        namespace: Optional[str] = None,
        sparse_values: Optional[Union[SparseValues, SparseVectorTypedDict]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        return self._vector_api.update_vector(
            IndexRequestFactory.update_request(
                id=id,
                values=values,
                set_metadata=set_metadata,
                namespace=namespace,
                sparse_values=sparse_values,
                **kwargs,
            ),
            **self._openapi_kwargs(kwargs),
        )

    @validate_and_convert_errors
    def describe_index_stats(
        self, filter: Optional[FilterTypedDict] = None, **kwargs
    ) -> DescribeIndexStatsResponse:
        return self._vector_api.describe_index_stats(
            IndexRequestFactory.describe_index_stats_request(filter, **kwargs),
            **self._openapi_kwargs(kwargs),
        )

    @validate_and_convert_errors
    def list_paginated(
        self,
        prefix: Optional[str] = None,
        limit: Optional[int] = None,
        pagination_token: Optional[str] = None,
        namespace: Optional[str] = None,
        **kwargs,
    ) -> ListResponse:
        args_dict = IndexRequestFactory.list_paginated_args(
            prefix=prefix,
            limit=limit,
            pagination_token=pagination_token,
            namespace=namespace,
            **kwargs,
        )
        return self._vector_api.list_vectors(**args_dict, **kwargs)

    @validate_and_convert_errors
    def list(self, **kwargs):
        done = False
        while not done:
            results = self.list_paginated(**kwargs)
            if len(results.vectors) > 0:
                yield [v.id for v in results.vectors]

            if results.pagination:
                kwargs.update({"pagination_token": results.pagination.next})
            else:
                done = True

    @validate_and_convert_errors
    def start_import(
        self,
        uri: str,
        integration_id: Optional[str] = None,
        error_mode: Optional[
            Union["ImportErrorMode", Literal["CONTINUE", "ABORT"], str]
        ] = "CONTINUE",
    ) -> "StartImportResponse":
        """
        Args:
            uri (str): The URI of the data to import. The URI must start with the scheme of a supported storage provider.
            integration_id (Optional[str], optional): If your bucket requires authentication to access, you need to pass the id of your storage integration using this property. Defaults to None.
            error_mode: Defaults to "CONTINUE". If set to "CONTINUE", the import operation will continue even if some
                records fail to import. Pass "ABORT" to stop the import operation if any records fail to import.

        Returns:
            `StartImportResponse`: Contains the id of the import operation.

        Import data from a storage provider into an index. The uri must start with the scheme of a supported
        storage provider. For buckets that are not publicly readable, you will also need to separately configure
        a storage integration and pass the integration id.

        Examples:
            >>> from pinecone import Pinecone
            >>> index = Pinecone().Index('my-index')
            >>> index.start_import(uri="s3://bucket-name/path/to/data.parquet")
            { id: "1" }
        """
        return self.bulk_import.start(uri=uri, integration_id=integration_id, error_mode=error_mode)

    @validate_and_convert_errors
    def list_imports(self, **kwargs) -> Iterator["ImportModel"]:
        """
        Args:
            limit (Optional[int]): The maximum number of operations to fetch in each network call. If unspecified, the server will use a default value. [optional]
            pagination_token (Optional[str]): When there are multiple pages of results, a pagination token is returned in the response. The token can be used
                to fetch the next page of results. [optional]

        Returns:
            Returns a generator that yields each import operation. It automatically handles pagination tokens on your behalf so you can
            easily iterate over all results. The `list_imports` method accepts all of the same arguments as list_imports_paginated

        ```python
        for op in index.list_imports():
            print(op)
        ```

        You can convert the generator into a list by wrapping the generator in a call to the built-in `list` function:

        ```python
        operations = list(index.list_imports())
        ```

        You should be cautious with this approach because it will fetch all operations at once, which could be a large number
        of network calls and a lot of memory to hold the results.
        """
        for i in self.bulk_import.list(**kwargs):
            yield i

    @validate_and_convert_errors
    def list_imports_paginated(
        self, limit: Optional[int] = None, pagination_token: Optional[str] = None, **kwargs
    ) -> "ListImportsResponse":
        """
        Args:
            limit (Optional[int]): The maximum number of ids to return. If unspecified, the server will use a default value. [optional]
            pagination_token (Optional[str]): A token needed to fetch the next page of results. This token is returned
                in the response if additional results are available. [optional]

        Returns: ListImportsResponse object which contains the list of operations as ImportModel objects, pagination information,
            and usage showing the number of read_units consumed.

        The list_imports_paginated operation returns information about import operations.
        It returns operations in a paginated form, with a pagination token to fetch the next page of results.

        Consider using the `list_imports` method to avoid having to handle pagination tokens manually.

        Examples:
            >>> results = index.list_imports_paginated(limit=5)
            >>> results.pagination.next
            eyJza2lwX3Bhc3QiOiI5OTMiLCJwcmVmaXgiOiI5OSJ9
            >>> results.data[0]
            {
                "id": "6",
                "uri": "s3://dev-bulk-import-datasets-pub/10-records-dim-10/",
                "status": "Completed",
                "percent_complete": 100.0,
                "records_imported": 10,
                "created_at": "2024-09-06T14:52:02.567776+00:00",
                "finished_at": "2024-09-06T14:52:28.130717+00:00"
            }
            >>> next_results = index.list_imports_paginated(limit=5, pagination_token=results.pagination.next)
        """
        return self.bulk_import.list_paginated(
            limit=limit, pagination_token=pagination_token, **kwargs
        )

    @validate_and_convert_errors
    def describe_import(self, id: str) -> "ImportModel":
        """
        Args:
            id (str): The id of the import operation. This value is returned when
                starting an import, and can be looked up using list_imports.

        Returns:
            `ImportModel`: An object containing operation id, status, and other details.

        describe_import is used to get detailed information about a specific import operation.
        """
        return self.bulk_import.describe(id=id)

    @validate_and_convert_errors
    def cancel_import(self, id: str):
        """Cancel an import operation.

        Args:
            id (str): The id of the import operation to cancel.
        """
        return self.bulk_import.cancel(id=id)
