"""
Pinecone Data Plane API

Pinecone is a vector database that makes it easy to search and retrieve billions of high-dimensional vectors.  # noqa: E501

This file is @generated using OpenAPI.

The version of the OpenAPI document: 2025-04
Contact: support@pinecone.io
"""

from pinecone.openapi_support import ApiClient, AsyncioApiClient
from pinecone.openapi_support.endpoint_utils import (
    ExtraOpenApiKwargsTypedDict,
    KwargsWithOpenApiKwargDefaultsTypedDict,
)
from pinecone.openapi_support.endpoint import Endpoint as _Endpoint, ExtraOpenApiKwargsTypedDict
from pinecone.openapi_support.asyncio_endpoint import AsyncioEndpoint as _AsyncioEndpoint
from pinecone.openapi_support.model_utils import (  # noqa: F401
    date,
    datetime,
    file_type,
    none_type,
    validate_and_convert_types,
)
from pinecone.core.openapi.db_data.model.delete_request import DeleteRequest
from pinecone.core.openapi.db_data.model.describe_index_stats_request import (
    DescribeIndexStatsRequest,
)
from pinecone.core.openapi.db_data.model.fetch_response import FetchResponse
from pinecone.core.openapi.db_data.model.index_description import IndexDescription
from pinecone.core.openapi.db_data.model.list_response import ListResponse
from pinecone.core.openapi.db_data.model.query_request import QueryRequest
from pinecone.core.openapi.db_data.model.query_response import QueryResponse
from pinecone.core.openapi.db_data.model.rpc_status import RpcStatus
from pinecone.core.openapi.db_data.model.search_records_request import SearchRecordsRequest
from pinecone.core.openapi.db_data.model.search_records_response import SearchRecordsResponse
from pinecone.core.openapi.db_data.model.update_request import UpdateRequest
from pinecone.core.openapi.db_data.model.upsert_record import UpsertRecord
from pinecone.core.openapi.db_data.model.upsert_request import UpsertRequest
from pinecone.core.openapi.db_data.model.upsert_response import UpsertResponse


class VectorOperationsApi:
    """NOTE: This class is @generated using OpenAPI.

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

        def __delete_vectors(self, delete_request, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Delete vectors  # noqa: E501

            Delete vectors by id from a single namespace.  For guidance and examples, see [Delete data](https://docs.pinecone.io/guides/manage-data/delete-data).  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.delete_vectors(delete_request, async_req=True)
            >>> result = thread.get()

            Args:
                delete_request (DeleteRequest):

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.
                async_req (bool): execute request asynchronously

            Returns:
                {str: (bool, dict, float, int, list, str, none_type)}
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["delete_request"] = delete_request
            return self.call_with_http_info(**kwargs)

        self.delete_vectors = _Endpoint(
            settings={
                "response_type": ({str: (bool, dict, float, int, list, str, none_type)},),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/vectors/delete",
                "operation_id": "delete_vectors",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["delete_request"],
                "required": ["delete_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"delete_request": (DeleteRequest,)},
                "attribute_map": {},
                "location_map": {"delete_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__delete_vectors,
        )

        def __describe_index_stats(
            self, describe_index_stats_request, **kwargs: ExtraOpenApiKwargsTypedDict
        ):
            """Get index stats  # noqa: E501

            Return statistics about the contents of an index, including the vector count per namespace, the number of dimensions, and the index fullness.  Serverless indexes scale automatically as needed, so index fullness is relevant only for pod-based indexes.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.describe_index_stats(describe_index_stats_request, async_req=True)
            >>> result = thread.get()

            Args:
                describe_index_stats_request (DescribeIndexStatsRequest):

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.
                async_req (bool): execute request asynchronously

            Returns:
                IndexDescription
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["describe_index_stats_request"] = describe_index_stats_request
            return self.call_with_http_info(**kwargs)

        self.describe_index_stats = _Endpoint(
            settings={
                "response_type": (IndexDescription,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/describe_index_stats",
                "operation_id": "describe_index_stats",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["describe_index_stats_request"],
                "required": ["describe_index_stats_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"describe_index_stats_request": (DescribeIndexStatsRequest,)},
                "attribute_map": {},
                "location_map": {"describe_index_stats_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__describe_index_stats,
        )

        def __fetch_vectors(self, ids, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Fetch vectors  # noqa: E501

            Look up and return vectors by ID from a single namespace. The returned vectors include the vector data and/or metadata.  For guidance and examples, see [Fetch data](https://docs.pinecone.io/guides/manage-data/fetch-data).  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.fetch_vectors(ids, async_req=True)
            >>> result = thread.get()

            Args:
                ids ([str]): The vector IDs to fetch. Does not accept values containing spaces.

            Keyword Args:
                namespace (str): [optional]
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.
                async_req (bool): execute request asynchronously

            Returns:
                FetchResponse
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["ids"] = ids
            return self.call_with_http_info(**kwargs)

        self.fetch_vectors = _Endpoint(
            settings={
                "response_type": (FetchResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/vectors/fetch",
                "operation_id": "fetch_vectors",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["ids", "namespace"],
                "required": ["ids"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"ids": ([str],), "namespace": (str,)},
                "attribute_map": {"ids": "ids", "namespace": "namespace"},
                "location_map": {"ids": "query", "namespace": "query"},
                "collection_format_map": {"ids": "multi"},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__fetch_vectors,
        )

        def __list_vectors(self, **kwargs: ExtraOpenApiKwargsTypedDict):
            """List vector IDs  # noqa: E501

            List the IDs of vectors in a single namespace of a serverless index. An optional prefix can be passed to limit the results to IDs with a common prefix.  Returns up to 100 IDs at a time by default in sorted order (bitwise \"C\" collation). If the `limit` parameter is set, `list` returns up to that number of IDs instead. Whenever there are additional IDs to return, the response also includes a `pagination_token` that you can use to get the next batch of IDs. When the response does not include a `pagination_token`, there are no more IDs to return.  For guidance and examples, see [List record IDs](https://docs.pinecone.io/guides/manage-data/list-record-ids).  **Note:** `list` is supported only for serverless indexes.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.list_vectors(async_req=True)
            >>> result = thread.get()


            Keyword Args:
                prefix (str): The vector IDs to fetch. Does not accept values containing spaces. [optional]
                limit (int): Max number of IDs to return per page. [optional]
                pagination_token (str): Pagination token to continue a previous listing operation. [optional]
                namespace (str): [optional]
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.
                async_req (bool): execute request asynchronously

            Returns:
                ListResponse
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            return self.call_with_http_info(**kwargs)

        self.list_vectors = _Endpoint(
            settings={
                "response_type": (ListResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/vectors/list",
                "operation_id": "list_vectors",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["prefix", "limit", "pagination_token", "namespace"],
                "required": [],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "prefix": (str,),
                    "limit": (int,),
                    "pagination_token": (str,),
                    "namespace": (str,),
                },
                "attribute_map": {
                    "prefix": "prefix",
                    "limit": "limit",
                    "pagination_token": "paginationToken",
                    "namespace": "namespace",
                },
                "location_map": {
                    "prefix": "query",
                    "limit": "query",
                    "pagination_token": "query",
                    "namespace": "query",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_vectors,
        )

        def __query_vectors(self, query_request, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Search with a vector  # noqa: E501

            Search a namespace using a query vector. It retrieves the ids of the most similar items in a namespace, along with their similarity scores.  For guidance and examples, see [Search](https://docs.pinecone.io/guides/search/semantic-search).  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.query_vectors(query_request, async_req=True)
            >>> result = thread.get()

            Args:
                query_request (QueryRequest):

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.
                async_req (bool): execute request asynchronously

            Returns:
                QueryResponse
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["query_request"] = query_request
            return self.call_with_http_info(**kwargs)

        self.query_vectors = _Endpoint(
            settings={
                "response_type": (QueryResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/query",
                "operation_id": "query_vectors",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["query_request"],
                "required": ["query_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"query_request": (QueryRequest,)},
                "attribute_map": {},
                "location_map": {"query_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__query_vectors,
        )

        def __search_records_namespace(
            self, namespace, search_records_request, **kwargs: ExtraOpenApiKwargsTypedDict
        ):
            """Search with text  # noqa: E501

            Search a namespace with a query text, query vector, or record ID and return the most similar records, along with their similarity scores. Optionally, rerank the initial results based on their relevance to the query.   Searching with text is supported only for [indexes with integrated embedding](https://docs.pinecone.io/guides/indexes/create-an-index#integrated-embedding). Searching with a query vector or record ID is supported for all indexes.   For guidance and examples, see [Search](https://docs.pinecone.io/guides/search/semantic-search).  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.search_records_namespace(namespace, search_records_request, async_req=True)
            >>> result = thread.get()

            Args:
                namespace (str): The namespace to search.
                search_records_request (SearchRecordsRequest):

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.
                async_req (bool): execute request asynchronously

            Returns:
                SearchRecordsResponse
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["namespace"] = namespace
            kwargs["search_records_request"] = search_records_request
            return self.call_with_http_info(**kwargs)

        self.search_records_namespace = _Endpoint(
            settings={
                "response_type": (SearchRecordsResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/records/namespaces/{namespace}/search",
                "operation_id": "search_records_namespace",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["namespace", "search_records_request"],
                "required": ["namespace", "search_records_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "namespace": (str,),
                    "search_records_request": (SearchRecordsRequest,),
                },
                "attribute_map": {"namespace": "namespace"},
                "location_map": {"namespace": "path", "search_records_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__search_records_namespace,
        )

        def __update_vector(self, update_request, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Update a vector  # noqa: E501

            Update a vector in a namespace. If a value is included, it will overwrite the previous value. If a `set_metadata` is included, the values of the fields specified in it will be added or overwrite the previous value.  For guidance and examples, see [Update data](https://docs.pinecone.io/guides/manage-data/update-data).  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.update_vector(update_request, async_req=True)
            >>> result = thread.get()

            Args:
                update_request (UpdateRequest):

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.
                async_req (bool): execute request asynchronously

            Returns:
                {str: (bool, dict, float, int, list, str, none_type)}
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["update_request"] = update_request
            return self.call_with_http_info(**kwargs)

        self.update_vector = _Endpoint(
            settings={
                "response_type": ({str: (bool, dict, float, int, list, str, none_type)},),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/vectors/update",
                "operation_id": "update_vector",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["update_request"],
                "required": ["update_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"update_request": (UpdateRequest,)},
                "attribute_map": {},
                "location_map": {"update_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__update_vector,
        )

        def __upsert_records_namespace(
            self, namespace, upsert_record, **kwargs: ExtraOpenApiKwargsTypedDict
        ):
            """Upsert text  # noqa: E501

            Upsert text into a namespace. Pinecone converts the text to vectors automatically using the hosted embedding model associated with the index.  Upserting text is supported only for [indexes with integrated embedding](https://docs.pinecone.io/reference/api/2025-01/control-plane/create_for_model).  For guidance and examples, see [Upsert data](https://docs.pinecone.io/guides/index-data/upsert-data#upsert-text).  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.upsert_records_namespace(namespace, upsert_record, async_req=True)
            >>> result = thread.get()

            Args:
                namespace (str): The namespace to upsert records into.
                upsert_record ([UpsertRecord]):

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.
                async_req (bool): execute request asynchronously

            Returns:
                None
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["namespace"] = namespace
            kwargs["upsert_record"] = upsert_record
            return self.call_with_http_info(**kwargs)

        self.upsert_records_namespace = _Endpoint(
            settings={
                "response_type": None,
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/records/namespaces/{namespace}/upsert",
                "operation_id": "upsert_records_namespace",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["namespace", "upsert_record"],
                "required": ["namespace", "upsert_record"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"namespace": (str,), "upsert_record": ([UpsertRecord],)},
                "attribute_map": {"namespace": "namespace"},
                "location_map": {"namespace": "path", "upsert_record": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/x-ndjson"]},
            api_client=api_client,
            callable=__upsert_records_namespace,
        )

        def __upsert_vectors(self, upsert_request, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Upsert vectors  # noqa: E501

            Upsert vectors into a namespace. If a new value is upserted for an existing vector ID, it will overwrite the previous value.  For guidance and examples, see [Upsert data](https://docs.pinecone.io/guides/index-data/upsert-data#upsert-vectors).  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.upsert_vectors(upsert_request, async_req=True)
            >>> result = thread.get()

            Args:
                upsert_request (UpsertRequest):

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.
                async_req (bool): execute request asynchronously

            Returns:
                UpsertResponse
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["upsert_request"] = upsert_request
            return self.call_with_http_info(**kwargs)

        self.upsert_vectors = _Endpoint(
            settings={
                "response_type": (UpsertResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/vectors/upsert",
                "operation_id": "upsert_vectors",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["upsert_request"],
                "required": ["upsert_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"upsert_request": (UpsertRequest,)},
                "attribute_map": {},
                "location_map": {"upsert_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__upsert_vectors,
        )


class AsyncioVectorOperationsApi:
    """NOTE: This class is @generated using OpenAPI

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = AsyncioApiClient()
        self.api_client = api_client

        async def __delete_vectors(self, delete_request, **kwargs):
            """Delete vectors  # noqa: E501

            Delete vectors by id from a single namespace.  For guidance and examples, see [Delete data](https://docs.pinecone.io/guides/manage-data/delete-data).  # noqa: E501


            Args:
                delete_request (DeleteRequest):

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.

            Returns:
                {str: (bool, dict, float, int, list, str, none_type)}
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["delete_request"] = delete_request
            return await self.call_with_http_info(**kwargs)

        self.delete_vectors = _AsyncioEndpoint(
            settings={
                "response_type": ({str: (bool, dict, float, int, list, str, none_type)},),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/vectors/delete",
                "operation_id": "delete_vectors",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["delete_request"],
                "required": ["delete_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"delete_request": (DeleteRequest,)},
                "attribute_map": {},
                "location_map": {"delete_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__delete_vectors,
        )

        async def __describe_index_stats(self, describe_index_stats_request, **kwargs):
            """Get index stats  # noqa: E501

            Return statistics about the contents of an index, including the vector count per namespace, the number of dimensions, and the index fullness.  Serverless indexes scale automatically as needed, so index fullness is relevant only for pod-based indexes.  # noqa: E501


            Args:
                describe_index_stats_request (DescribeIndexStatsRequest):

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.

            Returns:
                IndexDescription
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["describe_index_stats_request"] = describe_index_stats_request
            return await self.call_with_http_info(**kwargs)

        self.describe_index_stats = _AsyncioEndpoint(
            settings={
                "response_type": (IndexDescription,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/describe_index_stats",
                "operation_id": "describe_index_stats",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["describe_index_stats_request"],
                "required": ["describe_index_stats_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"describe_index_stats_request": (DescribeIndexStatsRequest,)},
                "attribute_map": {},
                "location_map": {"describe_index_stats_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__describe_index_stats,
        )

        async def __fetch_vectors(self, ids, **kwargs):
            """Fetch vectors  # noqa: E501

            Look up and return vectors by ID from a single namespace. The returned vectors include the vector data and/or metadata.  For guidance and examples, see [Fetch data](https://docs.pinecone.io/guides/manage-data/fetch-data).  # noqa: E501


            Args:
                ids ([str]): The vector IDs to fetch. Does not accept values containing spaces.

            Keyword Args:
                namespace (str): [optional]
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.

            Returns:
                FetchResponse
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["ids"] = ids
            return await self.call_with_http_info(**kwargs)

        self.fetch_vectors = _AsyncioEndpoint(
            settings={
                "response_type": (FetchResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/vectors/fetch",
                "operation_id": "fetch_vectors",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["ids", "namespace"],
                "required": ["ids"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"ids": ([str],), "namespace": (str,)},
                "attribute_map": {"ids": "ids", "namespace": "namespace"},
                "location_map": {"ids": "query", "namespace": "query"},
                "collection_format_map": {"ids": "multi"},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__fetch_vectors,
        )

        async def __list_vectors(self, **kwargs):
            """List vector IDs  # noqa: E501

            List the IDs of vectors in a single namespace of a serverless index. An optional prefix can be passed to limit the results to IDs with a common prefix.  Returns up to 100 IDs at a time by default in sorted order (bitwise \"C\" collation). If the `limit` parameter is set, `list` returns up to that number of IDs instead. Whenever there are additional IDs to return, the response also includes a `pagination_token` that you can use to get the next batch of IDs. When the response does not include a `pagination_token`, there are no more IDs to return.  For guidance and examples, see [List record IDs](https://docs.pinecone.io/guides/manage-data/list-record-ids).  **Note:** `list` is supported only for serverless indexes.  # noqa: E501



            Keyword Args:
                prefix (str): The vector IDs to fetch. Does not accept values containing spaces. [optional]
                limit (int): Max number of IDs to return per page. [optional]
                pagination_token (str): Pagination token to continue a previous listing operation. [optional]
                namespace (str): [optional]
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.

            Returns:
                ListResponse
            """
            self._process_openapi_kwargs(kwargs)
            return await self.call_with_http_info(**kwargs)

        self.list_vectors = _AsyncioEndpoint(
            settings={
                "response_type": (ListResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/vectors/list",
                "operation_id": "list_vectors",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["prefix", "limit", "pagination_token", "namespace"],
                "required": [],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "prefix": (str,),
                    "limit": (int,),
                    "pagination_token": (str,),
                    "namespace": (str,),
                },
                "attribute_map": {
                    "prefix": "prefix",
                    "limit": "limit",
                    "pagination_token": "paginationToken",
                    "namespace": "namespace",
                },
                "location_map": {
                    "prefix": "query",
                    "limit": "query",
                    "pagination_token": "query",
                    "namespace": "query",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_vectors,
        )

        async def __query_vectors(self, query_request, **kwargs):
            """Search with a vector  # noqa: E501

            Search a namespace using a query vector. It retrieves the ids of the most similar items in a namespace, along with their similarity scores.  For guidance and examples, see [Search](https://docs.pinecone.io/guides/search/semantic-search).  # noqa: E501


            Args:
                query_request (QueryRequest):

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.

            Returns:
                QueryResponse
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["query_request"] = query_request
            return await self.call_with_http_info(**kwargs)

        self.query_vectors = _AsyncioEndpoint(
            settings={
                "response_type": (QueryResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/query",
                "operation_id": "query_vectors",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["query_request"],
                "required": ["query_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"query_request": (QueryRequest,)},
                "attribute_map": {},
                "location_map": {"query_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__query_vectors,
        )

        async def __search_records_namespace(self, namespace, search_records_request, **kwargs):
            """Search with text  # noqa: E501

            Search a namespace with a query text, query vector, or record ID and return the most similar records, along with their similarity scores. Optionally, rerank the initial results based on their relevance to the query.   Searching with text is supported only for [indexes with integrated embedding](https://docs.pinecone.io/guides/indexes/create-an-index#integrated-embedding). Searching with a query vector or record ID is supported for all indexes.   For guidance and examples, see [Search](https://docs.pinecone.io/guides/search/semantic-search).  # noqa: E501


            Args:
                namespace (str): The namespace to search.
                search_records_request (SearchRecordsRequest):

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.

            Returns:
                SearchRecordsResponse
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["namespace"] = namespace
            kwargs["search_records_request"] = search_records_request
            return await self.call_with_http_info(**kwargs)

        self.search_records_namespace = _AsyncioEndpoint(
            settings={
                "response_type": (SearchRecordsResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/records/namespaces/{namespace}/search",
                "operation_id": "search_records_namespace",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["namespace", "search_records_request"],
                "required": ["namespace", "search_records_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "namespace": (str,),
                    "search_records_request": (SearchRecordsRequest,),
                },
                "attribute_map": {"namespace": "namespace"},
                "location_map": {"namespace": "path", "search_records_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__search_records_namespace,
        )

        async def __update_vector(self, update_request, **kwargs):
            """Update a vector  # noqa: E501

            Update a vector in a namespace. If a value is included, it will overwrite the previous value. If a `set_metadata` is included, the values of the fields specified in it will be added or overwrite the previous value.  For guidance and examples, see [Update data](https://docs.pinecone.io/guides/manage-data/update-data).  # noqa: E501


            Args:
                update_request (UpdateRequest):

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.

            Returns:
                {str: (bool, dict, float, int, list, str, none_type)}
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["update_request"] = update_request
            return await self.call_with_http_info(**kwargs)

        self.update_vector = _AsyncioEndpoint(
            settings={
                "response_type": ({str: (bool, dict, float, int, list, str, none_type)},),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/vectors/update",
                "operation_id": "update_vector",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["update_request"],
                "required": ["update_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"update_request": (UpdateRequest,)},
                "attribute_map": {},
                "location_map": {"update_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__update_vector,
        )

        async def __upsert_records_namespace(self, namespace, upsert_record, **kwargs):
            """Upsert text  # noqa: E501

            Upsert text into a namespace. Pinecone converts the text to vectors automatically using the hosted embedding model associated with the index.  Upserting text is supported only for [indexes with integrated embedding](https://docs.pinecone.io/reference/api/2025-01/control-plane/create_for_model).  For guidance and examples, see [Upsert data](https://docs.pinecone.io/guides/index-data/upsert-data#upsert-text).  # noqa: E501


            Args:
                namespace (str): The namespace to upsert records into.
                upsert_record ([UpsertRecord]):

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.

            Returns:
                None
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["namespace"] = namespace
            kwargs["upsert_record"] = upsert_record
            return await self.call_with_http_info(**kwargs)

        self.upsert_records_namespace = _AsyncioEndpoint(
            settings={
                "response_type": None,
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/records/namespaces/{namespace}/upsert",
                "operation_id": "upsert_records_namespace",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["namespace", "upsert_record"],
                "required": ["namespace", "upsert_record"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"namespace": (str,), "upsert_record": ([UpsertRecord],)},
                "attribute_map": {"namespace": "namespace"},
                "location_map": {"namespace": "path", "upsert_record": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/x-ndjson"]},
            api_client=api_client,
            callable=__upsert_records_namespace,
        )

        async def __upsert_vectors(self, upsert_request, **kwargs):
            """Upsert vectors  # noqa: E501

            Upsert vectors into a namespace. If a new value is upserted for an existing vector ID, it will overwrite the previous value.  For guidance and examples, see [Upsert data](https://docs.pinecone.io/guides/index-data/upsert-data#upsert-vectors).  # noqa: E501


            Args:
                upsert_request (UpsertRequest):

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.

            Returns:
                UpsertResponse
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["upsert_request"] = upsert_request
            return await self.call_with_http_info(**kwargs)

        self.upsert_vectors = _AsyncioEndpoint(
            settings={
                "response_type": (UpsertResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/vectors/upsert",
                "operation_id": "upsert_vectors",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["upsert_request"],
                "required": ["upsert_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"upsert_request": (UpsertRequest,)},
                "attribute_map": {},
                "location_map": {"upsert_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__upsert_vectors,
        )
