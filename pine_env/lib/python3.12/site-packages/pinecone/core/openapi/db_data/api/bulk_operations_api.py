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
from pinecone.core.openapi.db_data.model.import_model import ImportModel
from pinecone.core.openapi.db_data.model.list_imports_response import ListImportsResponse
from pinecone.core.openapi.db_data.model.rpc_status import RpcStatus
from pinecone.core.openapi.db_data.model.start_import_request import StartImportRequest
from pinecone.core.openapi.db_data.model.start_import_response import StartImportResponse


class BulkOperationsApi:
    """NOTE: This class is @generated using OpenAPI.

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

        def __cancel_bulk_import(self, id, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Cancel an import  # noqa: E501

            Cancel an import operation if it is not yet finished. It has no effect if the operation is already finished.  For guidance and examples, see [Import data](https://docs.pinecone.io/guides/index-data/import-data).  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.cancel_bulk_import(id, async_req=True)
            >>> result = thread.get()

            Args:
                id (str): Unique identifier for the import operation.

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
            kwargs["id"] = id
            return self.call_with_http_info(**kwargs)

        self.cancel_bulk_import = _Endpoint(
            settings={
                "response_type": ({str: (bool, dict, float, int, list, str, none_type)},),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/bulk/imports/{id}",
                "operation_id": "cancel_bulk_import",
                "http_method": "DELETE",
                "servers": None,
            },
            params_map={
                "all": ["id"],
                "required": ["id"],
                "nullable": [],
                "enum": [],
                "validation": ["id"],
            },
            root_map={
                "validations": {("id",): {"max_length": 1000, "min_length": 1}},
                "allowed_values": {},
                "openapi_types": {"id": (str,)},
                "attribute_map": {"id": "id"},
                "location_map": {"id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__cancel_bulk_import,
        )

        def __describe_bulk_import(self, id, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Describe an import  # noqa: E501

            Return details of a specific import operation.  For guidance and examples, see [Import data](https://docs.pinecone.io/guides/index-data/import-data).  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.describe_bulk_import(id, async_req=True)
            >>> result = thread.get()

            Args:
                id (str): Unique identifier for the import operation.

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
                ImportModel
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["id"] = id
            return self.call_with_http_info(**kwargs)

        self.describe_bulk_import = _Endpoint(
            settings={
                "response_type": (ImportModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/bulk/imports/{id}",
                "operation_id": "describe_bulk_import",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["id"],
                "required": ["id"],
                "nullable": [],
                "enum": [],
                "validation": ["id"],
            },
            root_map={
                "validations": {("id",): {"max_length": 1000, "min_length": 1}},
                "allowed_values": {},
                "openapi_types": {"id": (str,)},
                "attribute_map": {"id": "id"},
                "location_map": {"id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__describe_bulk_import,
        )

        def __list_bulk_imports(self, **kwargs: ExtraOpenApiKwargsTypedDict):
            """List imports  # noqa: E501

            List all recent and ongoing import operations.  By default, `list_imports` returns up to 100 imports per page. If the `limit` parameter is set, `list` returns up to that number of imports instead. Whenever there are additional IDs to return, the response also includes a `pagination_token` that you can use to get the next batch of imports. When the response does not include a `pagination_token`, there are no more imports to return.  For guidance and examples, see [Import data](https://docs.pinecone.io/guides/index-data/import-data).  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.list_bulk_imports(async_req=True)
            >>> result = thread.get()


            Keyword Args:
                limit (int): Max number of operations to return per page. [optional]
                pagination_token (str): Pagination token to continue a previous listing operation. [optional]
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
                ListImportsResponse
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            return self.call_with_http_info(**kwargs)

        self.list_bulk_imports = _Endpoint(
            settings={
                "response_type": (ListImportsResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/bulk/imports",
                "operation_id": "list_bulk_imports",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["limit", "pagination_token"],
                "required": [],
                "nullable": [],
                "enum": [],
                "validation": ["limit"],
            },
            root_map={
                "validations": {("limit",): {"inclusive_maximum": 100, "inclusive_minimum": 1}},
                "allowed_values": {},
                "openapi_types": {"limit": (int,), "pagination_token": (str,)},
                "attribute_map": {"limit": "limit", "pagination_token": "paginationToken"},
                "location_map": {"limit": "query", "pagination_token": "query"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_bulk_imports,
        )

        def __start_bulk_import(self, start_import_request, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Start import  # noqa: E501

            Start an asynchronous import of vectors from object storage into an index.  For guidance and examples, see [Import data](https://docs.pinecone.io/guides/index-data/import-data).  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.start_bulk_import(start_import_request, async_req=True)
            >>> result = thread.get()

            Args:
                start_import_request (StartImportRequest):

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
                StartImportResponse
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["start_import_request"] = start_import_request
            return self.call_with_http_info(**kwargs)

        self.start_bulk_import = _Endpoint(
            settings={
                "response_type": (StartImportResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/bulk/imports",
                "operation_id": "start_bulk_import",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["start_import_request"],
                "required": ["start_import_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"start_import_request": (StartImportRequest,)},
                "attribute_map": {},
                "location_map": {"start_import_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__start_bulk_import,
        )


class AsyncioBulkOperationsApi:
    """NOTE: This class is @generated using OpenAPI

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = AsyncioApiClient()
        self.api_client = api_client

        async def __cancel_bulk_import(self, id, **kwargs):
            """Cancel an import  # noqa: E501

            Cancel an import operation if it is not yet finished. It has no effect if the operation is already finished.  For guidance and examples, see [Import data](https://docs.pinecone.io/guides/index-data/import-data).  # noqa: E501


            Args:
                id (str): Unique identifier for the import operation.

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
            kwargs["id"] = id
            return await self.call_with_http_info(**kwargs)

        self.cancel_bulk_import = _AsyncioEndpoint(
            settings={
                "response_type": ({str: (bool, dict, float, int, list, str, none_type)},),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/bulk/imports/{id}",
                "operation_id": "cancel_bulk_import",
                "http_method": "DELETE",
                "servers": None,
            },
            params_map={
                "all": ["id"],
                "required": ["id"],
                "nullable": [],
                "enum": [],
                "validation": ["id"],
            },
            root_map={
                "validations": {("id",): {"max_length": 1000, "min_length": 1}},
                "allowed_values": {},
                "openapi_types": {"id": (str,)},
                "attribute_map": {"id": "id"},
                "location_map": {"id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__cancel_bulk_import,
        )

        async def __describe_bulk_import(self, id, **kwargs):
            """Describe an import  # noqa: E501

            Return details of a specific import operation.  For guidance and examples, see [Import data](https://docs.pinecone.io/guides/index-data/import-data).  # noqa: E501


            Args:
                id (str): Unique identifier for the import operation.

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
                ImportModel
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["id"] = id
            return await self.call_with_http_info(**kwargs)

        self.describe_bulk_import = _AsyncioEndpoint(
            settings={
                "response_type": (ImportModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/bulk/imports/{id}",
                "operation_id": "describe_bulk_import",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["id"],
                "required": ["id"],
                "nullable": [],
                "enum": [],
                "validation": ["id"],
            },
            root_map={
                "validations": {("id",): {"max_length": 1000, "min_length": 1}},
                "allowed_values": {},
                "openapi_types": {"id": (str,)},
                "attribute_map": {"id": "id"},
                "location_map": {"id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__describe_bulk_import,
        )

        async def __list_bulk_imports(self, **kwargs):
            """List imports  # noqa: E501

            List all recent and ongoing import operations.  By default, `list_imports` returns up to 100 imports per page. If the `limit` parameter is set, `list` returns up to that number of imports instead. Whenever there are additional IDs to return, the response also includes a `pagination_token` that you can use to get the next batch of imports. When the response does not include a `pagination_token`, there are no more imports to return.  For guidance and examples, see [Import data](https://docs.pinecone.io/guides/index-data/import-data).  # noqa: E501



            Keyword Args:
                limit (int): Max number of operations to return per page. [optional]
                pagination_token (str): Pagination token to continue a previous listing operation. [optional]
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
                ListImportsResponse
            """
            self._process_openapi_kwargs(kwargs)
            return await self.call_with_http_info(**kwargs)

        self.list_bulk_imports = _AsyncioEndpoint(
            settings={
                "response_type": (ListImportsResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/bulk/imports",
                "operation_id": "list_bulk_imports",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["limit", "pagination_token"],
                "required": [],
                "nullable": [],
                "enum": [],
                "validation": ["limit"],
            },
            root_map={
                "validations": {("limit",): {"inclusive_maximum": 100, "inclusive_minimum": 1}},
                "allowed_values": {},
                "openapi_types": {"limit": (int,), "pagination_token": (str,)},
                "attribute_map": {"limit": "limit", "pagination_token": "paginationToken"},
                "location_map": {"limit": "query", "pagination_token": "query"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_bulk_imports,
        )

        async def __start_bulk_import(self, start_import_request, **kwargs):
            """Start import  # noqa: E501

            Start an asynchronous import of vectors from object storage into an index.  For guidance and examples, see [Import data](https://docs.pinecone.io/guides/index-data/import-data).  # noqa: E501


            Args:
                start_import_request (StartImportRequest):

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
                StartImportResponse
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["start_import_request"] = start_import_request
            return await self.call_with_http_info(**kwargs)

        self.start_bulk_import = _AsyncioEndpoint(
            settings={
                "response_type": (StartImportResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/bulk/imports",
                "operation_id": "start_bulk_import",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["start_import_request"],
                "required": ["start_import_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"start_import_request": (StartImportRequest,)},
                "attribute_map": {},
                "location_map": {"start_import_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__start_bulk_import,
        )
