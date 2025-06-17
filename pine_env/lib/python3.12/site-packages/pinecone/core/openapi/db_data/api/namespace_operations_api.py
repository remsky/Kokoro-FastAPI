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
from pinecone.core.openapi.db_data.model.list_namespaces_response import ListNamespacesResponse
from pinecone.core.openapi.db_data.model.namespace_description import NamespaceDescription
from pinecone.core.openapi.db_data.model.rpc_status import RpcStatus


class NamespaceOperationsApi:
    """NOTE: This class is @generated using OpenAPI.

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

        def __delete_namespace(self, namespace, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Delete a namespace  # noqa: E501

            Delete a namespace from an index.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.delete_namespace(namespace, async_req=True)
            >>> result = thread.get()

            Args:
                namespace (str): The namespace to delete

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
            kwargs["namespace"] = namespace
            return self.call_with_http_info(**kwargs)

        self.delete_namespace = _Endpoint(
            settings={
                "response_type": ({str: (bool, dict, float, int, list, str, none_type)},),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/namespaces/{namespace}",
                "operation_id": "delete_namespace",
                "http_method": "DELETE",
                "servers": None,
            },
            params_map={
                "all": ["namespace"],
                "required": ["namespace"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"namespace": (str,)},
                "attribute_map": {"namespace": "namespace"},
                "location_map": {"namespace": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__delete_namespace,
        )

        def __describe_namespace(self, namespace, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Describe a namespace  # noqa: E501

            Describe a namespace within an index, showing the vector count within the namespace.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.describe_namespace(namespace, async_req=True)
            >>> result = thread.get()

            Args:
                namespace (str): The namespace to describe

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
                NamespaceDescription
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["namespace"] = namespace
            return self.call_with_http_info(**kwargs)

        self.describe_namespace = _Endpoint(
            settings={
                "response_type": (NamespaceDescription,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/namespaces/{namespace}",
                "operation_id": "describe_namespace",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["namespace"],
                "required": ["namespace"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"namespace": (str,)},
                "attribute_map": {"namespace": "namespace"},
                "location_map": {"namespace": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__describe_namespace,
        )

        def __list_namespaces_operation(self, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Get list of all namespaces  # noqa: E501

            Get a list of all namespaces within an index.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.list_namespaces_operation(async_req=True)
            >>> result = thread.get()


            Keyword Args:
                limit (int): Max number namespaces to return per page. [optional]
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
                ListNamespacesResponse
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            return self.call_with_http_info(**kwargs)

        self.list_namespaces_operation = _Endpoint(
            settings={
                "response_type": (ListNamespacesResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/namespaces",
                "operation_id": "list_namespaces_operation",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["limit", "pagination_token"],
                "required": [],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"limit": (int,), "pagination_token": (str,)},
                "attribute_map": {"limit": "limit", "pagination_token": "paginationToken"},
                "location_map": {"limit": "query", "pagination_token": "query"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_namespaces_operation,
        )


class AsyncioNamespaceOperationsApi:
    """NOTE: This class is @generated using OpenAPI

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = AsyncioApiClient()
        self.api_client = api_client

        async def __delete_namespace(self, namespace, **kwargs):
            """Delete a namespace  # noqa: E501

            Delete a namespace from an index.  # noqa: E501


            Args:
                namespace (str): The namespace to delete

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
            kwargs["namespace"] = namespace
            return await self.call_with_http_info(**kwargs)

        self.delete_namespace = _AsyncioEndpoint(
            settings={
                "response_type": ({str: (bool, dict, float, int, list, str, none_type)},),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/namespaces/{namespace}",
                "operation_id": "delete_namespace",
                "http_method": "DELETE",
                "servers": None,
            },
            params_map={
                "all": ["namespace"],
                "required": ["namespace"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"namespace": (str,)},
                "attribute_map": {"namespace": "namespace"},
                "location_map": {"namespace": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__delete_namespace,
        )

        async def __describe_namespace(self, namespace, **kwargs):
            """Describe a namespace  # noqa: E501

            Describe a namespace within an index, showing the vector count within the namespace.  # noqa: E501


            Args:
                namespace (str): The namespace to describe

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
                NamespaceDescription
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["namespace"] = namespace
            return await self.call_with_http_info(**kwargs)

        self.describe_namespace = _AsyncioEndpoint(
            settings={
                "response_type": (NamespaceDescription,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/namespaces/{namespace}",
                "operation_id": "describe_namespace",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["namespace"],
                "required": ["namespace"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"namespace": (str,)},
                "attribute_map": {"namespace": "namespace"},
                "location_map": {"namespace": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__describe_namespace,
        )

        async def __list_namespaces_operation(self, **kwargs):
            """Get list of all namespaces  # noqa: E501

            Get a list of all namespaces within an index.  # noqa: E501



            Keyword Args:
                limit (int): Max number namespaces to return per page. [optional]
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
                ListNamespacesResponse
            """
            self._process_openapi_kwargs(kwargs)
            return await self.call_with_http_info(**kwargs)

        self.list_namespaces_operation = _AsyncioEndpoint(
            settings={
                "response_type": (ListNamespacesResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/namespaces",
                "operation_id": "list_namespaces_operation",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["limit", "pagination_token"],
                "required": [],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"limit": (int,), "pagination_token": (str,)},
                "attribute_map": {"limit": "limit", "pagination_token": "paginationToken"},
                "location_map": {"limit": "query", "pagination_token": "query"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_namespaces_operation,
        )
