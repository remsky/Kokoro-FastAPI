"""
Pinecone Inference API

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
from pinecone.core.openapi.inference.model.embed_request import EmbedRequest
from pinecone.core.openapi.inference.model.embeddings_list import EmbeddingsList
from pinecone.core.openapi.inference.model.error_response import ErrorResponse
from pinecone.core.openapi.inference.model.model_info import ModelInfo
from pinecone.core.openapi.inference.model.model_info_list import ModelInfoList
from pinecone.core.openapi.inference.model.rerank_request import RerankRequest
from pinecone.core.openapi.inference.model.rerank_result import RerankResult


class InferenceApi:
    """NOTE: This class is @generated using OpenAPI.

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

        def __embed(self, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Generate vectors  # noqa: E501

            Generate vector embeddings for input data. This endpoint uses [Pinecone Inference](https://docs.pinecone.io/guides/index-data/indexing-overview#vector-embedding).  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.embed(async_req=True)
            >>> result = thread.get()


            Keyword Args:
                embed_request (EmbedRequest): Generate embeddings for inputs. [optional]
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
                EmbeddingsList
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            return self.call_with_http_info(**kwargs)

        self.embed = _Endpoint(
            settings={
                "response_type": (EmbeddingsList,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/embed",
                "operation_id": "embed",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["embed_request"],
                "required": [],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"embed_request": (EmbedRequest,)},
                "attribute_map": {},
                "location_map": {"embed_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__embed,
        )

        def __get_model(self, model_name, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Get available model details.  # noqa: E501

            Get model details.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.get_model(model_name, async_req=True)
            >>> result = thread.get()

            Args:
                model_name (str): The name of the model to look up.

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
                ModelInfo
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["model_name"] = model_name
            return self.call_with_http_info(**kwargs)

        self.get_model = _Endpoint(
            settings={
                "response_type": (ModelInfo,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/models/{model_name}",
                "operation_id": "get_model",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["model_name"],
                "required": ["model_name"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"model_name": (str,)},
                "attribute_map": {"model_name": "model_name"},
                "location_map": {"model_name": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__get_model,
        )

        def __list_models(self, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Get available models.  # noqa: E501

            Get available models.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.list_models(async_req=True)
            >>> result = thread.get()


            Keyword Args:
                type (str): Filter models by type ('embed' or 'rerank'). [optional]
                vector_type (str): Filter embedding models by vector type ('dense' or 'sparse'). Only relevant when `type=embed`. [optional]
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
                ModelInfoList
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            return self.call_with_http_info(**kwargs)

        self.list_models = _Endpoint(
            settings={
                "response_type": (ModelInfoList,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/models",
                "operation_id": "list_models",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["type", "vector_type"],
                "required": [],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"type": (str,), "vector_type": (str,)},
                "attribute_map": {"type": "type", "vector_type": "vector_type"},
                "location_map": {"type": "query", "vector_type": "query"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_models,
        )

        def __rerank(self, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Rerank documents  # noqa: E501

            Rerank documents according to their relevance to a query.  For guidance and examples, see [Rerank results](https://docs.pinecone.io/guides/search/rerank-results).  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.rerank(async_req=True)
            >>> result = thread.get()


            Keyword Args:
                rerank_request (RerankRequest): Rerank documents for the given query [optional]
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
                RerankResult
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            return self.call_with_http_info(**kwargs)

        self.rerank = _Endpoint(
            settings={
                "response_type": (RerankResult,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/rerank",
                "operation_id": "rerank",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["rerank_request"],
                "required": [],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"rerank_request": (RerankRequest,)},
                "attribute_map": {},
                "location_map": {"rerank_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__rerank,
        )


class AsyncioInferenceApi:
    """NOTE: This class is @generated using OpenAPI

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = AsyncioApiClient()
        self.api_client = api_client

        async def __embed(self, **kwargs):
            """Generate vectors  # noqa: E501

            Generate vector embeddings for input data. This endpoint uses [Pinecone Inference](https://docs.pinecone.io/guides/index-data/indexing-overview#vector-embedding).  # noqa: E501



            Keyword Args:
                embed_request (EmbedRequest): Generate embeddings for inputs. [optional]
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
                EmbeddingsList
            """
            self._process_openapi_kwargs(kwargs)
            return await self.call_with_http_info(**kwargs)

        self.embed = _AsyncioEndpoint(
            settings={
                "response_type": (EmbeddingsList,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/embed",
                "operation_id": "embed",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["embed_request"],
                "required": [],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"embed_request": (EmbedRequest,)},
                "attribute_map": {},
                "location_map": {"embed_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__embed,
        )

        async def __get_model(self, model_name, **kwargs):
            """Get available model details.  # noqa: E501

            Get model details.  # noqa: E501


            Args:
                model_name (str): The name of the model to look up.

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
                ModelInfo
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["model_name"] = model_name
            return await self.call_with_http_info(**kwargs)

        self.get_model = _AsyncioEndpoint(
            settings={
                "response_type": (ModelInfo,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/models/{model_name}",
                "operation_id": "get_model",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["model_name"],
                "required": ["model_name"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"model_name": (str,)},
                "attribute_map": {"model_name": "model_name"},
                "location_map": {"model_name": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__get_model,
        )

        async def __list_models(self, **kwargs):
            """Get available models.  # noqa: E501

            Get available models.  # noqa: E501



            Keyword Args:
                type (str): Filter models by type ('embed' or 'rerank'). [optional]
                vector_type (str): Filter embedding models by vector type ('dense' or 'sparse'). Only relevant when `type=embed`. [optional]
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
                ModelInfoList
            """
            self._process_openapi_kwargs(kwargs)
            return await self.call_with_http_info(**kwargs)

        self.list_models = _AsyncioEndpoint(
            settings={
                "response_type": (ModelInfoList,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/models",
                "operation_id": "list_models",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["type", "vector_type"],
                "required": [],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"type": (str,), "vector_type": (str,)},
                "attribute_map": {"type": "type", "vector_type": "vector_type"},
                "location_map": {"type": "query", "vector_type": "query"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_models,
        )

        async def __rerank(self, **kwargs):
            """Rerank documents  # noqa: E501

            Rerank documents according to their relevance to a query.  For guidance and examples, see [Rerank results](https://docs.pinecone.io/guides/search/rerank-results).  # noqa: E501



            Keyword Args:
                rerank_request (RerankRequest): Rerank documents for the given query [optional]
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
                RerankResult
            """
            self._process_openapi_kwargs(kwargs)
            return await self.call_with_http_info(**kwargs)

        self.rerank = _AsyncioEndpoint(
            settings={
                "response_type": (RerankResult,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/rerank",
                "operation_id": "rerank",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["rerank_request"],
                "required": [],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"rerank_request": (RerankRequest,)},
                "attribute_map": {},
                "location_map": {"rerank_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__rerank,
        )
