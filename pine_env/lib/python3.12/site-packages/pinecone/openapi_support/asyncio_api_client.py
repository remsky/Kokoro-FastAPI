import json
import io
from urllib3.fields import RequestField
import logging

from typing import Optional, List, Tuple, Dict, Any, Union


from .rest_aiohttp import AiohttpRestClient
from ..config.openapi_configuration import Configuration
from .exceptions import PineconeApiValueError, PineconeApiException
from .api_client_utils import (
    parameters_to_tuples,
    files_parameters,
    parameters_to_multipart,
    process_params,
    process_query_params,
    build_request_url,
)
from .serializer import Serializer
from .deserializer import Deserializer
from .auth_util import AuthUtil

logger = logging.getLogger(__name__)
""" @private """


class AsyncioApiClient(object):
    """Generic async API client for OpenAPI client library builds.

    :param configuration: .Configuration object for this client
    """

    def __init__(self, configuration=None, **kwargs) -> None:
        if configuration is None:
            configuration = Configuration.get_default_copy()
        self.configuration = configuration

        self.rest_client = AiohttpRestClient(configuration)

        self.default_headers: Dict[str, str] = {}
        # Set default User-Agent.
        self.user_agent = "OpenAPI-Generator/1.0.0/python"

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()

    async def close(self):
        logger.debug("Closing the aiohttp client")
        await self.rest_client.close()

    @property
    def user_agent(self):
        """User agent for this API client"""
        return self.default_headers["User-Agent"]

    @user_agent.setter
    def user_agent(self, value):
        self.default_headers["User-Agent"] = value

    def set_default_header(self, header_name, header_value):
        self.default_headers[header_name] = header_value

    async def __call_api(
        self,
        resource_path: str,
        method: str,
        path_params: Optional[Dict[str, Any]] = None,
        query_params: Optional[List[Tuple[str, Any]]] = None,
        header_params: Optional[Dict[str, Any]] = None,
        body: Optional[Any] = None,
        post_params: Optional[List[Tuple[str, Any]]] = None,
        files: Optional[Dict[str, List[io.IOBase]]] = None,
        response_type: Optional[Tuple[Any]] = None,
        auth_settings: Optional[List[str]] = None,
        _return_http_data_only: Optional[bool] = None,
        collection_formats: Optional[Dict[str, str]] = None,
        _preload_content: bool = True,
        _request_timeout: Optional[Union[int, float, Tuple]] = None,
        _host: Optional[str] = None,
        _check_type: Optional[bool] = None,
    ):
        config = self.configuration

        path_params = path_params or {}
        query_params = query_params or []
        header_params = header_params or {}
        post_params = post_params or []
        files = files or {}
        collection_formats = collection_formats or {}

        processed_header_params, processed_path_params, sanitized_path_params = process_params(
            default_headers=self.default_headers,
            header_params=header_params,
            path_params=path_params,
            collection_formats=collection_formats,
        )

        processed_query_params = process_query_params(query_params, collection_formats)

        # post parameters
        if post_params or files:
            post_params = post_params if post_params else []
            sanitized_post_params = Serializer.sanitize_for_serialization(post_params)
            if sanitized_path_params:
                processed_post_params = parameters_to_tuples(
                    sanitized_post_params, collection_formats
                )
                processed_post_params.extend(files_parameters(files))
            if processed_header_params["Content-Type"].startswith("multipart"):
                processed_post_params = parameters_to_multipart(sanitized_post_params, (dict))
        else:
            processed_post_params = None

        # body
        if body:
            body = Serializer.sanitize_for_serialization(body)

        # auth setting
        AuthUtil.update_params_for_auth(
            configuration=self.configuration,
            endpoint_auth_settings=auth_settings,
            headers=processed_header_params,
            querys=processed_query_params,
        )

        url = build_request_url(
            config=config,
            processed_path_params=processed_path_params,
            resource_path=resource_path,
            _host=_host,
        )

        try:
            # perform request and return response
            response_data = await self.request(
                method,
                url,
                query_params=processed_query_params,
                headers=processed_header_params,
                post_params=processed_post_params,
                body=body,
                _preload_content=_preload_content,
                _request_timeout=_request_timeout,
            )
        except PineconeApiException as e:
            e.body = e.body.decode("utf-8")
            raise e

        self.last_response = response_data

        return_data = response_data

        if not _preload_content:
            return return_data

        # deserialize response data
        if response_type:
            Deserializer.decode_response(response_type=response_type, response=response_data)
            return_data = Deserializer.deserialize(
                response_data, response_type, self.configuration, _check_type
            )
        else:
            return_data = None

        if _return_http_data_only:
            return return_data
        else:
            return (return_data, response_data.status, response_data.getheaders())

    def parameters_to_multipart(self, params, collection_types):
        """Get parameters as list of tuples, formatting as json if value is collection_types

        :param params: Parameters as list of two-tuples
        :param dict collection_types: Parameter collection types
        :return: Parameters as list of tuple or urllib3.fields.RequestField
        """
        new_params = []
        if collection_types is None:
            collection_types = dict
        for k, v in params.items() if isinstance(params, dict) else params:  # noqa: E501
            if isinstance(
                v, collection_types
            ):  # v is instance of collection_type, formatting as application/json
                v = json.dumps(v, ensure_ascii=False).encode("utf-8")
                field = RequestField(k, v)
                field.make_multipart(content_type="application/json; charset=utf-8")
                new_params.append(field)
            else:
                new_params.append((k, v))
        return new_params

    async def call_api(
        self,
        resource_path: str,
        method: str,
        path_params: Optional[Dict[str, Any]] = None,
        query_params: Optional[List[Tuple[str, Any]]] = None,
        header_params: Optional[Dict[str, Any]] = None,
        body: Optional[Any] = None,
        post_params: Optional[List[Tuple[str, Any]]] = None,
        files: Optional[Dict[str, List[io.IOBase]]] = None,
        response_type: Optional[Tuple[Any]] = None,
        auth_settings: Optional[List[str]] = None,
        _return_http_data_only: Optional[bool] = None,
        collection_formats: Optional[Dict[str, str]] = None,
        _preload_content: bool = True,
        _request_timeout: Optional[Union[int, float, Tuple]] = None,
        _host: Optional[str] = None,
        _check_type: Optional[bool] = None,
    ):
        """Makes the HTTP request (synchronous) and returns deserialized data.

        :param resource_path: Path to method endpoint.
        :param method: Method to call.
        :param path_params: Path parameters in the url.
        :param query_params: Query parameters in the url.
        :param header_params: Header parameters to be
            placed in the request header.
        :param body: Request body.
        :param post_params dict: Request post form parameters,
            for `application/x-www-form-urlencoded`, `multipart/form-data`.
        :param auth_settings list: Auth Settings names for the request.
        :param response_type: For the response, a tuple containing:
            valid classes
            a list containing valid classes (for list schemas)
            a dict containing a tuple of valid classes as the value
            Example values:
            (str,)
            (Pet,)
            (float, none_type)
            ([int, none_type],)
            ({str: (bool, str, int, float, date, datetime, str, none_type)},)
        :param files: key -> field name, value -> a list of open file
            objects for `multipart/form-data`.
        :type files: dict
        :param _return_http_data_only: response data without head status code
                                       and headers
        :type _return_http_data_only: bool, optional
        :param collection_formats: dict of collection formats for path, query,
            header, and post parameters.
        :type collection_formats: dict, optional
        :param _preload_content: if False, the urllib3.HTTPResponse object will
                                 be returned without reading/decoding response
                                 data. Default is True.
        :type _preload_content: bool, optional
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :param _check_type: boolean describing if the data back from the server
            should have its type checked.
        :type _check_type: bool, optional
        """
        return await self.__call_api(
            resource_path,
            method,
            path_params,
            query_params,
            header_params,
            body,
            post_params,
            files,
            response_type,
            auth_settings,
            _return_http_data_only,
            collection_formats,
            _preload_content,
            _request_timeout,
            _host,
            _check_type,
        )

    async def request(
        self,
        method,
        url,
        query_params=None,
        headers=None,
        post_params=None,
        body=None,
        _preload_content=True,
        _request_timeout=None,
    ):
        """Makes the HTTP request using RESTClient."""
        if method == "GET":
            return await self.rest_client.GET(
                url,
                query_params=query_params,
                _preload_content=_preload_content,
                _request_timeout=_request_timeout,
                headers=headers,
            )
        elif method == "HEAD":
            return await self.rest_client.HEAD(
                url,
                query_params=query_params,
                _preload_content=_preload_content,
                _request_timeout=_request_timeout,
                headers=headers,
            )
        elif method == "OPTIONS":
            return await self.rest_client.OPTIONS(
                url,
                query_params=query_params,
                headers=headers,
                post_params=post_params,
                _preload_content=_preload_content,
                _request_timeout=_request_timeout,
                body=body,
            )
        elif method == "POST":
            return await self.rest_client.POST(
                url,
                query_params=query_params,
                headers=headers,
                post_params=post_params,
                _preload_content=_preload_content,
                _request_timeout=_request_timeout,
                body=body,
            )
        elif method == "PUT":
            return await self.rest_client.PUT(
                url,
                query_params=query_params,
                headers=headers,
                post_params=post_params,
                _preload_content=_preload_content,
                _request_timeout=_request_timeout,
                body=body,
            )
        elif method == "PATCH":
            return await self.rest_client.PATCH(
                url,
                query_params=query_params,
                headers=headers,
                post_params=post_params,
                _preload_content=_preload_content,
                _request_timeout=_request_timeout,
                body=body,
            )
        elif method == "DELETE":
            return await self.rest_client.DELETE(
                url,
                query_params=query_params,
                headers=headers,
                _preload_content=_preload_content,
                _request_timeout=_request_timeout,
                body=body,
            )
        else:
            raise PineconeApiValueError(
                "http method must be `GET`, `HEAD`, `OPTIONS`, `POST`, `PATCH`, `PUT` or `DELETE`."
            )

    @staticmethod
    def get_file_data_and_close_file(file_instance: io.IOBase) -> bytes:
        file_data = file_instance.read()
        file_instance.close()
        return file_data
