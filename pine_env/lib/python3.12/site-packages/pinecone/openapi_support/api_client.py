import atexit
import io

from typing import Optional, List, Tuple, Dict, Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.pool import ThreadPool
    from concurrent.futures import ThreadPoolExecutor

from .rest_urllib3 import Urllib3RestClient
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
from .auth_util import AuthUtil
from .serializer import Serializer


class ApiClient(object):
    """Generic API client for OpenAPI client library builds.

    :param configuration: .Configuration object for this client
    :param pool_threads: The number of threads to use for async requests
        to the API. More threads means more concurrent API requests.
    """

    _pool: Optional["ThreadPool"] = None
    _threadpool_executor: Optional["ThreadPoolExecutor"] = None

    def __init__(
        self, configuration: Optional[Configuration] = None, pool_threads: Optional[int] = 1
    ) -> None:
        if configuration is None:
            configuration = Configuration.get_default_copy()
        self.configuration = configuration
        self.pool_threads = pool_threads

        self.rest_client = Urllib3RestClient(configuration)

        self.default_headers: Dict[str, str] = {}
        self.user_agent = "OpenAPI-Generator/1.0.0/python"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if self._threadpool_executor:
            self._threadpool_executor.shutdown()
            self._threadpool_executor = None
        if self._pool:
            self._pool.close()
            self._pool.join()
            self._pool = None
            if hasattr(atexit, "unregister"):
                atexit.unregister(self.close)

    @property
    def pool(self) -> "ThreadPool":
        """Create thread pool on first request
        avoids instantiating unused threadpool for blocking clients.
        """
        if self._pool is None:
            from multiprocessing.pool import ThreadPool

            atexit.register(self.close)
            self._pool = ThreadPool(self.pool_threads)
        return self._pool

    @property
    def threadpool_executor(self) -> "ThreadPoolExecutor":
        if self._threadpool_executor is None:
            from concurrent.futures import ThreadPoolExecutor

            self._threadpool_executor = ThreadPoolExecutor(max_workers=self.pool_threads)
        return self._threadpool_executor

    @property
    def user_agent(self):
        """User agent for this API client"""
        return self.default_headers["User-Agent"]

    @user_agent.setter
    def user_agent(self, value):
        self.default_headers["User-Agent"] = value

    def set_default_header(self, header_name, header_value):
        self.default_headers[header_name] = header_value

    def __call_api(
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
        _return_http_data_only: Optional[bool] = True,
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

        headers_tuple, path_params_tuple, sanitized_path_params = process_params(
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
            if headers_tuple["Content-Type"].startswith("multipart"):
                processed_post_params = parameters_to_multipart(sanitized_post_params, (dict))
        else:
            processed_post_params = None

        # body
        if body:
            body = Serializer.sanitize_for_serialization(body)

        # auth setting
        AuthUtil.update_params_for_auth(
            configuration=config,
            endpoint_auth_settings=auth_settings,
            headers=headers_tuple,
            querys=processed_query_params,
        )

        url = build_request_url(
            config=config,
            processed_path_params=path_params_tuple,
            resource_path=resource_path,
            _host=_host,
        )

        try:
            # perform request and return response
            response_data = self.request(
                method,
                url,
                query_params=processed_query_params,
                headers=headers_tuple,
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
            from .deserializer import Deserializer

            Deserializer.decode_response(response_type=response_type, response=response_data)
            return_data = Deserializer.deserialize(
                response=response_data,
                response_type=response_type,
                config=self.configuration,
                _check_type=_check_type,
            )
        else:
            return_data = None

        if _return_http_data_only:
            return return_data
        else:
            return (return_data, response_data.status, response_data.getheaders())

    def call_api(
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
        async_req: Optional[bool] = None,
        async_threadpool_executor: Optional[bool] = None,
        _return_http_data_only: Optional[bool] = None,
        collection_formats: Optional[Dict[str, str]] = None,
        _preload_content: bool = True,
        _request_timeout: Optional[Union[int, float, Tuple]] = None,
        _host: Optional[str] = None,
        _check_type: Optional[bool] = None,
    ):
        """Makes the HTTP request (synchronous) and returns deserialized data.

        To make an async_req request, set the async_req parameter.

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
        :param async_req bool: execute request asynchronously
        :type async_req: bool, optional
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
        :return:
            If async_req parameter is True,
            the request will be called asynchronously.
            The method will return the request thread.
            If parameter async_req is False or missing,
            then the method will return the response directly.
        """
        if async_threadpool_executor:
            return self.threadpool_executor.submit(
                self.__call_api,
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

        if not async_req:
            return self.__call_api(
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

        return self.pool.apply_async(
            self.__call_api,
            (
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
            ),
        )

    def request(
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
            return self.rest_client.GET(
                url,
                query_params=query_params,
                _preload_content=_preload_content,
                _request_timeout=_request_timeout,
                headers=headers,
            )
        elif method == "HEAD":
            return self.rest_client.HEAD(
                url,
                query_params=query_params,
                _preload_content=_preload_content,
                _request_timeout=_request_timeout,
                headers=headers,
            )
        elif method == "OPTIONS":
            return self.rest_client.OPTIONS(
                url,
                query_params=query_params,
                headers=headers,
                post_params=post_params,
                _preload_content=_preload_content,
                _request_timeout=_request_timeout,
                body=body,
            )
        elif method == "POST":
            return self.rest_client.POST(
                url,
                query_params=query_params,
                headers=headers,
                post_params=post_params,
                _preload_content=_preload_content,
                _request_timeout=_request_timeout,
                body=body,
            )
        elif method == "PUT":
            return self.rest_client.PUT(
                url,
                query_params=query_params,
                headers=headers,
                post_params=post_params,
                _preload_content=_preload_content,
                _request_timeout=_request_timeout,
                body=body,
            )
        elif method == "PATCH":
            return self.rest_client.PATCH(
                url,
                query_params=query_params,
                headers=headers,
                post_params=post_params,
                _preload_content=_preload_content,
                _request_timeout=_request_timeout,
                body=body,
            )
        elif method == "DELETE":
            return self.rest_client.DELETE(
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
