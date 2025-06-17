import json
import logging
import ssl
import os
from typing import Optional
from urllib.parse import urlencode, quote
from ..config.openapi_configuration import Configuration
from .rest_utils import raise_exceptions_or_return, RESTResponse, RestClientInterface

import urllib3
from .retry_urllib3 import JitterRetry
from .exceptions import PineconeApiException, PineconeApiValueError


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


logger = logging.getLogger(__name__)
""" @private """


class Urllib3RestClient(RestClientInterface):
    pool_manager: urllib3.PoolManager

    def __init__(
        self, configuration: Configuration, pools_size: int = 4, maxsize: Optional[int] = None
    ) -> None:
        # urllib3.PoolManager will pass all kw parameters to connectionpool
        # https://github.com/shazow/urllib3/blob/f9409436f83aeb79fbaf090181cd81b784f1b8ce/urllib3/poolmanager.py#L75  # noqa: E501
        # https://github.com/shazow/urllib3/blob/f9409436f83aeb79fbaf090181cd81b784f1b8ce/urllib3/connectionpool.py#L680  # noqa: E501
        # maxsize is the number of requests to host that are allowed in parallel  # noqa: E501
        # Custom SSL certificates and client certificates: http://urllib3.readthedocs.io/en/latest/advanced-usage.html  # noqa: E501

        # cert_reqs
        if configuration.verify_ssl:
            cert_reqs = ssl.CERT_REQUIRED
        else:
            cert_reqs = ssl.CERT_NONE

        addition_pool_args = {}
        if configuration.assert_hostname is not None:
            addition_pool_args["assert_hostname"] = configuration.assert_hostname  # noqa: E501

        if configuration.retries is not None:
            addition_pool_args["retries"] = configuration.retries
        else:
            addition_pool_args["retries"] = JitterRetry(
                total=5,
                backoff_factor=0.25,
                status_forcelist=(500, 502, 503, 504),
                allowed_methods=None,
            )

        if configuration.socket_options is not None:
            addition_pool_args["socket_options"] = configuration.socket_options

        if maxsize is None:
            if configuration.connection_pool_maxsize is not None:
                maxsize = configuration.connection_pool_maxsize
            else:
                maxsize = 4

        # https pool manager
        if configuration.proxy:
            self.pool_manager = urllib3.ProxyManager(
                num_pools=pools_size,
                maxsize=maxsize,
                cert_reqs=cert_reqs,
                ca_certs=configuration.ssl_ca_cert,
                cert_file=configuration.cert_file,
                key_file=configuration.key_file,
                proxy_url=configuration.proxy,
                proxy_headers=configuration.proxy_headers,
                **addition_pool_args,
            )
        else:
            self.pool_manager = urllib3.PoolManager(
                num_pools=pools_size,
                maxsize=maxsize,
                cert_reqs=cert_reqs,
                ca_certs=configuration.ssl_ca_cert,
                cert_file=configuration.cert_file,
                key_file=configuration.key_file,
                **addition_pool_args,
            )

    def request(
        self,
        method,
        url,
        query_params=None,
        headers=None,
        body=None,
        post_params=None,
        _preload_content=True,
        _request_timeout=None,
    ):
        """Perform requests.

        :param method: http request method
        :param url: http request url
        :param query_params: query parameters in the url
        :param headers: http request headers
        :param body: request json body, for `application/json`
        :param post_params: request post parameters,
                            `application/x-www-form-urlencoded`
                            and `multipart/form-data`
        :param _preload_content: if False, the urllib3.HTTPResponse object will
                                 be returned without reading/decoding response
                                 data. Default is True.
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        """
        logger.debug("Calling urllib3 request()")
        method = method.upper()
        assert method in ["GET", "HEAD", "DELETE", "POST", "PUT", "PATCH", "OPTIONS"]

        if os.environ.get("PINECONE_DEBUG_CURL"):
            formatted_headers = " ".join(["-H '{0}: {1}'".format(k, v) for k, v in headers.items()])
            formatted_query = urlencode(query_params)
            if formatted_query:
                formatted_url = f"{url}?{formatted_query}"
            else:
                formatted_url = url
            if body is None:
                print(
                    bcolors.OKBLUE
                    + "curl -X {method} '{url}' {formatted_headers}".format(
                        method=method, url=formatted_url, formatted_headers=formatted_headers
                    )
                    + bcolors.ENDC
                )
            else:
                formatted_body = json.dumps(body)
                print(
                    bcolors.OKBLUE
                    + "curl -X {method} '{url}' {formatted_headers} -d '{data}'".format(
                        method=method,
                        url=formatted_url,
                        formatted_headers=formatted_headers,
                        data=formatted_body,
                    )
                    + bcolors.ENDC
                )

        if post_params and body:
            raise PineconeApiValueError("body parameter cannot be used with post_params parameter.")

        post_params = post_params or {}
        headers = headers or {}

        timeout = None
        if _request_timeout:
            if isinstance(_request_timeout, (int, float)):  # noqa: E501,F821
                timeout = urllib3.Timeout(total=_request_timeout)
            elif isinstance(_request_timeout, tuple) and len(_request_timeout) == 2:
                timeout = urllib3.Timeout(connect=_request_timeout[0], read=_request_timeout[1])

        try:
            # For `POST`, `PUT`, `PATCH`, `OPTIONS`, `DELETE`
            if method in ["POST", "PUT", "PATCH", "OPTIONS", "DELETE"]:
                # Only set a default Content-Type for POST, PUT, PATCH and OPTIONS requests
                if (method != "DELETE") and ("Content-Type" not in headers):
                    headers["Content-Type"] = "application/json"
                if query_params:
                    url += "?" + urlencode(query_params, quote_via=quote)

                content_type = headers.get("Content-Type", "").lower()
                if content_type == "" or ("json" in content_type):
                    if body is None:
                        request_body = None
                    else:
                        if content_type == "application/x-ndjson":
                            # for x-ndjson requests, we are expecting an array of elements
                            # that need to be converted to a newline separated string
                            request_body = "\n".join(json.dumps(element) for element in body)
                        else:  # content_type == "application/json":
                            request_body = json.dumps(body)
                    r = self.pool_manager.request(
                        method,
                        url,
                        body=request_body,
                        preload_content=_preload_content,
                        timeout=timeout,
                        headers=headers,
                    )

                elif content_type == "application/x-www-form-urlencoded":  # noqa: E501
                    r = self.pool_manager.request(
                        method,
                        url,
                        fields=post_params,
                        encode_multipart=False,
                        preload_content=_preload_content,
                        timeout=timeout,
                        headers=headers,
                    )
                elif content_type == "multipart/form-data":
                    # must del headers['Content-Type'], or the correct
                    # Content-Type which generated by urllib3 will be
                    # overwritten.
                    del headers["Content-Type"]
                    r = self.pool_manager.request(
                        method,
                        url,
                        fields=post_params,
                        encode_multipart=True,
                        preload_content=_preload_content,
                        timeout=timeout,
                        headers=headers,
                    )
                # Pass a `string` parameter directly in the body to support
                # other content types than Json when `body` argument is
                # provided in serialized form
                elif isinstance(body, str) or isinstance(body, bytes):
                    request_body = body
                    r = self.pool_manager.request(
                        method,
                        url,
                        body=request_body,
                        preload_content=_preload_content,
                        timeout=timeout,
                        headers=headers,
                    )
                else:
                    # Cannot generate the request from given parameters
                    msg = """Cannot prepare a request message for provided
                             arguments. Please check that your arguments match
                             declared content type."""
                    raise PineconeApiException(status=0, reason=msg)
            # For `GET`, `HEAD`
            else:
                if query_params:
                    url += "?" + urlencode(query_params, quote_via=quote)
                r = self.pool_manager.request(
                    method, url, preload_content=_preload_content, timeout=timeout, headers=headers
                )
        except urllib3.exceptions.SSLError as e:
            msg = "{0}\n{1}".format(type(e).__name__, str(e))
            raise PineconeApiException(status=0, reason=msg)

        if os.environ.get("PINECONE_DEBUG_CURL"):
            o = RESTResponse(r.status, r.data, r.headers, r.reason)

            if o.status <= 300:
                print(bcolors.OKGREEN + o.data.decode("utf-8") + bcolors.ENDC)
            else:
                print(bcolors.FAIL + o.data.decode("utf-8") + bcolors.ENDC)

        if _preload_content:
            r = RESTResponse(r.status, r.data, r.headers, r.reason)

            # log response body
            logger.debug("response body: %s", r.data)

        return raise_exceptions_or_return(r)
