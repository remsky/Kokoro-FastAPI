import io
from abc import ABC, abstractmethod
import logging

from .exceptions import (
    PineconeApiException,
    UnauthorizedException,
    ForbiddenException,
    NotFoundException,
    ServiceException,
)

logger = logging.getLogger(__name__)
""" @private """


class RESTResponse(io.IOBase):
    def __init__(self, status, data, headers, reason=None) -> None:
        self.status = status
        self.reason = reason
        self.data = data
        self.headers = headers

    def getheaders(self):
        """Returns a dictionary of the response headers."""
        return self.headers

    def getheader(self, name, default=None):
        """Returns a given response header."""
        return self.headers.get(name, default)


def raise_exceptions_or_return(r: RESTResponse):
    logger.debug("response status: %s", r.status)

    if not 200 <= r.status <= 299:
        if r.status == 401:
            raise UnauthorizedException(http_resp=r)

        if r.status == 403:
            raise ForbiddenException(http_resp=r)

        if r.status == 404:
            raise NotFoundException(http_resp=r)

        if 500 <= r.status <= 599:
            raise ServiceException(http_resp=r)

        raise PineconeApiException(http_resp=r)

    return r


class RestClientInterface(ABC):
    def __init__(self, configuration, **kwargs) -> None:
        pass

    @abstractmethod
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
        pass

    def GET(
        self, url, headers=None, query_params=None, _preload_content=True, _request_timeout=None
    ):
        return self.request(
            "GET",
            url,
            headers=headers,
            _preload_content=_preload_content,
            _request_timeout=_request_timeout,
            query_params=query_params,
        )

    def HEAD(
        self, url, headers=None, query_params=None, _preload_content=True, _request_timeout=None
    ):
        return self.request(
            "HEAD",
            url,
            headers=headers,
            _preload_content=_preload_content,
            _request_timeout=_request_timeout,
            query_params=query_params,
        )

    def OPTIONS(
        self,
        url,
        headers=None,
        query_params=None,
        post_params=None,
        body=None,
        _preload_content=True,
        _request_timeout=None,
    ):
        return self.request(
            "OPTIONS",
            url,
            headers=headers,
            query_params=query_params,
            post_params=post_params,
            _preload_content=_preload_content,
            _request_timeout=_request_timeout,
            body=body,
        )

    def DELETE(
        self,
        url,
        headers=None,
        query_params=None,
        body=None,
        _preload_content=True,
        _request_timeout=None,
    ):
        return self.request(
            "DELETE",
            url,
            headers=headers,
            query_params=query_params,
            _preload_content=_preload_content,
            _request_timeout=_request_timeout,
            body=body,
        )

    def POST(
        self,
        url,
        headers=None,
        query_params=None,
        post_params=None,
        body=None,
        _preload_content=True,
        _request_timeout=None,
    ):
        return self.request(
            "POST",
            url,
            headers=headers,
            query_params=query_params,
            post_params=post_params,
            _preload_content=_preload_content,
            _request_timeout=_request_timeout,
            body=body,
        )

    def PUT(
        self,
        url,
        headers=None,
        query_params=None,
        post_params=None,
        body=None,
        _preload_content=True,
        _request_timeout=None,
    ):
        return self.request(
            "PUT",
            url,
            headers=headers,
            query_params=query_params,
            post_params=post_params,
            _preload_content=_preload_content,
            _request_timeout=_request_timeout,
            body=body,
        )

    def PATCH(
        self,
        url,
        headers=None,
        query_params=None,
        post_params=None,
        body=None,
        _preload_content=True,
        _request_timeout=None,
    ):
        return self.request(
            "PATCH",
            url,
            headers=headers,
            query_params=query_params,
            post_params=post_params,
            _preload_content=_preload_content,
            _request_timeout=_request_timeout,
            body=body,
        )
