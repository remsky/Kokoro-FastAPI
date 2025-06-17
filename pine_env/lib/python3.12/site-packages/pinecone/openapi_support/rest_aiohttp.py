import ssl
import certifi
import json
from .rest_utils import RestClientInterface, RESTResponse, raise_exceptions_or_return
from ..config.openapi_configuration import Configuration


class AiohttpRestClient(RestClientInterface):
    def __init__(self, configuration: Configuration) -> None:
        try:
            import aiohttp
            from aiohttp_retry import RetryClient
            from .retry_aiohttp import JitterRetry
        except ImportError:
            raise ImportError(
                "Additional dependencies are required to use Pinecone with asyncio. Include these extra dependencies in your project by installing `pinecone[asyncio]`."
            ) from None

        if configuration.ssl_ca_cert is not None:
            ca_certs = configuration.ssl_ca_cert
        else:
            ca_certs = certifi.where()

        ssl_context = ssl.create_default_context(cafile=ca_certs)

        conn = aiohttp.TCPConnector(verify_ssl=configuration.verify_ssl, ssl=ssl_context)

        if configuration.proxy:
            self._session = aiohttp.ClientSession(connector=conn, proxy=configuration.proxy)
        else:
            self._session = aiohttp.ClientSession(connector=conn)

        if configuration.retries is not None:
            retry_options = configuration.retries
        else:
            retry_options = JitterRetry(
                attempts=5,
                start_timeout=0.1,
                max_timeout=3.0,
                statuses={500, 502, 503, 504},
                methods=None,  # retry on all methods
                exceptions={aiohttp.ClientError, aiohttp.ServerDisconnectedError},
            )
        self._retry_client = RetryClient(client_session=self._session, retry_options=retry_options)

    async def close(self):
        await self._retry_client.close()

    async def request(
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
        if method in ["POST", "PUT", "PATCH", "OPTIONS"] and ("Content-Type" not in headers):
            headers["Content-Type"] = "application/json"

        if "application/x-ndjson" in headers.get("Content-Type", "").lower():
            ndjson_data = "\n".join(json.dumps(record) for record in body)

            async with self._retry_client.request(
                method, url, params=query_params, headers=headers, data=ndjson_data
            ) as resp:
                content = await resp.read()
                return raise_exceptions_or_return(
                    RESTResponse(resp.status, content, resp.headers, resp.reason)
                )

        else:
            async with self._retry_client.request(
                method, url, params=query_params, headers=headers, json=body
            ) as resp:
                content = await resp.read()
                return raise_exceptions_or_return(
                    RESTResponse(resp.status, content, resp.headers, resp.reason)
                )

    async def GET(
        self, url, headers=None, query_params=None, _preload_content=True, _request_timeout=None
    ):
        return await self.request(
            "GET",
            url,
            headers=headers,
            _preload_content=_preload_content,
            _request_timeout=_request_timeout,
            query_params=query_params,
        )

    async def HEAD(
        self, url, headers=None, query_params=None, _preload_content=True, _request_timeout=None
    ):
        return await self.request(
            "HEAD",
            url,
            headers=headers,
            _preload_content=_preload_content,
            _request_timeout=_request_timeout,
            query_params=query_params,
        )

    async def OPTIONS(
        self,
        url,
        headers=None,
        query_params=None,
        post_params=None,
        body=None,
        _preload_content=True,
        _request_timeout=None,
    ):
        return await self.request(
            "OPTIONS",
            url,
            headers=headers,
            query_params=query_params,
            post_params=post_params,
            _preload_content=_preload_content,
            _request_timeout=_request_timeout,
            body=body,
        )

    async def DELETE(
        self,
        url,
        headers=None,
        query_params=None,
        body=None,
        _preload_content=True,
        _request_timeout=None,
    ):
        return await self.request(
            "DELETE",
            url,
            headers=headers,
            query_params=query_params,
            _preload_content=_preload_content,
            _request_timeout=_request_timeout,
            body=body,
        )

    async def POST(
        self,
        url,
        headers=None,
        query_params=None,
        post_params=None,
        body=None,
        _preload_content=True,
        _request_timeout=None,
    ):
        return await self.request(
            "POST",
            url,
            headers=headers,
            query_params=query_params,
            post_params=post_params,
            _preload_content=_preload_content,
            _request_timeout=_request_timeout,
            body=body,
        )

    async def PUT(
        self,
        url,
        headers=None,
        query_params=None,
        post_params=None,
        body=None,
        _preload_content=True,
        _request_timeout=None,
    ):
        return await self.request(
            "PUT",
            url,
            headers=headers,
            query_params=query_params,
            post_params=post_params,
            _preload_content=_preload_content,
            _request_timeout=_request_timeout,
            body=body,
        )

    async def PATCH(
        self,
        url,
        headers=None,
        query_params=None,
        post_params=None,
        body=None,
        _preload_content=True,
        _request_timeout=None,
    ):
        return await self.request(
            "PATCH",
            url,
            headers=headers,
            query_params=query_params,
            post_params=post_params,
            _preload_content=_preload_content,
            _request_timeout=_request_timeout,
            body=body,
        )
