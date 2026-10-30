"""Request body size limit, enforced as the body arrives."""

from fastapi.responses import JSONResponse

from .config import settings


class _BodyTooLarge(Exception):
    pass


def _too_large(limit: int) -> JSONResponse:
    return JSONResponse(
        {
            "detail": {
                "error": "too_large",
                "message": f"request body exceeds {limit >> 20} MB",
            }
        },
        status_code=413,
        headers={"connection": "close"},
    )


class BodySizeLimitMiddleware:
    """413 a request whose body passes MAX_REQUEST_BODY_MB, counted as it arrives."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = settings.max_request_body_mb * 1024 * 1024
        declared = dict(scope["headers"]).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > limit:
            return await _too_large(limit)(scope, receive, send)

        received = 0
        over = False
        started = False

        async def counted_receive():
            nonlocal received, over
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit and not started:
                    over = True
                    raise _BodyTooLarge()
            return message

        # FastAPI turns a failed body read into its own 400, so swap in the 413
        async def guarded_send(message):
            nonlocal started
            if over:
                if message["type"] == "http.response.start" and not started:
                    started = True
                    await _too_large(limit)(scope, receive, send)
                return
            started = started or message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, counted_receive, guarded_send)
        except _BodyTooLarge:
            if not started:
                await _too_large(limit)(scope, receive, send)
