"""On-demand PNG delivery for an authorized browser and its current CV result."""

import io
import logging
import re
from dataclasses import dataclass
from secrets import token_urlsafe
from threading import Lock
from time import monotonic
from typing import Annotated, Literal
from uuid import UUID
from weakref import ReferenceType, ref

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse, Response

from app.core.errors import AppError, ErrorCode
from app.core.logging_config import log_error
from app.core.session import SessionState
from app.core.temporary_files import TemporaryFiles
from app.cv.schemas import CVResult
from app.cv.visualization import compose_overlay, draw_localization, encode_png, open_aligned_images

logger = logging.getLogger(__name__)
EXPORT_COOKIE = "lesion_browser"
EXPORT_PATH = "/lesion-export"
UNAVAILABLE = "Chưa có kết quả hiện tại để xuất PNG."
NO_CACHE = {"Cache-Control": "no-store"}


def browser_owner(request) -> str | None:
    token = getattr(request, "cookies", {}).get(EXPORT_COOKIE, "")
    return token if re.fullmatch(r"[A-Za-z0-9_-]{43}", token) else None


class ExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    session: UUID
    run: UUID
    action: Annotated[int, Field(gt=0)]
    proof: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    kind: Literal["overlay", "localization"]
    opacity: Annotated[int, Field(ge=0, le=100)]


@dataclass(frozen=True, slots=True)
class ExportGrant:
    state: ReferenceType[SessionState]
    result: CVResult
    expires: float


class PNGExports:
    def __init__(self, files: TemporaryFiles):
        self.files = files
        self._grants: dict[UUID, ExportGrant] = {}
        self._encoding: set[UUID] = set()
        self._lock = Lock()

    def issue(self, current: SessionState, result: CVResult) -> None:
        session_id = current.session_id
        with self._lock:
            self._grants[session_id] = ExportGrant(
                ref(current, lambda _: self.revoke(session_id)), result, monotonic() + self.files.ttl_seconds,
            )

    def revoke(self, session_id: UUID) -> None:
        with self._lock:
            self._grants.pop(session_id, None)

    def response(self, owner: str, requested: ExportRequest) -> Response:
        with self._lock:
            grant = self._grants.get(requested.session)
        current = grant.state() if grant else None
        if current is None or current.browser_owner != owner or grant.result.run_id != requested.run:
            return self.unavailable()
        result = grant.result
        with current.lock:
            if not self._matches(current, grant, requested) or (requested.kind == "localization" and result.localization is None):
                return self.unavailable()
            source_file = current.image.temporary_file
            with self._lock:
                if requested.session in self._encoding:
                    return PlainTextResponse("Đang chuẩn bị PNG cho phiên này.", 409, headers=NO_CACHE)
                self._encoding.add(requested.session)
        try:
            with open_aligned_images(self.files, current.session_id, source_file, result) as (source, mask):
                if requested.kind == "overlay":
                    rendered = compose_overlay(source, mask, requested.opacity)
                else:
                    rendered = draw_localization(source, result.localization.box)
                with rendered, io.BytesIO() as buffer:
                    encode_png(rendered, buffer)
                    content = buffer.getvalue()
            with current.lock:
                if not self._matches(current, grant, requested):
                    return self.unavailable()
        except (AppError, OSError, ValueError, MemoryError) as exc:
            with current.lock:
                if not self._matches(current, grant, requested):
                    return self.unavailable()
            log_error(logger, exc, current.session_id)
            message = str(exc) if isinstance(exc, AppError) else str(AppError(ErrorCode.TEMPORARY_FILE))
            return PlainTextResponse(message, 410, headers=NO_CACHE)
        finally:
            with self._lock:
                self._encoding.remove(requested.session)

        filename = f"segformer-{requested.kind}-{result.run_id.hex[:12]}.png"
        return Response(content, media_type="image/png", headers={
            **NO_CACHE,
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Content-Type-Options": "nosniff",
        })

    @staticmethod
    def _matches(current: SessionState, grant: ExportGrant, requested: ExportRequest) -> bool:
        return (
            monotonic() < grant.expires and current.image is not None and current.result is grant.result
            and current.matches_image_action(requested.action) and current.browser_proof == requested.proof
        )

    @staticmethod
    def unavailable() -> Response:
        return PlainTextResponse(UNAVAILABLE, 404, headers=NO_CACHE)


class PNGExportTransport(BaseHTTPMiddleware):
    def __init__(self, app, exports: PNGExports):
        super().__init__(app)
        self.exports = exports

    async def dispatch(self, request, call_next):
        path = request.url.path
        if path == EXPORT_PATH:
            if request.method != "POST":
                return Response(status_code=405, headers={**NO_CACHE, "Allow": "POST"})
            owner = browser_owner(request)
            origin = request.headers.get("origin")
            if owner is None or (origin is not None and origin != f"{request.url.scheme}://{request.url.netloc}"):
                return self.exports.unavailable()
            payload = bytearray()
            async for chunk in request.stream():
                if len(payload) + len(chunk) > 1024:
                    return self.exports.unavailable()
                payload.extend(chunk)
            try:
                requested = ExportRequest.model_validate_json(payload)
            except ValidationError:
                return self.exports.unavailable()
            return await run_in_threadpool(self.exports.response, owner, requested)
        response = await call_next(request)
        if path == "/" and request.method == "GET" and response.status_code == 200 and browser_owner(request) is None:
            response.set_cookie(
                EXPORT_COOKIE, token_urlsafe(32), httponly=True, samesite="strict", secure=request.url.scheme == "https",
            )
        return response
