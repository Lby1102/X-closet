"""HTTP 契约层（框架无关）：02 的路由、包装、鉴权、幂等与错误码。

为什么不是 FastAPI：当前环境装不上 FastAPI（03 的团队栈要求，由郭子骏落地）。
本文件把所有 HTTP 语义做成纯函数式入口 ``Api.dispatch(method, path, ...)``，
FastAPI 路由或 stdlib HTTP 适配器（``http_server.py``）都只是它的薄壳，
联调时换成 ``APIRouter`` 的依赖注入即可，业务服务不用改。

约定（02）
    * 所有路径以 /api/v1 开头，ID 均为 string；
    * 成功 ``{"data": ..., "request_id": ...}``，失败 ``{"error": {...}, "request_id": ...}``；
    * 普通成功 200，创建资源 201，耗时任务 202 + Job；
    * 缺失/未知 X-User-Id -> 401；资源在当前用户下不存在 -> 404；
      幂等键冲突 -> 409；字段或业务条件无效 -> 422；外部服务失败 -> 502。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .common import ModuleError, request_id
from .trends import SEASONS

__all__ = [
    "API_PREFIX",
    "PATH_ANALYZE",
    "PATH_COLLECT",
    "Api",
    "ApiNotFound",
    "InvalidInput",
    "Response",
    "Unauthorized",
    "Users",
    "error_response",
]

API_PREFIX = "/api/v1"
PATH_ANALYZE = f"{API_PREFIX}/garments:analyze"
PATH_COLLECT = f"{API_PREFIX}/inspirations:collect"


# --------------------------------------------------------------------------- #
# 错误与响应
# --------------------------------------------------------------------------- #
class Unauthorized(ModuleError):
    """02：缺失或未知 X-User-Id 返回 401。"""

    code = "UNAUTHORIZED"
    status_code = 401


class ApiNotFound(ModuleError):
    code = "NOT_FOUND"
    status_code = 404


class InvalidInput(ModuleError):
    code = "INVALID_INPUT"
    status_code = 422


@dataclass(frozen=True)
class Response:
    status_code: int
    body: dict[str, Any]


def _ok(data: Any, status_code: int = 200) -> Response:
    return Response(status_code, {"data": data, "request_id": request_id()})


def _error(exc: ModuleError) -> Response:
    return Response(
        exc.status_code,
        {"error": exc.to_error(), "request_id": request_id()},
    )


def error_response(exc: ModuleError) -> Response:
    """供 HTTP 适配器复用的错误包装（无效 JSON、缺少上传文件等）。"""
    return _error(exc)


# --------------------------------------------------------------------------- #
# 用户（课程演示用 X-User-Id 指定预置用户）
# --------------------------------------------------------------------------- #
class Users:
    """预置用户目录。真实接入点：郭子骏的鉴权中间件。

    02：课程演示用 X-User-Id 指定预置用户，缺失或未知返回 401。
    传入空集合表示不做白名单（仅本地开发/快速试跑用）。
    """

    def __init__(self, user_ids: Iterable[str] = ()) -> None:
        self._ids: set[str] = {str(u).strip() for u in user_ids if str(u).strip()}

    def add(self, user_id: str) -> str:
        value = str(user_id).strip()
        if value:
            self._ids.add(value)
        return value

    @property
    def known(self) -> tuple[str, ...]:
        return tuple(sorted(self._ids))

    def ensure(self, user_id: str | None) -> str:
        value = (user_id or "").strip() if isinstance(user_id, str) else ""
        if not value:
            raise Unauthorized("缺少 X-User-Id")
        if self._ids and value not in self._ids:
            raise Unauthorized(f"未知用户：{value}")
        return value


# --------------------------------------------------------------------------- #
# 入参校验
# --------------------------------------------------------------------------- #
def _mapping(body: Any) -> dict[str, Any]:
    if body is None:
        return {}
    if isinstance(body, Mapping):
        return dict(body)
    raise InvalidInput("请求体必须是 JSON 对象")


def _check_fields(
    payload: Mapping[str, Any],
    *,
    required: Iterable[str] = (),
    optional: Iterable[str] = (),
) -> None:
    allowed = set(required) | set(optional)
    unknown = set(payload) - allowed
    if unknown:
        raise InvalidInput(f"未知字段：{sorted(unknown)}")
    missing = [field for field in required if field not in payload]
    if missing:
        raise InvalidInput(f"缺少必填字段：{sorted(missing)}")


def _text(value: Any, field: str, *, required: bool = True) -> str | None:
    if value is None:
        if required:
            raise InvalidInput(f"{field} 不能为空")
        return None
    if not isinstance(value, str):
        raise InvalidInput(f"{field} 必须是字符串")
    text = value.strip()
    if not text:
        if required:
            raise InvalidInput(f"{field} 不能为空")
        return None
    return text


def _dump(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    return dump(mode="json") if dump is not None else value


def _require_idempotency_key(idempotency_key: str | None) -> str:
    """02：提交任务时携带 Idempotency-Key；缺失属于请求问题（422），先于资源查询。"""
    key = (idempotency_key or "").strip() if isinstance(idempotency_key, str) else ""
    if not key:
        raise InvalidInput("缺少 Idempotency-Key")
    return key


# --------------------------------------------------------------------------- #
# 契约层
# --------------------------------------------------------------------------- #
class Api:
    """02 的全部 HTTP 入口。服务通过构造参数注入（见 container.build_services）。"""

    def __init__(
        self,
        *,
        users: Users,
        images: Any,
        recognition: Any,
        wardrobe: Any,
        inspirations: Any,
        trends: Any,
        jobs: Any,
    ) -> None:
        self._users = users
        self._images = images
        self._recognition = recognition
        self._wardrobe = wardrobe
        self._inspirations = inspirations
        self._trends = trends
        self._jobs = jobs

    # ---- 各端点 ----------------------------------------------------------- #
    async def upload_image(
        self, user_id: str | None, data: bytes, *, content_type: str | None = None
    ) -> Response:
        return await self._guard(self._upload_image(user_id, data, content_type))

    async def analyze_garment(
        self,
        user_id: str | None,
        idempotency_key: str | None,
        body: Any,
    ) -> Response:
        return await self._guard(
            self._analyze_garment(user_id, idempotency_key, body)
        )

    async def create_garment(self, user_id: str | None, body: Any) -> Response:
        return await self._guard(self._create_garment(user_id, body))

    async def list_garments(
        self, user_id: str | None, query: Mapping[str, Any] | None = None
    ) -> Response:
        return await self._guard(self._list_garments(user_id, query or {}))

    async def update_garment(
        self, user_id: str | None, garment_id: str, body: Any
    ) -> Response:
        return await self._guard(self._update_garment(user_id, garment_id, body))

    async def get_job(self, user_id: str | None, job_id: str) -> Response:
        return await self._guard(self._get_job(user_id, job_id))

    async def create_inspiration(self, user_id: str | None, body: Any) -> Response:
        return await self._guard(self._create_inspiration(user_id, body))

    async def list_inspirations(self, user_id: str | None) -> Response:
        return await self._guard(self._list_inspirations(user_id))

    async def collect_inspirations(
        self,
        user_id: str | None,
        idempotency_key: str | None,
        body: Any,
    ) -> Response:
        return await self._guard(
            self._collect_inspirations(user_id, idempotency_key, body)
        )

    # ---- 实现 ------------------------------------------------------------- #
    async def _upload_image(
        self, user_id: str | None, data: bytes, content_type: str | None
    ) -> Response:
        user = self._users.ensure(user_id)
        image = await self._images.add(user, data, content_type=content_type)
        return _ok(
            {"image_id": image.image_id, "image_url": image.image_url}, 201
        )

    async def _analyze_garment(
        self, user_id: str | None, idempotency_key: str | None, body: Any
    ) -> Response:
        user = self._users.ensure(user_id)
        key = _require_idempotency_key(idempotency_key)
        payload = _mapping(body)
        _check_fields(payload, required={"image_id"})
        image_id = _text(payload["image_id"], "image_id")
        # 02：图片不存在时 404 在提交阶段就返回
        await self._images.get(user, image_id)
        job = await self._jobs.submit(
            user_id=user,
            path=PATH_ANALYZE,
            job_type="garment_analysis",
            params={"image_id": image_id},
            idempotency_key=key,
        )
        return _ok(_dump(job), 202)

    async def _create_garment(self, user_id: str | None, body: Any) -> Response:
        user = self._users.ensure(user_id)
        payload = _mapping(body)
        _check_fields(payload, required={"image_id", "tags"})
        garment = await self._wardrobe.create(
            user, _text(payload["image_id"], "image_id"), payload["tags"]
        )
        return _ok(_dump(garment), 201)

    async def _list_garments(
        self, user_id: str | None, query: Mapping[str, Any]
    ) -> Response:
        user = self._users.ensure(user_id)
        category = _text(query.get("category"), "category", required=False)
        items = await self._wardrobe.list_items(user, category)
        return _ok({"items": [_dump(item) for item in items["items"]]}, 200)

    async def _update_garment(
        self, user_id: str | None, garment_id: str, body: Any
    ) -> Response:
        user = self._users.ensure(user_id)
        payload = _mapping(body)
        _check_fields(payload, required={"tags"})
        garment = await self._wardrobe.patch(
            user, _text(garment_id, "garment_id"), payload["tags"]
        )
        return _ok(_dump(garment), 200)

    async def _get_job(self, user_id: str | None, job_id: str) -> Response:
        user = self._users.ensure(user_id)
        job = await self._jobs.get(user, _text(job_id, "job_id"))
        return _ok(_dump(job), 200)

    async def _create_inspiration(
        self, user_id: str | None, body: Any
    ) -> Response:
        user = self._users.ensure(user_id)
        payload = _mapping(body)
        _check_fields(payload, required={"image_id"}, optional={"tags", "source_url"})
        image_id = _text(payload["image_id"], "image_id")
        image = await self._images.get(user, image_id)
        inspiration = await self._inspirations.create(
            user,
            image_id,
            payload.get("tags"),
            payload.get("source_url"),
            image_url=image.image_url,
        )
        return _ok(_dump(inspiration), 201)

    async def _list_inspirations(self, user_id: str | None) -> Response:
        user = self._users.ensure(user_id)
        payload = await self._inspirations.list_items(user)
        return _ok({"items": [_dump(item) for item in payload["items"]]}, 200)

    async def _collect_inspirations(
        self, user_id: str | None, idempotency_key: str | None, body: Any
    ) -> Response:
        user = self._users.ensure(user_id)
        key = _require_idempotency_key(idempotency_key)
        payload = _mapping(body)
        _check_fields(payload, required={"style"}, optional={"season"})
        style = _text(payload["style"], "style")
        season = _text(payload.get("season"), "season", required=False)
        if season is not None and season not in SEASONS:
            raise InvalidInput(
                f"season 必须是 {'/'.join(SEASONS)} 之一", code="INVALID_SEASON"
            )
        job = await self._jobs.submit(
            user_id=user,
            path=PATH_COLLECT,
            job_type="inspiration_collection",
            params={"style": style, "season": season},
            idempotency_key=key,
        )
        return _ok(_dump(job), 202)

    # ---- 框架无关分发（FastAPI / stdlib HTTP 适配器共用） ------------------ #
    async def dispatch(
        self,
        method: str,
        path: str,
        *,
        user_id: str | None = None,
        idempotency_key: str | None = None,
        body: Any = None,
        query: Mapping[str, Any] | None = None,
        upload: tuple[bytes, str | None] | None = None,
    ) -> Response:
        verb = (method or "GET").upper()
        clean = (path or "/").split("?", 1)[0]
        segments = [part for part in clean.strip("/").split("/") if part]
        if len(segments) < 3 or segments[0] != "api" or segments[1] != "v1":
            return _error(ApiNotFound(f"未知路径：{clean}"))
        rest = segments[2:]
        params = dict(query or {})

        if verb == "POST" and rest == ["images"]:
            if upload is None:
                return _error(InvalidInput("缺少上传文件 file"))
            return await self.upload_image(user_id, upload[0], content_type=upload[1])
        if verb == "POST" and rest == ["garments:analyze"]:
            return await self.analyze_garment(user_id, idempotency_key, body)
        if verb == "POST" and rest == ["garments"]:
            return await self.create_garment(user_id, body)
        if verb == "GET" and rest == ["garments"]:
            return await self.list_garments(user_id, params)
        if len(rest) == 2 and rest[0] == "garments" and verb == "PATCH":
            return await self.update_garment(user_id, rest[1], body)
        if len(rest) == 2 and rest[0] == "jobs" and verb == "GET":
            return await self.get_job(user_id, rest[1])
        if verb == "POST" and rest == ["inspirations"]:
            return await self.create_inspiration(user_id, body)
        if verb == "GET" and rest == ["inspirations"]:
            return await self.list_inspirations(user_id)
        if verb == "POST" and rest == ["inspirations:collect"]:
            return await self.collect_inspirations(user_id, idempotency_key, body)
        return _error(ApiNotFound(f"未知路径：{verb} {clean}"))

    # ---- 异常 -> 02 的 error 包装 ----------------------------------------- #
    async def _guard(self, awaitable: Any) -> Response:
        try:
            return await awaitable
        except ModuleError as exc:
            return _error(exc)
