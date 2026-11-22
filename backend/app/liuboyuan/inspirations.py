"""X Closet v1 · 个人喜好图片（inspirations）— 负责人：刘博远

对外契约（见 02 接口协议）
    POST /inspirations   {image_id*:string, tags?:string[], source_url?:string}
                         -> 201 Inspiration
    GET  /inspirations   -> 200 {"items": Inspiration[]}     # 创建时间倒序，返回全部

内部契约（见 03 内部接口与运行流程）
    - 推荐读取“最近三张”个人喜好图 -> recent_image_urls(user_id, limit=3)
    - 采集成功后刷新喜好列表        -> add_collected(...) / list_items(...)

v1 共享数据结构
    Inspiration = {inspiration_id, image_id, image_url, tags:string[], source_url:string|null}

本文件只依赖 pydantic；持久化与图片落地通过 Protocol 注入，默认给内存实现
（可直接跑通验证），真实实现由郭子骏的 SQL/对象存储模块替换，避免跨窗口改共享文件。
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Iterable, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "Inspiration",
    "InspirationError",
    "InspirationRepository",
    "InMemoryInspirationRepository",
    "InspirationsService",
    "contract_status",
    "default_image_ingest",
    "default_media_url",
    "get_inspiration_service",
    "reset_default_service",
    "shared_contract",
    "utcnow",
]


# --------------------------------------------------------------------------- #
# 通用工具
# --------------------------------------------------------------------------- #
def utcnow() -> datetime:
    """当前时间（带时区）。日期/天气按 02 使用 Asia/Shanghai，存储用 UTC 便于比较。"""
    return datetime.now(timezone.utc)


def default_media_url(image_id: str) -> str:
    """image_id -> image_url。真实接入点：郭子骏上传接口返回的 image_url。"""
    return f"/media/{image_id}.jpg"


async def default_image_ingest(image_url: str) -> tuple[str, str]:
    """采集图片落地（默认实现）。

    真实接入点：郭子骏的图片上传/对象存储模块——下载远端图片、写入存储、
    返回 (image_id, 本地 image_url)。默认实现按 URL 推导 image_id 并透传远端 URL，
    保证采集链路在没有上传服务时也能跑通。
    """
    digest = hashlib.sha1(image_url.encode("utf-8")).hexdigest()[:12]
    return f"img_{digest}", image_url


# --------------------------------------------------------------------------- #
# 共享契约桥接（唯一接入点：app/shared）
# --------------------------------------------------------------------------- #
_SHARED_MODULE_CANDIDATES: tuple[str, ...] = (
    "app.shared.models",
    "app.shared.contracts",
    "app.shared.schemas",
)
_SHARED_CACHE: dict[str, Any | None] = {}
_CONTRACT_STATUS: dict[str, str] = {}


def shared_contract(name: str) -> Any | None:
    """取郭子骏 app/shared 中同名 Pydantic 模型；缺失时返回 None（走本地等价模型）。

    骨架落地后只要 app/shared 暴露同名模型，本模块自动切换为共享类型，
    无需改动业务代码。这也是“共享契约”唯一需要联调的地方。
    """
    if name in _SHARED_CACHE:
        return _SHARED_CACHE[name]
    found: Any | None = None
    for path in _SHARED_MODULE_CANDIDATES:
        try:
            module = importlib.import_module(path)
        except Exception:  # pragma: no cover - 骨架未落地时的正常分支
            continue
        candidate = getattr(module, name, None)
        if isinstance(candidate, type):
            found = candidate
            break
    _SHARED_CACHE[name] = found
    _CONTRACT_STATUS[name] = (
        f"{found.__module__}.{found.__name__}" if found is not None else "local_fallback"
    )
    return found


def contract_status() -> dict[str, str]:
    """当前生效的契约来源，供联调自查（local_fallback 表示用的是本文件等价模型）。"""
    shared_contract("Inspiration")
    return dict(_CONTRACT_STATUS)


# --------------------------------------------------------------------------- #
# 错误（02：422 字段或业务条件无效）
# --------------------------------------------------------------------------- #
class InspirationError(Exception):
    """喜好图片业务错误，携带 02 约定的 error 结构。"""

    code = "INVALID_REQUEST"
    status_code = 422
    retryable = False

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code:
            self.code = code

    def to_error(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "retryable": self.retryable}


# --------------------------------------------------------------------------- #
# v1 契约模型（本地等价实现，字段与 02 完全一致）
# --------------------------------------------------------------------------- #
class Inspiration(BaseModel):
    model_config = ConfigDict(extra="ignore")

    inspiration_id: str
    image_id: str
    image_url: str
    tags: list[str] = Field(default_factory=list)
    source_url: str | None = None


def as_contract(inspiration: Inspiration) -> Any:
    """转成当前生效的共享契约实例；转换失败时退回本地模型，保证接口不中断。"""
    model = shared_contract("Inspiration")
    if model is None:
        return inspiration
    try:
        return model(**inspiration.model_dump())
    except Exception:  # pragma: no cover - 共享模型字段漂移时的兜底
        return inspiration


# --------------------------------------------------------------------------- #
# 存储端口
# --------------------------------------------------------------------------- #
class InspirationRepository(Protocol):
    """持久化端口。真实实现：PostgreSQL（郭子骏 shared 存储层）。"""

    async def add(
        self, user_id: str, inspiration: Inspiration, created_at: datetime
    ) -> Inspiration: ...

    async def list_desc(
        self, user_id: str, limit: int | None = None
    ) -> list[Inspiration]: ...

    async def find_by_source_url(
        self, user_id: str, source_url: str
    ) -> Inspiration | None: ...

    async def find_by_image_url(
        self, user_id: str, image_url: str
    ) -> Inspiration | None: ...


@dataclass(frozen=True)
class _Record:
    inspiration: Inspiration
    created_at: datetime
    seq: int


class InMemoryInspirationRepository:
    """进程内实现：倒序按 (created_at, 自增 seq) 排序，稳定可测。"""

    def __init__(self) -> None:
        self._records: dict[str, list[_Record]] = {}
        self._seq = 0
        self._lock = asyncio.Lock()

    async def add(
        self, user_id: str, inspiration: Inspiration, created_at: datetime
    ) -> Inspiration:
        async with self._lock:
            self._seq += 1
            self._records.setdefault(user_id, []).append(
                _Record(inspiration=inspiration, created_at=created_at, seq=self._seq)
            )
        return inspiration

    def _sorted(self, user_id: str) -> list[_Record]:
        records = self._records.get(user_id, [])
        return sorted(records, key=lambda r: (r.created_at, r.seq), reverse=True)

    async def list_desc(
        self, user_id: str, limit: int | None = None
    ) -> list[Inspiration]:
        if limit is not None and limit <= 0:
            return []
        async with self._lock:
            rows = [r.inspiration for r in self._sorted(user_id)]
        return rows[:limit] if limit is not None else rows

    async def find_by_source_url(
        self, user_id: str, source_url: str
    ) -> Inspiration | None:
        target = _normalize_url(source_url)
        async with self._lock:
            for record in self._sorted(user_id):
                current = record.inspiration.source_url
                if current and _normalize_url(current) == target:
                    return record.inspiration
        return None

    async def find_by_image_url(
        self, user_id: str, image_url: str
    ) -> Inspiration | None:
        target = _normalize_url(image_url)
        async with self._lock:
            for record in self._sorted(user_id):
                if _normalize_url(record.inspiration.image_url) == target:
                    return record.inspiration
        return None


# --------------------------------------------------------------------------- #
# 入参归一化
# --------------------------------------------------------------------------- #
_MAX_TAGS = 32
_MAX_TAG_LEN = 40
_HTTP_URL_RE = re.compile(r"^https?://[^\s]+$", re.IGNORECASE)


def _normalize_url(url: str) -> str:
    """去重用的 URL 归一化：去首尾空白、忽略大小写主机、去 fragment、去尾斜杠。"""
    value = (url or "").strip()
    if not value:
        return ""
    value = value.split("#", 1)[0].rstrip("/")
    return value.lower()


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InspirationError(f"{field} 不能为空", code="INVALID_FIELD")
    return value.strip()


def _normalize_tags(tags: Iterable[str] | str | None) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        raw: Iterable[str] = re.split(r"[,，;；]", tags)
    else:
        raw = tags
    result: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise InspirationError("tags 只能包含字符串", code="INVALID_FIELD")
        value = item.strip()
        if not value:
            continue
        if len(value) > _MAX_TAG_LEN:
            value = value[:_MAX_TAG_LEN]
        if value not in result:
            result.append(value)
        if len(result) >= _MAX_TAGS:
            break
    return result


def _normalize_source_url(source_url: str | None) -> str | None:
    if source_url is None:
        return None
    value = str(source_url).strip()
    if not value:
        return None
    if not _HTTP_URL_RE.match(value):
        raise InspirationError(
            "source_url 必须是 http(s) 地址", code="INVALID_SOURCE_URL"
        )
    return value


# --------------------------------------------------------------------------- #
# 业务服务
# --------------------------------------------------------------------------- #
class InspirationsService:
    """喜好图片业务：新增、倒序列表、最近三张、采集写入。"""

    def __init__(
        self,
        repository: InspirationRepository | None = None,
        *,
        now: Callable[[], datetime] = utcnow,
        media_url: Callable[[str], str] = default_media_url,
        image_ingest: Callable[[str], Awaitable[tuple[str, str]]] | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._repository = repository or InMemoryInspirationRepository()
        self._now = now
        self._media_url = media_url
        self._ingest = image_ingest or default_image_ingest
        self._id_factory = id_factory or (lambda: f"insp_{uuid.uuid4().hex[:12]}")

    # ---- 写：POST /inspirations ------------------------------------------- #
    async def create(
        self,
        user_id: str,
        image_id: str,
        tags: Sequence[str] | str | None = None,
        source_url: str | None = None,
        image_url: str | None = None,
    ) -> Inspiration:
        user = _require_text(user_id, "user_id")
        image = _require_text(image_id, "image_id")
        resolved_url = _require_text(
            image_url or self._media_url(image), "image_url"
        )
        inspiration = Inspiration(
            inspiration_id=self._id_factory(),
            image_id=image,
            image_url=resolved_url,
            tags=_normalize_tags(tags),
            source_url=_normalize_source_url(source_url),
        )
        await self._repository.add(user, inspiration, self._now())
        return as_contract(inspiration)

    # ---- 读：GET /inspirations（倒序，全部） ------------------------------ #
    async def list_items(self, user_id: str) -> dict[str, list[Any]]:
        user = _require_text(user_id, "user_id")
        rows = await self._repository.list_desc(user, None)
        return {"items": [as_contract(row) for row in rows]}

    async def recent_image_urls(self, user_id: str, limit: int = 3) -> list[str]:
        """推荐读取的“最近三张”个人喜好图（03）。"""
        user = _require_text(user_id, "user_id")
        if limit <= 0:
            return []
        rows = await self._repository.list_desc(user, limit)
        return [row.image_url for row in rows][:limit]

    # ---- 采集写入（trends 调用） ------------------------------------------ #
    async def add_collected(
        self,
        user_id: str,
        *,
        image_url: str,
        tags: Sequence[str] | None = None,
        source_url: str | None = None,
    ) -> Inspiration | None:
        """采集写入一条喜好图；图片已存在时返回 None。

        去重口径：喜好图按图片 URL 去重（同一张图不会重复出现在喜好列表）；
        案例级的“按来源去重”由 trends 的 TrendRepository 负责。
        """
        user = _require_text(user_id, "user_id")
        remote = _require_text(image_url, "image_url")
        source = _normalize_source_url(source_url)
        if await self._repository.find_by_image_url(user, remote):
            return None

        image_id, stored_url = await self._ingest(remote)
        inspiration = Inspiration(
            inspiration_id=self._id_factory(),
            image_id=image_id,
            image_url=stored_url,
            tags=_normalize_tags(tags),
            source_url=source,
        )
        await self._repository.add(user, inspiration, self._now())
        return as_contract(inspiration)


# --------------------------------------------------------------------------- #
# 默认单例（供路由/容器接线；测试请自行构造，避免共享状态）
# --------------------------------------------------------------------------- #
_default_service: InspirationsService | None = None


def get_inspiration_service() -> InspirationsService:
    global _default_service
    if _default_service is None:
        _default_service = InspirationsService()
    return _default_service


def reset_default_service() -> None:
    global _default_service
    _default_service = None
