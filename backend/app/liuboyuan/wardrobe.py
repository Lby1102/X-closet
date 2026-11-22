"""衣橱业务：确认入库、查询、改标签，以及「确认入库后」的向量同步。

契约（02 / 03 / 04）
    POST   /garments             {image_id*, tags*}      -> 201 Garment
    GET    /garments             ?category=              -> 200 {items: Garment[]}
    PATCH  /garments/{garment_id} {tags*}（完整标签）    -> 200 Garment
    03：查询输入 user_id + 可选品类；只有用户确认入库的单品才进入衣橱，
    推荐与以图搜衣都只读这里，所以本文件是衣橱的唯一写入口。

向量同步接入点（03「以图搜衣」：服装图与参考图用同一嵌入模型，向量关联 garment_id）
    1. :meth:`WardrobeService.create` —— 用户确认入库成功后立即
       :meth:`WardrobeService.index_garment`（同一事务之后、返回 Garment 之前）；
    2. :meth:`WardrobeService.patch` —— 品类变化时重建向量（向量按品类召回，
       category 变了必须 upsert 新品类）；只改颜色等字段不重建；
    3. :meth:`WardrobeService.ensure_indexed` —— 检索前的懒补建：入库时向量
       写入失败不影响衣橱，检索时补建并返回仍失败的 garment_id（转 warnings）。

真实接入点（郭子骏的 PostgreSQL + pgvector）：实现 :class:`GarmentRepository`
与 :class:`GarmentVectorStore` 两个端口即可，方法签名不用改；向量列建议
``vector(512)``，写入用 ``ON CONFLICT (garment_id) DO UPDATE``。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, Sequence

from pydantic import BaseModel, ConfigDict

from .common import ModuleError, as_shared, brief, utcnow
from .images import Image
from .recognition import CATEGORIES, Tags, normalize_tags

__all__ = [
    "DEFAULT_EMBEDDING_DIM",
    "Garment",
    "GarmentNotFound",
    "GarmentRepository",
    "GarmentVectorStore",
    "HashImageEmbedder",
    "ImageEmbedder",
    "InMemoryGarmentRepository",
    "InMemoryGarmentVectorStore",
    "WardrobeError",
    "WardrobeService",
    "cosine_similarity",
]

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_DIM = 512


# --------------------------------------------------------------------------- #
# 错误
# --------------------------------------------------------------------------- #
class WardrobeError(ModuleError):
    code = "INVALID_INPUT"
    status_code = 422


class GarmentNotFound(WardrobeError):
    """02：404 表示资源在当前用户下不存在。"""

    code = "GARMENT_NOT_FOUND"
    status_code = 404


# --------------------------------------------------------------------------- #
# 02 的 Garment
# --------------------------------------------------------------------------- #
class Garment(BaseModel):
    """02：garment_id / image_id / image_url / tags。"""

    model_config = ConfigDict(extra="ignore")

    garment_id: str
    image_id: str
    image_url: str
    tags: Tags


# --------------------------------------------------------------------------- #
# 图像 Embedding（同一个端口同时服务服装图与参考图）
# --------------------------------------------------------------------------- #
class ImageEmbedder(Protocol):
    """图像 Embedding 端口。真实实现：03 的「图像 Embedding API」。

    服装图（入库时）与参考图（检索时）必须用**同一个实现/同一个模型**，
    否则向量不在同一空间，相似度没有意义。
    """

    @property
    def model(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    async def embed_image(self, image: Image, content: bytes) -> list[float]: ...


class HashImageEmbedder:
    """离线确定性替身：同一图片内容 -> 同一单位向量。

    不做真实语义理解，只保证“同一模型空间 + 可复现”：
    相同内容余弦相似度 1.0，不同内容近似随机。真实模型接入后仅用于自测。
    """

    def __init__(self, *, dimension: int = DEFAULT_EMBEDDING_DIM) -> None:
        if dimension <= 0:
            raise ValueError("dimension 必须大于 0")
        self._dimension = int(dimension)

    @property
    def model(self) -> str:
        return f"hash-embed-{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_image(self, image: Image, content: bytes) -> list[float]:
        digest = hashlib.sha256(bytes(content or b"")).digest()
        repeated = digest * (self._dimension // len(digest) + 1)
        values = [(byte - 127.5) / 127.5 for byte in repeated[: self._dimension]]
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:  # pragma: no cover - 极短内容的兜底
            return [0.0] * self._dimension
        return [value / norm for value in values]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度；维度不一致或零向量返回 0.0（不抛错，交给调用方排序）。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for left, right in zip(a, b):
        dot += left * right
        norm_a += left * left
        norm_b += right * right
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


# --------------------------------------------------------------------------- #
# 向量存储端口（真实实现：pgvector）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _VectorRecord:
    category: str
    vector: tuple[float, ...]
    model: str


class GarmentVectorStore(Protocol):
    """按 (user_id, category) 召回；向量按 garment_id 关联。真实实现：pgvector。"""

    async def upsert_garment(
        self,
        user_id: str,
        *,
        garment_id: str,
        category: str,
        vector: Sequence[float],
        model: str,
    ) -> None: ...

    async def search_garments(
        self,
        user_id: str,
        *,
        category: str,
        vector: Sequence[float],
        limit: int,
    ) -> list[tuple[str, float]]: ...

    async def indexed_garments(self, user_id: str) -> set[str]: ...


class InMemoryGarmentVectorStore:
    """进程内实现：user_id + category 过滤，余弦相似度排序，维度不一致记 0 分。"""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, _VectorRecord]] = {}
        self._lock = asyncio.Lock()

    async def upsert_garment(
        self,
        user_id: str,
        *,
        garment_id: str,
        category: str,
        vector: Sequence[float],
        model: str,
    ) -> None:
        record = _VectorRecord(
            category=str(category),
            vector=tuple(float(value) for value in vector),
            model=str(model),
        )
        async with self._lock:
            # 与 SQL 的 ON CONFLICT (garment_id) DO UPDATE 等价：换品类即覆盖
            self._data.setdefault(user_id, {})[str(garment_id)] = record

    async def search_garments(
        self,
        user_id: str,
        *,
        category: str,
        vector: Sequence[float],
        limit: int,
    ) -> list[tuple[str, float]]:
        if limit <= 0:
            return []
        async with self._lock:
            rows = dict(self._data.get(user_id, {}))
        scored: list[tuple[str, float]] = []
        for garment_id, record in rows.items():
            if record.category != category:
                continue
            scored.append((garment_id, cosine_similarity(vector, record.vector)))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:limit]

    async def indexed_garments(self, user_id: str) -> set[str]:
        async with self._lock:
            return set(self._data.get(user_id, {}))

    async def clear(self, user_id: str) -> None:
        """清空某用户的向量（重建索引 / 模拟 pgvector 数据丢失时用）。"""
        async with self._lock:
            self._data.pop(user_id, None)


# --------------------------------------------------------------------------- #
# 衣橱仓储端口（真实实现：PostgreSQL）
# --------------------------------------------------------------------------- #
class GarmentRepository(Protocol):
    """衣橱存储端口。真实实现：PostgreSQL garments 表（含 tags 列与 user_id）。"""

    async def add(
        self, user_id: str, garment: Garment, created_at: datetime
    ) -> Garment: ...

    async def list(
        self, user_id: str, category: str | None = None
    ) -> list[Garment]: ...

    async def get(self, user_id: str, garment_id: str) -> Garment | None: ...

    async def replace(
        self, user_id: str, garment_id: str, garment: Garment
    ) -> Garment | None: ...


@dataclass
class _GarmentRecord:
    garment: Garment
    created_at: datetime
    seq: int


class InMemoryGarmentRepository:
    """进程内实现：按 user_id 隔离，created_at 倒序（同刻按自增 seq）。"""

    def __init__(self) -> None:
        self._records: dict[str, list[_GarmentRecord]] = {}
        self._seq = 0
        self._lock = asyncio.Lock()

    async def add(
        self, user_id: str, garment: Garment, created_at: datetime
    ) -> Garment:
        async with self._lock:
            self._seq += 1
            self._records.setdefault(user_id, []).append(
                _GarmentRecord(
                    garment=garment.model_copy(deep=True),
                    created_at=created_at,
                    seq=self._seq,
                )
            )
        return garment.model_copy(deep=True)

    def _sorted(self, user_id: str) -> list[_GarmentRecord]:
        rows = self._records.get(user_id, [])
        return sorted(rows, key=lambda record: (record.created_at, record.seq), reverse=True)

    async def list(
        self, user_id: str, category: str | None = None
    ) -> list[Garment]:
        async with self._lock:
            rows = [record.garment.model_copy(deep=True) for record in self._sorted(user_id)]
        if category is not None:
            rows = [row for row in rows if row.tags.category == category]
        return rows

    async def get(self, user_id: str, garment_id: str) -> Garment | None:
        target = str(garment_id)
        async with self._lock:
            for record in self._sorted(user_id):
                if record.garment.garment_id == target:
                    return record.garment.model_copy(deep=True)
        return None

    async def replace(
        self, user_id: str, garment_id: str, garment: Garment
    ) -> Garment | None:
        target = str(garment_id)
        async with self._lock:
            for record in self._records.get(user_id, []):
                if record.garment.garment_id == target:
                    record.garment = garment.model_copy(deep=True)
                    return record.garment.model_copy(deep=True)
        return None


# --------------------------------------------------------------------------- #
# 业务服务
# --------------------------------------------------------------------------- #
class WardrobeService:
    """衣橱唯一写入口：确认入库（含向量同步）、查询、整体替换标签。"""

    def __init__(
        self,
        *,
        images: Any,
        repository: GarmentRepository | None = None,
        embedder: ImageEmbedder | None = None,
        vectors: GarmentVectorStore | None = None,
        now=utcnow,
        id_factory=None,
    ) -> None:
        self._images = images
        self._repository = repository or InMemoryGarmentRepository()
        self._embedder = embedder or HashImageEmbedder()
        self._vectors = vectors or InMemoryGarmentVectorStore()
        self._now = now
        self._id_factory = id_factory or (lambda: f"gar_{uuid.uuid4().hex[:12]}")

    # ---- 以图搜衣共用同一实例，保证“同一图像模型” ------------------------ #
    @property
    def embedder(self) -> ImageEmbedder:
        return self._embedder

    @property
    def vectors(self) -> GarmentVectorStore:
        return self._vectors

    # ---- 写：POST /garments（用户确认/修正标签后入库） -------------------- #
    async def create(self, user_id: str, image_id: str, raw_tags: Any) -> Any:
        user = _require_text(user_id, "user_id")
        key = _require_text(image_id, "image_id")
        image = await self._images.get(user, key)  # 归属校验：越权 404
        tags = normalize_tags(raw_tags)  # 422 INVALID_TAGS
        garment = Garment(
            garment_id=self._id_factory(),
            image_id=image.image_id,
            image_url=image.image_url,
            tags=tags,
        )
        await self._repository.add(user, garment, self._now())
        # 接入点 1：确认入库成功后立即同步向量（失败不回滚衣橱，检索时补建）
        await self.index_garment(user, garment)
        return as_shared("Garment", garment)

    # ---- 读：GET /garments ------------------------------------------------ #
    async def list(self, user_id: str, category: str | None = None) -> list[Any]:
        user = _require_text(user_id, "user_id")
        normalized = _normalize_category(category)
        rows = await self._repository.list(user, normalized)
        return [as_shared("Garment", row) for row in rows]

    async def list_items(
        self, user_id: str, category: str | None = None
    ) -> dict[str, list[Any]]:
        return {"items": await self.list(user_id, category)}

    async def get(self, user_id: str, garment_id: str) -> Any:
        user = _require_text(user_id, "user_id")
        key = _require_text(garment_id, "garment_id")
        garment = await self._repository.get(user, key)
        if garment is None:
            raise GarmentNotFound(f"衣物不存在：{key}")
        return as_shared("Garment", garment)

    # ---- 改：PATCH /garments/{garment_id}（提交完整标签） ------------------ #
    async def patch(self, user_id: str, garment_id: str, raw_tags: Any) -> Any:
        user = _require_text(user_id, "user_id")
        key = _require_text(garment_id, "garment_id")
        garment = await self._repository.get(user, key)
        if garment is None:
            raise GarmentNotFound(f"衣物不存在：{key}")

        tags = normalize_tags(raw_tags)  # 完整标签：省略的可选字段被清空
        updated = garment.model_copy(update={"tags": tags})
        stored = await self._repository.replace(user, key, updated)
        if stored is None:  # pragma: no cover - 并发删除时的兜底
            raise GarmentNotFound(f"衣物不存在：{key}")

        # 接入点 2：品类变化 -> 向量按新品类重建（只改颜色等字段不重建）
        if tags.category != garment.tags.category:
            await self.index_garment(user, stored)
        return as_shared("Garment", stored)

    # ---- 向量同步 --------------------------------------------------------- #
    async def index_garment(self, user_id: str, garment: Any) -> bool:
        """为一件已确认入库的衣物生成向量；服装图与参考图共用同一个 Embedder。"""
        try:
            content = await self._images.content(user_id, garment.image_id)
            image = Image(
                image_id=garment.image_id, image_url=garment.image_url
            )
            vector = await self._embedder.embed_image(image, content)
            await self._vectors.upsert_garment(
                user_id,
                garment_id=garment.garment_id,
                category=garment.tags.category,
                vector=vector,
                model=self._embedder.model,
            )
            return True
        except Exception as exc:  # 索引失败不影响衣橱写入
            logger.warning(
                "衣橱向量索引失败 garment_id=%s: %s", garment.garment_id, brief(exc)
            )
            return False

    async def ensure_indexed(self, user_id: str) -> list[str]:
        """接入点 3：检索前为缺失向量的衣物补建，返回仍失败的 garment_id。"""
        user = _require_text(user_id, "user_id")
        indexed = await self._vectors.indexed_garments(user)
        failed: list[str] = []
        for garment in await self._repository.list(user):
            if garment.garment_id in indexed:
                continue
            if not await self.index_garment(user, garment):
                failed.append(garment.garment_id)
        return failed


def _normalize_category(category: str | None) -> str | None:
    if category is None:
        return None
    value = str(category).strip()
    if not value:
        return None
    if value not in CATEGORIES:
        raise WardrobeError(
            f"category 必须是 {'/'.join(CATEGORIES)} 之一", code="INVALID_INPUT"
        )
    return value


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WardrobeError(f"{field} 不能为空", code="INVALID_FIELD")
    return value.strip()
