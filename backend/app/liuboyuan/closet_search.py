"""以图搜衣：``closet.search(user_id, image_id, limit=10) -> Candidate[]``。

03 的交接（提供给唐睿彬的推荐编排）
    输入：user_id、参考图 image_id、limit（默认 10）
    输出：Candidate[]，每项 ``garment_id / image_url / score``；**每品类取前 limit 个**
    时机：参考图已上传、衣物已确认入库；再与天气/禁忌过滤结果取交集

实现要点
1. 服装图与参考图用**同一个** :class:`~app.liuboyuan.wardrobe.ImageEmbedder`
   实例（直接取 ``wardrobe.embedder``），保证同一模型空间；
2. 只召回 ``wardrobe`` 里当前用户已确认入库的单品（衣橱仓储即确认后的数据源）；
3. 向量按品类召回：对衣橱中实际存在的品类逐类调用向量检索，各取前 limit；
4. 向量缺失先懒补建；补建仍失败则跳过该件并给出 warnings（``search`` 的返回
   契约不变，编排层可用 :meth:`ClosetSearchService.search_with_warnings`）；
5. 参考图不是当前用户的 -> 404 IMAGE_NOT_FOUND；图片向量化或向量检索不可用
   -> 502 UPSTREAM_FAILED（retryable），由编排决定转 warning 还是终止。
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict

from .common import ModuleError, brief
from .recognition import CATEGORIES

__all__ = [
    "DEFAULT_SEARCH_LIMIT",
    "MAX_SEARCH_LIMIT",
    "Candidate",
    "ClosetSearchError",
    "ClosetSearchService",
    "SearchUpstreamError",
]

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_LIMIT = 10
MAX_SEARCH_LIMIT = 50


# --------------------------------------------------------------------------- #
# 错误
# --------------------------------------------------------------------------- #
class ClosetSearchError(ModuleError):
    code = "INVALID_INPUT"
    status_code = 422


class SearchUpstreamError(ClosetSearchError):
    """02 / 03：502 外部服务失败（Embedding 或向量检索不可用），可重试。"""

    code = "UPSTREAM_FAILED"
    status_code = 502
    retryable = True


# --------------------------------------------------------------------------- #
# 03 的 Candidate
# --------------------------------------------------------------------------- #
class Candidate(BaseModel):
    """03：Candidate 含 garment_id、image_url、score。"""

    model_config = ConfigDict(extra="ignore")

    garment_id: str
    image_url: str
    score: float


# --------------------------------------------------------------------------- #
# 业务服务
# --------------------------------------------------------------------------- #
class ClosetSearchService:
    """以图搜衣。依赖衣橱服务（含 Embedder 与向量存储），不重复持有模型。"""

    def __init__(
        self,
        *,
        wardrobe: Any,
        images: Any,
        max_limit: int = MAX_SEARCH_LIMIT,
    ) -> None:
        self._wardrobe = wardrobe
        self._images = images
        self._max_limit = max(1, int(max_limit))

    @property
    def wardrobe(self) -> Any:
        return self._wardrobe

    @property
    def images(self) -> Any:
        return self._images

    async def search(
        self, user_id: str, image_id: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[Candidate]:
        """契约方法：返回 Candidate[]（每品类最多 limit 件）。"""
        candidates, _warnings = await self.search_with_warnings(user_id, image_id, limit)
        return candidates

    async def search_with_warnings(
        self, user_id: str, image_id: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> tuple[list[Candidate], list[str]]:
        """同 :meth:`search`，额外返回编排层可用的 warnings（如向量缺失）。"""
        user = _require_text(user_id, "user_id")
        key = _require_text(image_id, "image_id")
        bounded = _clamp_limit(limit, self._max_limit)

        # 归属校验：别人的参考图当不存在（404），不泄露图片是否存在
        image = await self._images.get(user, key)

        try:
            garments = await self._wardrobe.list(user)
        except Exception as exc:
            raise SearchUpstreamError(f"衣橱查询失败：{brief(exc)}") from exc
        if not garments:
            # 03：衣橱为空时由推荐层返回 WARDROBE_EMPTY；这里给空候选
            return [], []

        warnings: list[str] = []
        try:
            failed = await self._wardrobe.ensure_indexed(user)
        except Exception as exc:
            raise SearchUpstreamError(f"向量索引查询失败：{brief(exc)}") from exc
        if failed:
            warnings.append(
                f"有 {len(failed)} 件衣物尚未生成图像向量，本次未参与检索："
                + ", ".join(sorted(failed))
            )

        content = await self._images.content(user, key)
        try:
            # 与入库时用的是 wardrobe.embedder（同一模型、同一实例）
            vector = await self._wardrobe.embedder.embed_image(image, content)
        except Exception as exc:
            raise SearchUpstreamError(f"参考图向量生成失败：{brief(exc)}") from exc

        by_id = {garment.garment_id: garment for garment in garments}
        present = {garment.tags.category for garment in garments}
        candidates: list[Candidate] = []
        for category in CATEGORIES:
            if category not in present:
                continue
            try:
                scored = await self._wardrobe.vectors.search_garments(
                    user, category=category, vector=vector, limit=bounded
                )
            except Exception as exc:
                raise SearchUpstreamError(f"向量检索失败：{brief(exc)}") from exc

            # 分数降序，同分按 garment_id 稳定排序
            ordered: Sequence[tuple[str, float]] = sorted(
                scored, key=lambda item: (-float(item[1]), str(item[0]))
            )
            for garment_id, score in ordered[:bounded]:
                garment = by_id.get(garment_id)
                if garment is None:
                    # 向量表残留了已删除/非本人的数据：忽略，不返回给调用方
                    logger.warning(
                        "向量结果未对应到衣橱记录，已忽略 user=%s garment=%s",
                        user,
                        garment_id,
                    )
                    continue
                candidates.append(
                    Candidate(
                        garment_id=garment.garment_id,
                        image_url=garment.image_url,
                        score=round(float(score), 4),
                    )
                )
        return candidates, warnings


def _clamp_limit(limit: Any, max_limit: int) -> int:
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = DEFAULT_SEARCH_LIMIT
    return max(1, min(value, max_limit))


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ClosetSearchError(f"{field} 不能为空", code="INVALID_FIELD")
    return value.strip()
