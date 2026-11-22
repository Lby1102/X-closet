"""服装识别（``POST /garments:analyze``）：视觉语言 API 出候选标签，不直接入库。

契约（02 / 03）
    POST /garments:analyze  {image_id*}  -> 202 Job（type=garment_analysis）
    Job 成功 result = {image_id, tags}；tags 交用户修改后，由 wardrobe 的
    ``POST /garments`` 确认入库。识别本身绝不写衣橱。

本文件负责两件事：
1. 02 的 Tags 契约与校验（衣橱入库、改标签共用同一份校验）；
2. 视觉模型端口的调用与失败映射（外部失败 -> Job.error=UPSTREAM_FAILED，可重试）。

真实接入点：把 :class:`VisionAnalyzer` 换成模型网关的视觉能力（03 的
「视觉语言 API」），入参是图片 URL 与字节，出参是 02 的 Tags 字段。
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal, Mapping, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .common import ModuleError, brief
from .images import Image

__all__ = [
    "CATEGORIES",
    "TAGS_FIELDS",
    "WARMTH_LEVELS",
    "FailingVisionAnalyzer",
    "GarmentRecognitionService",
    "HeuristicVisionAnalyzer",
    "InvalidTags",
    "RecognitionError",
    "RecognitionUpstreamError",
    "Tags",
    "VisionAnalyzer",
    "normalize_tags",
]

CATEGORIES: tuple[str, ...] = ("top", "bottom", "dress", "shoes", "outerwear")
WARMTH_LEVELS: tuple[str, ...] = ("light", "medium", "warm")
TAGS_FIELDS: frozenset[str] = frozenset(
    {"category", "color", "pattern", "fit", "warmth", "style_tags"}
)
MAX_STYLE_TAGS = 32
MAX_TAG_LEN = 40


# --------------------------------------------------------------------------- #
# 错误
# --------------------------------------------------------------------------- #
class RecognitionError(ModuleError):
    code = "INVALID_TAGS"


class InvalidTags(RecognitionError):
    """02：422 字段或业务条件无效。"""

    code = "INVALID_TAGS"
    status_code = 422


class RecognitionUpstreamError(RecognitionError):
    """02 / 03：502 外部服务失败；任务侧写进 Job.error 且可重试。"""

    code = "UPSTREAM_FAILED"
    status_code = 502
    retryable = True


# --------------------------------------------------------------------------- #
# 02 的 Tags
# --------------------------------------------------------------------------- #
class Tags(BaseModel):
    """02：category*；color*；pattern?；fit?；warmth?；style_tags?。"""

    model_config = ConfigDict(extra="forbid")

    category: Literal["top", "bottom", "dress", "shoes", "outerwear"]
    color: str
    pattern: str | None = None
    fit: str | None = None
    warmth: Literal["light", "medium", "warm"] | None = None
    style_tags: list[str] = Field(default_factory=list)


def _tag_text(value: Any, field: str, *, required: bool = False) -> str | None:
    if value is None:
        if required:
            raise InvalidTags(f"{field} 不能为空")
        return None
    if not isinstance(value, str):
        raise InvalidTags(f"{field} 必须是字符串")
    text = value.strip()
    if not text:
        if required:
            raise InvalidTags(f"{field} 不能为空")
        return None
    return text[:MAX_TAG_LEN]


def _style_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw: Any = [value]
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        raise InvalidTags("style_tags 必须是字符串数组")
    result: list[str] = []
    for item in raw:
        text = _tag_text(item, "style_tags")
        if text and text not in result:
            result.append(text)
        if len(result) >= MAX_STYLE_TAGS:
            break
    return result


def normalize_tags(raw: Any) -> Tags:
    """校验并归一化 Tags：PATCH 也用这里，因此提交的是**完整**标签。

    省略的可选字段（pattern / fit / warmth / style_tags）会归一化为
    None / []，即“整体替换”而不是“合并”，与 02「提交完整标签」一致。
    """
    if isinstance(raw, Tags):
        data: dict[str, Any] = raw.model_dump()
    elif isinstance(raw, BaseModel):
        data = raw.model_dump()
    elif isinstance(raw, Mapping):
        data = dict(raw)
    else:
        raise InvalidTags("tags 必须是对象")

    unknown = set(data) - TAGS_FIELDS
    if unknown:
        raise InvalidTags(f"未知标签字段：{sorted(unknown)}")

    category = _tag_text(data.get("category"), "category", required=True)
    if category not in CATEGORIES:
        raise InvalidTags(f"category 必须是 {'/'.join(CATEGORIES)} 之一")
    color = _tag_text(data.get("color"), "color", required=True)

    warmth = _tag_text(data.get("warmth"), "warmth")
    if warmth is not None and warmth not in WARMTH_LEVELS:
        raise InvalidTags(f"warmth 必须是 {'/'.join(WARMTH_LEVELS)} 之一")

    return Tags(
        category=category,
        color=color,
        pattern=_tag_text(data.get("pattern"), "pattern"),
        fit=_tag_text(data.get("fit"), "fit"),
        warmth=warmth,
        style_tags=_style_tags(data.get("style_tags")),
    )


# --------------------------------------------------------------------------- #
# 视觉模型端口 + 两个可替换实现
# --------------------------------------------------------------------------- #
class VisionAnalyzer(Protocol):
    """视觉语言 API 端口。真实实现：模型网关（03「视觉语言 API」）。"""

    async def analyze_garment(
        self, image: Image, content: bytes
    ) -> Mapping[str, Any]: ...


_COLOR_PALETTE = (
    "white",
    "black",
    "navy",
    "beige",
    "grey",
    "olive",
    "red",
    "denim",
)
_STYLE_PALETTE = ("minimal", "casual", "business", "street", "sporty")
_FIT_PALETTE = ("slim", "regular", "oversized")
_PATTERN_PALETTE = ("solid", "stripe", "check", "floral")


class HeuristicVisionAnalyzer:
    """离线视觉模型替身：同一图片内容得到稳定标签，可用 :meth:`hint` 覆盖。

    不是随机数：用内容摘要取模选标签，因此“同一张图 -> 同一组标签”，
    测试与演示可复现。真实模型接入后本类只用于自测。
    """

    def __init__(self, hints: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        self._hints: dict[str, Mapping[str, Any]] = dict(hints or {})

    def hint(self, key: str, tags: Mapping[str, Any]) -> None:
        """按 image_id 或 image_url 预置识别结果（测试/演示用）。"""
        self._hints[str(key)] = dict(tags)

    async def analyze_garment(
        self, image: Image, content: bytes
    ) -> Mapping[str, Any]:
        for key in (image.image_id, image.image_url):
            if key in self._hints:
                return dict(self._hints[key])
        digest = hashlib.sha256(content or b"").digest()
        return {
            "category": CATEGORIES[digest[0] % len(CATEGORIES)],
            "color": _COLOR_PALETTE[digest[1] % len(_COLOR_PALETTE)],
            "pattern": _PATTERN_PALETTE[digest[2] % len(_PATTERN_PALETTE)],
            "fit": _FIT_PALETTE[digest[3] % len(_FIT_PALETTE)],
            "warmth": WARMTH_LEVELS[digest[4] % len(WARMTH_LEVELS)],
            "style_tags": [_STYLE_PALETTE[digest[5] % len(_STYLE_PALETTE)]],
        }


class FailingVisionAnalyzer:
    """视觉服务故障替身（504/超时），用于验证 Job 失败分支。"""

    def __init__(self, message: str = "视觉模型 504", *, retryable: bool = True) -> None:
        self._message = message
        self._retryable = retryable

    async def analyze_garment(
        self, image: Image, content: bytes
    ) -> Mapping[str, Any]:
        raise RecognitionUpstreamError(self._message, retryable=self._retryable)


# --------------------------------------------------------------------------- #
# 业务服务
# --------------------------------------------------------------------------- #
class GarmentRecognitionService:
    """识别服务：校验图片归属 -> 调视觉模型 -> 校验标签 -> 返回 {image_id, tags}。"""

    def __init__(
        self,
        *,
        images: Any,
        vision: VisionAnalyzer | None = None,
    ) -> None:
        self._images = images
        self._vision = vision or HeuristicVisionAnalyzer()

    @property
    def vision(self) -> VisionAnalyzer:
        return self._vision

    @property
    def images(self) -> Any:
        return self._images

    async def analyze(self, user_id: str, image_id: str) -> dict[str, Any]:
        user = _require_text(user_id, "user_id")
        key = _require_text(image_id, "image_id")
        image = await self._images.get(user, key)
        content = await self._images.content(user, key)

        try:
            raw = await self._vision.analyze_garment(image, content)
        except RecognitionError:
            raise
        except Exception as exc:  # 网络/超时/协议错误：外部服务失败，可重试
            raise RecognitionUpstreamError(f"视觉模型调用失败：{brief(exc)}") from exc

        try:
            tags = normalize_tags(raw)
        except RecognitionError as exc:
            # 模型返回不可用标签属于外部服务问题（02 的 Job.error=UPSTREAM_FAILED）
            raise RecognitionUpstreamError(f"识别结果不可用：{exc.message}") from exc

        return {"image_id": image.image_id, "tags": tags.model_dump()}


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecognitionError(f"{field} 不能为空", code="INVALID_FIELD")
    return value.strip()
