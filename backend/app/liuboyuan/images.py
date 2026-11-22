"""上传图片（``POST /images``）在本模块的 Mock 实现。

02 的约定：``multipart/form-data``，照片为 JPEG 或 PNG，单张上限 10 MB，
超限 413，无法解码 422。图片按用户隔离，其他用户的 image_id 一律当不存在（404）。

真实接入点（郭子骏的「图片上传」模块）：对象存储保存文件、PostgreSQL 保存
``image_id / 所属用户 / 存储位置``。替换时只需实现 :class:`ImageStore` 的
``add / get / content`` 三个方法，识别、衣库与以图搜衣都不用改。
"""

from __future__ import annotations

import asyncio
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from .common import ModuleError

__all__ = [
    "MAX_IMAGE_BYTES",
    "Image",
    "ImageError",
    "ImageNotFound",
    "ImageStore",
    "ImageTooLarge",
    "InMemoryImageStore",
    "InvalidImage",
    "default_media_url",
    "sniff_media_type",
]

MAX_IMAGE_BYTES = 10 * 1024 * 1024
_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------- #
# 错误（02：404 资源不存在，413 超过 10 MB，422 图片无法解码）
# --------------------------------------------------------------------------- #
class ImageError(ModuleError):
    code = "INVALID_IMAGE"


class ImageNotFound(ImageError):
    code = "IMAGE_NOT_FOUND"
    status_code = 404


class InvalidImage(ImageError):
    code = "INVALID_IMAGE"
    status_code = 422


class ImageTooLarge(ImageError):
    code = "IMAGE_TOO_LARGE"
    status_code = 413


# --------------------------------------------------------------------------- #
# 模型与工具
# --------------------------------------------------------------------------- #
class Image(BaseModel):
    """02 的 Image：``image_id:string；image_url:string``，另带两个便于排查的字段。"""

    model_config = ConfigDict(extra="ignore")

    image_id: str
    image_url: str
    media_type: str = "image/jpeg"
    size_bytes: int = 0


def sniff_media_type(data: bytes) -> str | None:
    """按文件头判断格式；无法识别返回 None（对应 02 的“图片无法解码”）。"""
    if data.startswith(_JPEG_MAGIC):
        return "image/jpeg"
    if data.startswith(_PNG_MAGIC):
        return "image/png"
    return None


def default_media_url(image_id: str, media_type: str = "image/jpeg") -> str:
    """image_id -> image_url。真实接入点：对象存储/CDN 返回的地址。"""
    suffix = "png" if media_type == "image/png" else "jpg"
    return f"/media/{image_id}.{suffix}"


# --------------------------------------------------------------------------- #
# 端口
# --------------------------------------------------------------------------- #
class ImageStore(Protocol):
    """图片存储端口。真实实现：郭子骏的上传模块（对象存储 + SQL 记录）。"""

    async def add(
        self, user_id: str, data: bytes, *, content_type: str | None = None
    ) -> Image: ...

    async def get(self, user_id: str, image_id: str) -> Image: ...

    async def content(self, user_id: str, image_id: str) -> bytes: ...


class InMemoryImageStore:
    """进程内实现：内存保存内容，跨用户读取一律 404，保证用户隔离。"""

    def __init__(
        self,
        *,
        max_bytes: int = MAX_IMAGE_BYTES,
        media_url=default_media_url,
    ) -> None:
        self._max_bytes = max_bytes
        self._media_url = media_url
        self._content: dict[tuple[str, str], bytes] = {}
        self._images: dict[tuple[str, str], Image] = {}
        self._seq = 0
        self._lock = asyncio.Lock()

    async def add(
        self, user_id: str, data: bytes, *, content_type: str | None = None
    ) -> Image:
        payload = bytes(data or b"")
        if not payload:
            raise InvalidImage("图片内容为空，无法解码")
        if len(payload) > self._max_bytes:
            raise ImageTooLarge(
                f"单张图片上限 {self._max_bytes // (1024 * 1024)} MB，"
                f"本次 {len(payload)} 字节",
                code="IMAGE_TOO_LARGE",
            )
        # 以文件头为准：声明的 Content-Type 不可信，文件头对不上即视为无法解码（02）
        media_type = sniff_media_type(payload)
        if media_type is None:
            raise InvalidImage("图片无法解码：仅支持 JPEG 或 PNG")

        async with self._lock:
            self._seq += 1
            image_id = f"img_{self._seq}"
            image = Image(
                image_id=image_id,
                image_url=self._media_url(image_id, media_type),
                media_type=media_type,
                size_bytes=len(payload),
            )
            key = (user_id, image_id)
            self._content[key] = payload
            self._images[key] = image
        return image

    async def get(self, user_id: str, image_id: str) -> Image:
        key = (user_id, str(image_id))
        async with self._lock:
            image = self._images.get(key)
        if image is None:
            # 其他用户的图片同样返回 404，不泄露资源是否存在（02）
            raise ImageNotFound(f"图片不存在：{image_id}")
        return image

    async def content(self, user_id: str, image_id: str) -> bytes:
        key = (user_id, str(image_id))
        async with self._lock:
            payload = self._content.get(key)
        if payload is None:
            raise ImageNotFound(f"图片不存在：{image_id}")
        return payload
