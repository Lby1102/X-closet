"""本模块公共工具：错误基类、时间、request_id 与共享契约桥接。

与窗口 B 的三个文件保持同一约定：

* 错误携带 02 约定的 ``code / message / retryable``，同时带 ``status_code``；
* 时间统一用带时区的 UTC（02 的日期/天气按 Asia/Shanghai 展示，存储用 UTC 便于比较）；
* 共享契约优先用郭子骏 ``app.shared`` 的同名模型，缺失时退回本模块的等价模型，
  骨架落地后不需要改业务代码。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

__all__ = [
    "ModuleError",
    "as_shared",
    "brief",
    "request_id",
    "shared_contract",
    "utcnow",
]


class ModuleError(Exception):
    """本模块统一的业务错误：可直接映射成 02 的 error 结构。"""

    code = "INVALID_REQUEST"
    status_code = 422
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        if retryable is not None:
            self.retryable = retryable

    def to_error(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }


def utcnow() -> datetime:
    """当前时间（带时区）。"""
    return datetime.now(timezone.utc)


def request_id() -> str:
    """02：每个响应带 request_id，用于定位问题。"""
    return f"req_{uuid.uuid4().hex[:16]}"


def shared_contract(name: str) -> Any | None:
    """取郭子骏 ``app/shared`` 中的同名 Pydantic 模型；缺失时返回 None。

    具体探测逻辑沿用窗口 B 在 ``inspirations.py`` 里实现的唯一接入点，
    避免出现两套“共享契约探测”。
    """
    try:
        from .inspirations import shared_contract as _shared
    except Exception:  # pragma: no cover - 包结构异常时的兜底
        return None
    try:
        return _shared(name)
    except Exception:  # pragma: no cover
        return None


def as_shared(name: str, value: Any) -> Any:
    """把本地等价模型转成当前生效的共享契约实例；失败时退回本地模型。"""
    model = shared_contract(name)
    if model is None or value is None:
        return value
    dump = getattr(value, "model_dump", None)
    if dump is None:  # pragma: no cover - 非 pydantic 值
        return value
    try:
        return model(**dump())
    except Exception:  # pragma: no cover - 共享模型字段漂移时的兜底
        return value


def brief(exc: BaseException, limit: int = 160) -> str:
    """把外部异常压成一行短描述，避免 error.message / warnings 过长。"""
    text = " ".join(str(exc).split())
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__
