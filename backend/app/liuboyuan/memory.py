"""X Closet v1 · 偏好记忆（memory）— 负责人：刘博远

内部契约（见 03 内部接口与运行流程）
    memory.get(user_id)              -> Profile
    memory.update(user_id, feedback_id) -> {"profile_version": int}
    “读取返回 Profile；更新返回 {profile_version:int}”
    “更新须等 Feedback 入库；使用新偏好须等记忆 Job 成功”

  03 规定的 Profile 结构：
    {explicit: Preferences, learned_weights: dict, inspiration_image_urls: string[], version: int}

  03 规定的更新规则（逐条对应实现）：
    - 按用户串行处理                       -> _UserLocks + MemoryService.update
    - 以 feedback_id 去重                  -> MemoryState.applied_feedback_ids + store.commit
    - 近 30 天反馈按 style_tags 累计
      like +1 / dislike -1                 -> compute_learned_weights
    - 权重与版本同一事务提交               -> ProfileStore.commit(weights, version) 单次提交
    - 用户明确禁忌优先                     -> explicit.avoid_tags 覆盖学习到的正权重
    - 画像取最近三张个人喜好图             -> InspirationsService.recent_image_urls(limit=3)

  对外 HTTP 契约（02）：POST /feedback 由何梓森返回 {feedback_id, memory_job_id}；
  memory.update 就是 memory_job 的任务体，Job.result 为 {"profile_version": int}。

依赖注入（真实接入点）
    FeedbackSource     <- 何梓森的反馈记录（app.modules.feedback）
    PreferencesSource  <- 何梓森的基础偏好（GET/PUT /preferences，app.modules.preferences）
    ProfileStore       <- 郭子骏的 PostgreSQL 事务（权重 + 版本 + feedback_id 唯一键）
    缺失上述实现时使用本文件的 Mock（MockFeedbackSource / MockPreferencesSource /
    InMemoryProfileStore），可用内存态完整验证全部规则。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Literal, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

if __package__:  # pragma: no cover - 包内导入
    from .inspirations import (
        InspirationsService,
        get_inspiration_service,
        shared_contract,
        utcnow,
    )
else:  # pragma: no cover - 兼容 python memory.py 直接加载
    from inspirations import (
        InspirationsService,
        get_inspiration_service,
        shared_contract,
        utcnow,
    )

__all__ = [
    "FeedbackNotReady",
    "FeedbackRecord",
    "FeedbackSource",
    "InMemoryProfileStore",
    "MemoryError",
    "MemoryService",
    "MemoryState",
    "MockFeedbackSource",
    "MockPreferencesSource",
    "Preferences",
    "PreferencesSource",
    "Profile",
    "ProfileStore",
    "StoreConflict",
    "compute_learned_weights",
    "get_memory_service",
    "reset_default_service",
]

FEEDBACK_ACTIONS: tuple[str, ...] = ("like", "dislike")
_WINDOW_DAYS = 30
_MAX_UPDATE_ATTEMPTS = 3


# --------------------------------------------------------------------------- #
# 错误（02：422 字段或业务条件无效，409 幂等冲突）
# --------------------------------------------------------------------------- #
class MemoryError(Exception):
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


class FeedbackNotReady(MemoryError):
    """反馈尚未入库：03 要求“更新须等 Feedback 入库”。"""

    code = "FEEDBACK_NOT_FOUND"


class StoreConflict(MemoryError):
    """画像版本冲突（并发写入/事务重试）。"""

    code = "PROFILE_VERSION_CONFLICT"
    status_code = 409
    retryable = True


# --------------------------------------------------------------------------- #
# v1 契约模型（本地等价实现；app/shared 就绪后自动切换）
# --------------------------------------------------------------------------- #
class Preferences(BaseModel):
    model_config = ConfigDict(extra="ignore")

    liked_styles: list[str] = Field(default_factory=list)
    avoid_tags: list[str] = Field(default_factory=list)


class FeedbackRecord(BaseModel):
    """何梓森反馈记录在本模块的输入视图（03：feedback 记录供 memory.update 读取）。"""

    model_config = ConfigDict(extra="ignore")

    feedback_id: str
    user_id: str
    outfit_id: str
    action: Literal["like", "dislike"]
    reason: str | None = None
    style_tags: list[str] = Field(default_factory=list)
    created_at: datetime


class MemoryState(BaseModel):
    """持久化状态（真实实现对应 SQL 一行：版本 + 权重 + 已应用反馈）。"""

    model_config = ConfigDict(extra="ignore")

    version: int = 0
    learned_weights: dict[str, float] = Field(default_factory=dict)
    applied_feedback_ids: list[str] = Field(default_factory=list)


class Profile(BaseModel):
    """03 规定的 Profile。"""

    model_config = ConfigDict(extra="ignore")

    explicit: Preferences = Field(default_factory=Preferences)
    learned_weights: dict[str, float] = Field(default_factory=dict)
    inspiration_image_urls: list[str] = Field(default_factory=list)
    version: int = 0


def _as_contract(model: str, value: BaseModel) -> Any:
    shared = shared_contract(model)
    if shared is None:
        return value
    try:
        return shared(**value.model_dump())
    except Exception:  # pragma: no cover - 共享模型字段漂移时的兜底
        return value


# --------------------------------------------------------------------------- #
# 输入源端口 + Mock
# --------------------------------------------------------------------------- #
class FeedbackSource(Protocol):
    """反馈记录来源。真实接入点：何梓森的反馈表（含搭配的 style_tags）。"""

    async def get(self, user_id: str, feedback_id: str) -> FeedbackRecord | None: ...

    async def list_since(
        self, user_id: str, since: datetime
    ) -> list[FeedbackRecord]: ...


class PreferencesSource(Protocol):
    """基础偏好来源。真实接入点：何梓森的 /preferences（返回 Preferences）。"""

    async def get(self, user_id: str) -> Preferences: ...


class MockFeedbackSource:
    """联调 Mock：何梓森反馈接口未就绪时驱动 memory.update。"""

    def __init__(self, records: Iterable[FeedbackRecord] = ()) -> None:
        self._records: dict[tuple[str, str], FeedbackRecord] = {}
        for record in records:
            self.add(record)

    def add(self, record: FeedbackRecord) -> FeedbackRecord:
        self._records[(record.user_id, record.feedback_id)] = record
        return record

    def add_feedback(
        self,
        feedback_id: str,
        user_id: str,
        *,
        action: Literal["like", "dislike"] = "like",
        style_tags: Sequence[str] = (),
        outfit_id: str = "outfit_1",
        created_at: datetime | None = None,
        reason: str | None = None,
    ) -> FeedbackRecord:
        return self.add(
            FeedbackRecord(
                feedback_id=feedback_id,
                user_id=user_id,
                outfit_id=outfit_id,
                action=action,
                reason=reason,
                style_tags=list(style_tags),
                created_at=created_at or utcnow(),
            )
        )

    async def get(self, user_id: str, feedback_id: str) -> FeedbackRecord | None:
        return self._records.get((user_id, feedback_id))

    async def list_since(
        self, user_id: str, since: datetime
    ) -> list[FeedbackRecord]:
        threshold = _ensure_aware(since)
        rows = [
            record
            for (owner, _), record in self._records.items()
            if owner == user_id and _ensure_aware(record.created_at) >= threshold
        ]
        return sorted(rows, key=lambda item: _ensure_aware(item.created_at))


class MockPreferencesSource:
    """联调 Mock：何梓森的 /preferences 未就绪时的显式偏好来源。"""

    def __init__(self, initial: dict[str, Preferences] | None = None) -> None:
        self._preferences: dict[str, Preferences] = dict(initial or {})

    def set(
        self,
        user_id: str,
        liked_styles: Sequence[str] = (),
        avoid_tags: Sequence[str] = (),
    ) -> Preferences:
        preferences = Preferences(
            liked_styles=list(liked_styles), avoid_tags=list(avoid_tags)
        )
        self._preferences[user_id] = preferences
        return preferences

    async def get(self, user_id: str) -> Preferences:
        stored = self._preferences.get(user_id)
        return stored.model_copy(deep=True) if stored else Preferences()


# --------------------------------------------------------------------------- #
# 画像存储端口（真实接入点：PostgreSQL 事务）
# --------------------------------------------------------------------------- #
class ProfileStore(Protocol):
    """画像存储。

    commit 必须是一个事务：权重、版本、feedback_id 唯一键一起落库；
    版本不匹配时抛 StoreConflict（对应 SQL: UPDATE ... WHERE version = expected）。
    """

    async def load(self, user_id: str) -> MemoryState: ...

    async def commit(
        self, user_id: str, state: MemoryState, expected_version: int
    ) -> MemoryState: ...


class InMemoryProfileStore:
    """进程内事务语义：版本 CAS + 单锁提交，模拟 SQL 事务边界。"""

    def __init__(self) -> None:
        self._states: dict[str, MemoryState] = {}
        self._lock = asyncio.Lock()

    async def load(self, user_id: str) -> MemoryState:
        async with self._lock:
            stored = self._states.get(user_id)
            return stored.model_copy(deep=True) if stored else MemoryState()

    async def commit(
        self, user_id: str, state: MemoryState, expected_version: int
    ) -> MemoryState:
        async with self._lock:
            current = self._states.get(user_id) or MemoryState()
            if current.version != expected_version:
                raise StoreConflict(
                    f"画像版本已变化（期望 {expected_version}，实际 {current.version}）"
                )
            committed = state.model_copy(deep=True)
            self._states[user_id] = committed
            return committed.model_copy(deep=True)


# --------------------------------------------------------------------------- #
# 归一化与计权
# --------------------------------------------------------------------------- #
def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _normalize_tag_list(tags: Iterable[str] | str | None) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        tags = [tags]
    result: list[str] = []
    for tag in tags:
        value = str(tag).strip().lower()
        if value and value not in result:
            result.append(value)
    return result


def compute_learned_weights(
    feedbacks: Iterable[FeedbackRecord],
    *,
    avoid_tags: Iterable[str] = (),
) -> dict[str, float]:
    """近 30 天反馈按 style_tags 计权：like +1，dislike -1。

    每个反馈内同标签只计一次；显式禁忌（avoid_tags）优先：学习到的正权重归零，
    负权重保留（与用户明确禁忌方向一致）。
    """
    weights: dict[str, float] = {}
    for record in feedbacks:
        delta = 1.0 if record.action == "like" else -1.0
        for tag in _normalize_tag_list(record.style_tags):
            weights[tag] = weights.get(tag, 0.0) + delta

    avoided = set(_normalize_tag_list(avoid_tags))
    result: dict[str, float] = {}
    for tag, value in weights.items():
        if tag in avoided:
            # 显式禁忌优先：不允许学习出正权重（0.0 保留，便于排查）
            result[tag] = 0.0 if value > 0 else value
        elif value != 0:
            result[tag] = value
    return result


def _apply_explicit_priority(
    learned: dict[str, float], avoid_tags: Iterable[str]
) -> dict[str, float]:
    """读取时再次保证“明确禁忌优先”（用户可能刚改过 /preferences）。"""
    avoided = set(_normalize_tag_list(avoid_tags))
    result: dict[str, float] = {}
    for tag, value in learned.items():
        if tag in avoided and value > 0:
            result[tag] = 0.0
        else:
            result[tag] = value
    return result


class _UserLocks:
    """按 user_id 串行（03：按用户串行处理）。"""

    def __init__(self) -> None:
        self._guard = asyncio.Lock()
        self._locks: dict[str, asyncio.Lock] = {}

    async def acquire(self, user_id: str) -> asyncio.Lock:
        async with self._guard:
            lock = self._locks.get(user_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[user_id] = lock
            return lock


# --------------------------------------------------------------------------- #
# 业务服务
# --------------------------------------------------------------------------- #
class MemoryService:
    """偏好记忆：画像查询 + 反馈驱动的画像更新。"""

    def __init__(
        self,
        *,
        store: ProfileStore | None = None,
        feedback: FeedbackSource | None = None,
        preferences: PreferencesSource | None = None,
        inspirations: InspirationsService | None = None,
        now: Callable[[], datetime] = utcnow,
        window_days: int = _WINDOW_DAYS,
    ) -> None:
        self._store = store or InMemoryProfileStore()
        self._feedback = feedback or MockFeedbackSource()
        self._preferences = preferences or MockPreferencesSource()
        self._inspirations = inspirations
        self._now = now
        self._window = timedelta(days=window_days)
        self._locks = _UserLocks()

    # ---- 读：memory.get(user_id) ------------------------------------------ #
    async def get(self, user_id: str) -> Any:
        user = _require_user(user_id)
        explicit = await self._preferences.get(user)
        state = await self._store.load(user)
        urls = await self._recent_inspiration_urls(user)

        profile = Profile(
            explicit=explicit,
            learned_weights=_apply_explicit_priority(
                state.learned_weights, explicit.avoid_tags
            ),
            inspiration_image_urls=urls,
            version=state.version,
        )
        return _as_contract("Profile", profile)

    async def _recent_inspiration_urls(self, user_id: str) -> list[str]:
        service = self._inspiration_service()
        if service is None:
            return []
        try:
            return await service.recent_image_urls(user_id, 3)
        except Exception:  # pragma: no cover - 喜好图读取失败不阻塞画像
            return []

    def _inspiration_service(self) -> InspirationsService | None:
        if self._inspirations is not None:
            return self._inspirations
        try:
            return get_inspiration_service()
        except Exception:  # pragma: no cover
            return None

    # ---- 写：memory.update(user_id, feedback_id) --------------------------- #
    async def update(self, user_id: str, feedback_id: str) -> dict[str, int]:
        user = _require_user(user_id)
        if not isinstance(feedback_id, str) or not feedback_id.strip():
            raise MemoryError("feedback_id 不能为空", code="INVALID_FIELD")
        feedback_key = feedback_id.strip()

        # 03：更新须等 Feedback 入库（Job 失败码可重试或等待反馈事务提交）
        record = await self._feedback.get(user, feedback_key)
        if record is None:
            raise FeedbackNotReady(f"反馈 {feedback_key} 尚未入库，暂不能更新画像")
        if record.user_id != user:
            raise FeedbackNotReady(
                f"反馈 {feedback_key} 不属于当前用户", code="FEEDBACK_NOT_FOUND"
            )

        lock = await self._locks.acquire(user)
        async with lock:
            return await self._apply_update(user, record)

    async def _apply_update(
        self, user_id: str, record: FeedbackRecord
    ) -> dict[str, int]:
        for _ in range(_MAX_UPDATE_ATTEMPTS):
            state = await self._store.load(user_id)
            if record.feedback_id in state.applied_feedback_ids:
                # feedback_id 去重：重放/重复点击不改权重、不改版本
                return {"profile_version": state.version}

            explicit = await self._preferences.get(user_id)
            window = await self._window_feedbacks(user_id, record)
            weights = compute_learned_weights(
                window, avoid_tags=explicit.avoid_tags
            )
            next_state = MemoryState(
                version=state.version + 1,
                learned_weights=weights,
                applied_feedback_ids=[*state.applied_feedback_ids, record.feedback_id],
            )
            try:
                committed = await self._store.commit(
                    user_id, next_state, expected_version=state.version
                )
            except StoreConflict:
                continue
            return {"profile_version": committed.version}

        raise StoreConflict("画像更新并发冲突，请重试")

    async def _window_feedbacks(
        self, user_id: str, current: FeedbackRecord
    ) -> list[FeedbackRecord]:
        since = _ensure_aware(self._now()) - self._window
        rows = await self._feedback.list_since(user_id, since)
        seen: set[str] = {record.feedback_id for record in rows}
        if current.feedback_id not in seen:
            # 反馈源可能晚于事务可见，保证本次反馈一定参与计权
            rows.append(current)
        return [
            record
            for record in rows
            if record.user_id == user_id
            and _ensure_aware(record.created_at) >= since
        ]


def _require_user(user_id: str) -> str:
    if not isinstance(user_id, str) or not user_id.strip():
        raise MemoryError("user_id 不能为空", code="INVALID_FIELD")
    return user_id.strip()


# --------------------------------------------------------------------------- #
# 默认单例
# --------------------------------------------------------------------------- #
_default_service: MemoryService | None = None


def get_memory_service() -> MemoryService:
    global _default_service
    if _default_service is None:
        _default_service = MemoryService()
    return _default_service


def reset_default_service() -> None:
    global _default_service
    _default_service = None
