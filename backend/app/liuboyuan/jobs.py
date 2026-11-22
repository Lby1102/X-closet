"""本模块的后台任务等价物（03「任务与数据提交」）。

02 / 03 要求：
    * Job 状态依次为 queued / running / succeeded / failed；
    * 创建任务、保存幂等键与输入在一个事务完成；同一用户同一路径同一键且输入
      相同复用原任务，输入不同返回 409；
    * 结果与成功状态一起提交；查询失败的 Job 仍返回 200（读 Job.error）；
    * 执行超时 300 秒。

本文件是任务框架的最小可运行实现：进程内 asyncio 执行 + 内存幂等表，
接口与郭子骏的 Celery/Redis 框架一致（submit / register / get），
联调时把 :class:`JobRunner` 换成共享框架即可，三个处理函数不用改。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping

from pydantic import BaseModel, ConfigDict, Field

from .common import ModuleError, brief

__all__ = [
    "JOB_STATUSES",
    "JOB_TIMEOUT_SECONDS",
    "JOB_TYPES",
    "Handler",
    "IdempotencyConflict",
    "InvalidJobRequest",
    "Job",
    "JobContext",
    "JobErrorItem",
    "JobFailure",
    "JobNotFound",
    "JobRunner",
    "JobsError",
]

#: 02 的 Job.type 取值（本组目前注册前三个，其余由别的窗口接线）
JOB_TYPES: tuple[str, ...] = (
    "garment_analysis",
    "inspiration_collection",
    "recommendation",
    "tryon",
    "memory_update",
    "plan",
)
JOB_STATUSES: tuple[str, ...] = ("queued", "running", "succeeded", "failed")
JOB_TIMEOUT_SECONDS = 300.0


# --------------------------------------------------------------------------- #
# 错误
# --------------------------------------------------------------------------- #
class JobsError(ModuleError):
    code = "INVALID_INPUT"
    status_code = 422


class InvalidJobRequest(JobsError):
    code = "INVALID_INPUT"
    status_code = 422


class JobNotFound(JobsError):
    """02：404 表示资源在当前用户下不存在（含他人的 Job）。"""

    code = "JOB_NOT_FOUND"
    status_code = 404


class IdempotencyConflict(JobsError):
    """02：同一幂等键对应不同输入。"""

    code = "IDEMPOTENCY_CONFLICT"
    status_code = 409


class JobFailure(Exception):
    """处理函数主动失败：code / message / retryable 直接写进 Job.error。"""

    def __init__(
        self, code: str, message: str, *, retryable: bool = False
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


# --------------------------------------------------------------------------- #
# 02 的 Job
# --------------------------------------------------------------------------- #
class JobErrorItem(BaseModel):
    """02：error 包含 code、message、retryable。"""

    model_config = ConfigDict(extra="ignore")

    code: str
    message: str
    retryable: bool = False


class Job(BaseModel):
    """02 的 Job：stage 表示执行步骤；成功 result 有值且 error=null，失败相反。"""

    model_config = ConfigDict(extra="ignore")

    job_id: str
    type: str
    status: str = "queued"
    stage: str = "queued"
    result: dict[str, Any] | None = None
    error: JobErrorItem | None = None
    warnings: list[str] = Field(default_factory=list)


Handler = Callable[["JobContext"], Awaitable[Mapping[str, Any] | None]]


# --------------------------------------------------------------------------- #
# 任务上下文
# --------------------------------------------------------------------------- #
@dataclass
class JobContext:
    """处理函数看到的输入与副作用入口（stage / warnings 会被持久化到 Job）。"""

    user_id: str
    params: Mapping[str, Any]
    job_id: str
    job_type: str
    runner: "JobRunner"
    deps: Any = None
    warnings: list[str] = field(default_factory=list)

    def set_stage(self, stage: str) -> None:
        self.runner._set_stage(self.job_id, stage)

    def add_warning(self, message: str) -> None:
        text = str(message).strip()
        if text:
            self.warnings.append(text)

    def param(self, name: str, default: Any = None) -> Any:
        return self.params.get(name, default)


# --------------------------------------------------------------------------- #
# 任务运行器
# --------------------------------------------------------------------------- #
class JobRunner:
    """进程内任务框架：提交即排队，异步执行，幂等键表按 (user, path, key) 去重。"""

    def __init__(
        self,
        *,
        deps: Any = None,
        timeout: float = JOB_TIMEOUT_SECONDS,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._deps = deps
        self._timeout = float(timeout)
        self._id_factory = id_factory or (lambda: f"job_{uuid.uuid4().hex[:12]}")
        self._handlers: dict[str, Handler] = {}
        self._jobs: dict[str, Job] = {}
        self._owners: dict[str, str] = {}
        self._idempotency: dict[tuple[str, str, str], tuple[str, str]] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    # ---- 注册 ------------------------------------------------------------- #
    def register(self, job_type: str, handler: Handler) -> None:
        if job_type not in JOB_TYPES:
            raise InvalidJobRequest(f"未知任务类型：{job_type}")
        self._handlers[job_type] = handler

    def handler_types(self) -> tuple[str, ...]:
        return tuple(sorted(self._handlers))

    # ---- 提交 ------------------------------------------------------------- #
    async def submit(
        self,
        *,
        user_id: str,
        path: str,
        job_type: str,
        params: Mapping[str, Any] | None,
        idempotency_key: str | None,
    ) -> Job:
        user = _require_text(user_id, "user_id")
        route = _require_text(path, "path")
        key = (idempotency_key or "").strip()
        if not key:
            # 02：提交任务时携带 Idempotency-Key
            raise InvalidJobRequest("缺少 Idempotency-Key")
        handler = self._handlers.get(job_type)
        if handler is None:
            raise InvalidJobRequest(f"未注册的任务类型：{job_type}")

        payload = dict(params or {})
        digest = _digest(payload)
        idem_key = (user, route, key)

        async with self._lock:
            existing = self._idempotency.get(idem_key)
            if existing is not None:
                job_id, existing_digest = existing
                if existing_digest != digest:
                    raise IdempotencyConflict(
                        "同一 Idempotency-Key 对应不同输入，拒绝复用原任务"
                    )
                return self._jobs[job_id].model_copy(deep=True)

            job = Job(job_id=self._id_factory(), type=job_type)
            self._jobs[job.job_id] = job
            self._owners[job.job_id] = user
            self._idempotency[idem_key] = (job.job_id, digest)
            ctx = JobContext(
                user_id=user,
                params=payload,
                job_id=job.job_id,
                job_type=job_type,
                runner=self,
                deps=self._deps,
            )
            self._tasks[job.job_id] = asyncio.create_task(
                self._run(ctx, handler), name=f"job:{job.job_id}"
            )
            return job.model_copy(deep=True)

    # ---- 执行 ------------------------------------------------------------- #
    async def _run(self, ctx: JobContext, handler: Handler) -> None:
        job = self._jobs[ctx.job_id]
        job.status = "running"
        job.stage = "starting"
        try:
            result = await asyncio.wait_for(handler(ctx), timeout=self._timeout)
        except TimeoutError:
            job.status = "failed"
            job.stage = "timeout"
            job.result = None
            job.error = JobErrorItem(
                code="TASK_TIMEOUT",
                message=f"任务执行超过 {int(self._timeout)} 秒",
                retryable=True,
            )
        except JobFailure as exc:
            job.status = "failed"
            job.stage = "failed"
            job.result = None
            job.error = JobErrorItem(
                code=exc.code, message=exc.message, retryable=exc.retryable
            )
        except asyncio.CancelledError:  # pragma: no cover - 停机时取消
            job.status = "failed"
            job.stage = "cancelled"
            job.error = JobErrorItem(
                code="TASK_CANCELLED", message="任务被取消", retryable=True
            )
            raise
        except Exception as exc:  # 未预期异常：不把堆栈抛给调用方
            job.status = "failed"
            job.stage = "failed"
            job.result = None
            job.error = JobErrorItem(
                code="TASK_FAILED", message=brief(exc), retryable=False
            )
        else:
            job.result = dict(result or {})
            job.status = "succeeded"
            job.stage = "finished"
            job.error = None
        finally:
            job.warnings = list(ctx.warnings)

    def _set_stage(self, job_id: str, stage: str) -> None:
        job = self._jobs.get(job_id)
        if job is not None and job.status not in ("succeeded", "failed"):
            job.stage = str(stage)

    # ---- 查询 ------------------------------------------------------------- #
    async def get(self, user_id: str, job_id: str) -> Job:
        user = _require_text(user_id, "user_id")
        key = _require_text(job_id, "job_id")
        job = self._jobs.get(key)
        if job is None or self._owners.get(key) != user:
            raise JobNotFound(f"任务不存在：{key}")
        return job.model_copy(deep=True)

    async def wait(
        self, user_id: str, job_id: str, timeout: float | None = None
    ) -> Job:
        """等任务结束（测试与同步调用方用）；超时返回当前快照，不抛错。"""
        job = await self.get(user_id, job_id)
        task = self._tasks.get(job.job_id)
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=self._timeout if timeout is None else timeout,
                )
            except TimeoutError:
                pass
        return await self.get(user_id, job_id)

    async def wait_all(self, timeout: float = 10.0) -> None:
        """等待所有已提交任务落定（测试收尾，避免任务泄漏到下一个用例）。"""
        tasks = [task for task in self._tasks.values() if not task.done()]
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)


def _digest(params: Mapping[str, Any]) -> str:
    raw = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidJobRequest(f"{field} 不能为空")
    return value.strip()
