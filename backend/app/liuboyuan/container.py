"""服务装配（容器）与三个任务处理函数。

03 的交接点：
    识别与衣橱   刘博远 -> 徐宇飞、唐睿彬、郭子骏   识别 {image_id,tags}；入库 Garment；查询 Garment[]
    趋势 RAG     刘博远 -> 唐睿彬、徐宇飞           trends.search(scene, season, limit=3)
    偏好记忆     刘博远 -> 唐睿彬、何梓森           memory.get / memory.update
    以图搜衣     刘博远 -> 唐睿彬                   closet.search(user_id, image_id, limit=10)

本文件把窗口 B 的喜好图片/趋势/记忆与窗口 A 的识别/衣橱/以图搜衣装成一套服务，
并把 03 的三个任务处理函数注册进任务框架：

    garment_analysis      -> recognition.analyze   result={image_id, tags}
    inspiration_collection-> trends.collect        result={added_count}
    memory_update         -> memory.update         result={profile_version}

真实接入点（郭子骏）：把 :class:`JobRunner` 换成交付的 Celery 框架，
``register`` 的三个 handler 原样搬过去；Http 路由用 ``api.Api`` 的方法挂到 APIRouter。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .api import Api, Users
from .closet_search import ClosetSearchService
from .images import InMemoryImageStore
from .inspirations import InspirationsService
from .jobs import JobContext, JobFailure, JobRunner
from .memory import MemoryService
from .recognition import GarmentRecognitionService, HeuristicVisionAnalyzer
from .trends import TrendsService
from .wardrobe import (
    HashImageEmbedder,
    InMemoryGarmentVectorStore,
    WardrobeService,
)

__all__ = ["DEFAULT_USERS", "Services", "build_services", "drain"]

DEFAULT_USERS: tuple[str, ...] = ("u_1", "u_demo")


@dataclass
class Services:
    """一次装配的全部服务；测试与路由都从这里取，避免各自新建实例。"""

    users: Users
    images: InMemoryImageStore
    embedder: Any
    vectors: Any
    vision: Any
    jobs: JobRunner
    recognition: GarmentRecognitionService
    wardrobe: WardrobeService
    closet: ClosetSearchService
    inspirations: InspirationsService
    trends: TrendsService
    memory: MemoryService
    api: Api


def build_services(
    *,
    users: Iterable[str] = DEFAULT_USERS,
    vision: Any = None,
    embedder: Any = None,
    vectors: Any = None,
    image_store: Any = None,
    fetcher: Any = None,
    extractor: Any = None,
    feedback: Any = None,
    preferences: Any = None,
    trend_sources: Iterable[str] | None = None,
    settings: Any = None,
    now: Any = None,
) -> Services:
    """装配全 Mock 可跑的服务；真实实现按参数替换（都是可选的端口注入）。"""
    user_directory = Users(users)
    images = image_store or InMemoryImageStore()
    image_embedder = embedder or HashImageEmbedder()
    vector_store = vectors or InMemoryGarmentVectorStore()
    vision_analyzer = vision or HeuristicVisionAnalyzer()

    inspirations = InspirationsService()
    trends = TrendsService(
        inspirations=inspirations,
        fetcher=fetcher,
        extractor=extractor,
        sources=list(trend_sources) if trend_sources is not None else None,
        settings=settings,
    )
    memory_kwargs: dict[str, Any] = {
        "feedback": feedback,
        "preferences": preferences,
        "inspirations": inspirations,
    }
    wardrobe_kwargs: dict[str, Any] = {
        "images": images,
        "embedder": image_embedder,
        "vectors": vector_store,
    }
    if now is not None:  # 测试/演示注入时钟（30 天窗口、排序）
        memory_kwargs["now"] = now
        wardrobe_kwargs["now"] = now
    memory = MemoryService(**memory_kwargs)
    wardrobe = WardrobeService(**wardrobe_kwargs)
    closet = ClosetSearchService(wardrobe=wardrobe, images=images)
    recognition = GarmentRecognitionService(images=images, vision=vision_analyzer)
    jobs = JobRunner()

    api = Api(
        users=user_directory,
        images=images,
        recognition=recognition,
        wardrobe=wardrobe,
        inspirations=inspirations,
        trends=trends,
        jobs=jobs,
    )

    services = Services(
        users=user_directory,
        images=images,
        embedder=image_embedder,
        vectors=vector_store,
        vision=vision_analyzer,
        jobs=jobs,
        recognition=recognition,
        wardrobe=wardrobe,
        closet=closet,
        inspirations=inspirations,
        trends=trends,
        memory=memory,
        api=api,
    )
    _register_handlers(services)
    return services


def _job_failure_from(exc: BaseException) -> JobFailure | None:
    """把领域错误转成 Job.error。

    按属性识别（code / message / retryable），因此本模块的 ``ModuleError``、
    窗口 B 的 ``MemoryError`` / ``TrendError`` 与将来共享框架的错误类型都能复用，
    不必逐个 import；不像领域错误的异常返回 None，由框架记 TASK_FAILED。
    """
    code = getattr(exc, "code", None)
    message = getattr(exc, "message", None)
    if not isinstance(code, str) or not isinstance(message, str):
        return None
    return JobFailure(code, message, retryable=bool(getattr(exc, "retryable", False)))


def _register_handlers(services: Services) -> None:
    """注册 03 的三个任务处理函数；领域错误统一转成 Job.error。"""

    async def garment_analysis(ctx: JobContext) -> dict[str, Any]:
        ctx.set_stage("calling_vision_api")
        try:
            result = await services.recognition.analyze(
                ctx.user_id, str(ctx.param("image_id") or "")
            )
        except Exception as exc:
            failure = _job_failure_from(exc)
            if failure is None:
                raise
            raise failure from exc
        ctx.set_stage("saving_result")
        return result

    async def inspiration_collection(ctx: JobContext) -> dict[str, Any]:
        ctx.set_stage("collecting_sources")
        try:
            collected = await services.trends.collect(
                ctx.user_id,
                str(ctx.param("style") or ""),
                ctx.param("season"),
            )
        except Exception as exc:
            failure = _job_failure_from(exc)
            if failure is None:
                raise
            raise failure from exc
        for warning in collected.warnings:
            ctx.add_warning(warning)
        ctx.set_stage("saving_result")
        return collected.as_job_result()

    async def memory_update(ctx: JobContext) -> dict[str, Any]:
        ctx.set_stage("loading_feedback")
        try:
            result = await services.memory.update(
                ctx.user_id, str(ctx.param("feedback_id") or "")
            )
        except Exception as exc:
            failure = _job_failure_from(exc)
            if failure is None:
                raise
            raise failure from exc
        ctx.set_stage("saving_profile")
        return dict(result)

    services.jobs.register("garment_analysis", garment_analysis)
    services.jobs.register("inspiration_collection", inspiration_collection)
    services.jobs.register("memory_update", memory_update)


async def drain(services: Services, timeout: float = 10.0) -> None:
    """等本进程内所有任务落定（测试收尾 / 优雅停机用）。"""
    await services.jobs.wait_all(timeout=timeout)
