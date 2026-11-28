from __future__ import annotations

import asyncio
import re
from datetime import date

from app.adapters import (DemoProfile, DemoTrends, DemoWardrobe, DemoWeather, LocalReview,
                          RuleBasedStylist, season_for)
from app.context import compress
from app.schemas import (ChatMessage, Garment, Job, JobStatus, MessageRole, Outfit, Profile, Recommendation,
                         RecommendationDraft, RecommendationRequest, ReviewIssue, Weather)


class WorkflowError(Exception):
    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


class RecommendationService:
    """Coordinates the assigned recommendation workflow; ports can be replaced at startup."""
    def __init__(self, weather=None, wardrobe=None, profile=None, trends=None, stylist=None, review=None):
        self.weather = weather or DemoWeather()
        self.wardrobe = wardrobe or DemoWardrobe()
        self.profile = profile or DemoProfile()
        self.trends = trends or DemoTrends()
        self.stylist = stylist or RuleBasedStylist()
        self.review = review or LocalReview(self.wardrobe)
        self.recommendations: dict[tuple[str, str], Recommendation] = {}
        self.weather_snapshots: dict[tuple[str, str], Weather] = {}

    async def generate(self, user_id: str, request: RecommendationRequest) -> Recommendation:
        # Required reads run concurrently. A trend outage is the one best-effort dependency.
        weather_task = self.weather.get(request.city, request.date)
        garments_task = self.wardrobe.list_confirmed(user_id)
        profile_task = self.profile.get(user_id)
        trend_task = self.trends.search_with_warnings(
            request.scene.value, season_for(request.date), limit=3, user_id=user_id
        )
        messages = list(request.messages)
        if request.request:
            messages.append(ChatMessage(role=MessageRole.user, text=request.request))
        context_task = asyncio.to_thread(compress, messages, request.scene.value)
        results = await asyncio.gather(weather_task, garments_task, profile_task, trend_task,
                                       context_task, return_exceptions=True)
        weather, garments, profile, trend_result, compressed = results
        if isinstance(weather, Exception):
            raise WorkflowError("WEATHER_UNAVAILABLE", "无法获取指定日期的天气，暂时不能生成推荐。", True)
        if isinstance(garments, Exception):
            raise WorkflowError("UPSTREAM_FAILED", "读取衣橱失败。", True)
        if isinstance(profile, Exception):
            raise WorkflowError("UPSTREAM_FAILED", "读取用户偏好失败。", True)
        if isinstance(compressed, Exception):
            raise WorkflowError("UPSTREAM_FAILED", "压缩对话上下文失败。", True)

        warnings: list[str] = []
        if isinstance(trend_result, Exception):
            trends = []
            warnings.append("TREND_UNAVAILABLE：趋势查询失败，本次使用衣橱和用户偏好继续推荐。")
        else:
            trends, trend_warnings = trend_result
            warnings.extend(trend_warnings)
        if not garments:
            raise WorkflowError("WARDROBE_EMPTY", "衣橱中没有已确认入库的单品。")

        eligible = self._filter_garments(garments, weather, profile, compressed.constraints)
        if not eligible:
            raise WorkflowError("NO_VALID_OUTFIT", "天气和明确禁忌过滤后，没有可用于搭配的单品。")
        if request.reference_image_id:
            try:
                similar = await self.wardrobe.search(user_id, request.reference_image_id, limit=10)
                by_id = {item.garment_id: item for item in eligible}
                ordered = [by_id[c.garment_id] for c in sorted(similar, key=lambda c: c.score, reverse=True)
                           if c.garment_id in by_id]
                # Preserve visual matches while supplementing a sparse intersection from eligible items.
                if len(ordered) < 4:
                    seen = {g.garment_id for g in ordered}
                    ordered.extend(g for g in eligible if g.garment_id not in seen)
                eligible = ordered
            except Exception:
                warnings.append("REFERENCE_SEARCH_UNAVAILABLE：参考图相似检索失败，已使用其他合规衣物。")

        context = {
            "user_id": user_id, "request": request, "weather": weather, "garments": eligible,
            "profile": profile, "trends": trends, "compressed": compressed,
            "reference_image_id": request.reference_image_id,
            "inspiration_image_urls": profile.inspiration_image_urls[:3],
        }
        previous_issues: list[dict] = []
        accepted: RecommendationDraft | None = None
        for attempt in range(2):
            try:
                draft = await self.stylist.propose(context, previous_issues or None)
            except Exception as exc:
                raise WorkflowError("UPSTREAM_FAILED", "搭配生成服务失败。", True) from exc
            if not draft.outfits:
                raise WorkflowError("NO_VALID_OUTFIT", "当前衣橱无法组成一套完整搭配。")
            checked = await self.review.check(user_id, draft, weather, profile)
            if checked.passed:
                accepted = draft
                break
            previous_issues = [issue.model_dump() for issue in checked.issues]
        if accepted is None:
            raise WorkflowError("NO_VALID_OUTFIT", "搭配复核未通过，按反馈修订一次后仍不符合要求。")

        if len(accepted.outfits) == 1:
            warnings.append("ONLY_ONE_OUTFIT：当前条件下仅有一套搭配通过复核。")
        outfits = [Outfit(garment_ids=o.garment_ids, reason=o.reason) for o in accepted.outfits]
        recommendation = Recommendation(outfits=outfits, warnings=warnings)
        # In a production integration, save this object and child outfits in one DB transaction.
        self.recommendations[(user_id, recommendation.recommendation_id)] = recommendation
        self.weather_snapshots[(user_id, recommendation.recommendation_id)] = weather
        return recommendation

    @staticmethod
    def _filter_garments(garments: list[Garment], weather: Weather, profile: Profile,
                         constraints: list[str] | None = None) -> list[Garment]:
        avoids = [tag.casefold() for tag in profile.explicit.avoid_tags]
        # Pull short, explicit avoid terms from the current text. The complete sentence remains
        # in compressed.constraints for the model/reviewer to interpret as well.
        for constraint in constraints or []:
            match = re.search(r"(?:不要(?:推荐|穿)?|别(?:推荐|穿)?|避免|不穿|不能穿|过敏|不喜欢)\s*([\u4e00-\u9fffA-Za-z0-9]{1,12})", constraint)
            if match:
                avoids.append(match.group(1).casefold())
        result = []
        for garment in garments:
            tags = [garment.tags.category, garment.tags.color, garment.tags.pattern or "",
                    garment.tags.fit or "", *garment.tags.style_tags]
            text = " ".join(tags).casefold()
            if any(tag and tag in text for tag in avoids):
                continue
            warmth = (garment.tags.warmth or "").casefold()
            if weather.apparent_temp_c >= 28 and warmth == "warm":
                continue
            if weather.apparent_temp_c < 8 and warmth == "light" and garment.tags.category in {"top", "outerwear"}:
                continue
            result.append(garment)
        return result


class InMemoryJobStore:
    """Development adapter. Replace with the shared PostgreSQL/Redis task framework."""
    def __init__(self):
        self.jobs: dict[tuple[str, str], Job] = {}
        self.idempotency: dict[tuple[str, str, str], tuple[str, str]] = {}
        self.tasks: set[asyncio.Task] = set()

    def create(self, user_id: str, idempotency_key: str, input_fingerprint: str, request_id: str) -> tuple[Job, bool]:
        key = (user_id, "/api/v1/recommendations", idempotency_key)
        existing = self.idempotency.get(key)
        if existing:
            old_fingerprint, job_id = existing
            if old_fingerprint != input_fingerprint:
                raise WorkflowError("IDEMPOTENCY_CONFLICT", "相同 Idempotency-Key 已用于不同请求内容。")
            return self.jobs[(user_id, job_id)], False
        job = Job(request_id=request_id)
        self.jobs[(user_id, job.job_id)] = job
        self.idempotency[key] = (input_fingerprint, job.job_id)
        return job, True

    def get(self, user_id: str, job_id: str) -> Job | None:
        return self.jobs.get((user_id, job_id))

    def add_task(self, task: asyncio.Task) -> None:
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

