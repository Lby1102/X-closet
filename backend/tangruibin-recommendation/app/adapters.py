"""Integration seams and local demo implementations for teammate-owned services."""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Protocol

from app.schemas import Candidate, Garment, Preferences, Profile, RecommendationDraft, ReviewResult, Scene, Tags, Trend, Weather


class WeatherPort(Protocol):
    async def get(self, city: str, day: date) -> Weather: ...


class WardrobePort(Protocol):
    async def list_confirmed(self, user_id: str) -> list[Garment]: ...
    async def search(self, user_id: str, image_id: str, limit: int = 10) -> list[Candidate]: ...


class ProfilePort(Protocol):
    async def get(self, user_id: str) -> Profile: ...


class TrendPort(Protocol):
    async def search(self, scene: str, season: str, limit: int = 3,
                     user_id: str | None = None) -> list[Trend]: ...

    async def search_with_warnings(self, scene: str, season: str, limit: int = 3,
                                   *, user_id: str) -> tuple[list[Trend], list[str]]: ...


class StylistPort(Protocol):
    async def propose(self, context: dict, previous_issues: list[dict] | None = None) -> RecommendationDraft: ...


class ReviewPort(Protocol):
    async def check(self, user_id: str, draft: RecommendationDraft, weather: Weather,
                    profile: Profile) -> ReviewResult: ...


class DemoWeather:
    async def get(self, city: str, day: date) -> Weather:
        return Weather(apparent_temp_c=18, rain_probability=15, wind_kph=8,
                       fetched_at=datetime.now(timezone.utc))


class DemoWardrobe:
    def __init__(self):
        self.items = [
            Garment(garment_id="g_top_white", image_id="img_top_white", image_url="/demo/white-shirt.jpg",
                    tags=Tags(category="top", color="白色", warmth="light", style_tags=["简约", "通勤"])),
            Garment(garment_id="g_top_blue", image_id="img_top_blue", image_url="/demo/blue-knit.jpg",
                    tags=Tags(category="top", color="蓝色", warmth="medium", style_tags=["休闲", "复古"])),
            Garment(garment_id="g_bottom_black", image_id="img_bottom_black", image_url="/demo/black-trousers.jpg",
                    tags=Tags(category="bottom", color="黑色", warmth="medium", style_tags=["简约", "通勤"])),
            Garment(garment_id="g_bottom_jeans", image_id="img_bottom_jeans", image_url="/demo/jeans.jpg",
                    tags=Tags(category="bottom", color="蓝色", warmth="medium", style_tags=["休闲"])),
            Garment(garment_id="g_outer_beige", image_id="img_outer_beige", image_url="/demo/beige-coat.jpg",
                    tags=Tags(category="outerwear", color="米色", warmth="warm", style_tags=["优雅", "通勤"])),
            Garment(garment_id="g_shoes_white", image_id="img_shoes_white", image_url="/demo/sneakers.jpg",
                    tags=Tags(category="shoes", color="白色", warmth="light", style_tags=["休闲"])),
        ]

    async def list_confirmed(self, user_id: str) -> list[Garment]:
        return list(self.items)

    async def search(self, user_id: str, image_id: str, limit: int = 10) -> list[Candidate]:
        return [Candidate(garment_id=g.garment_id, image_url=g.image_url, score=0.9 - i * 0.05)
                for i, g in enumerate(self.items[:limit])]


class DemoProfile:
    async def get(self, user_id: str) -> Profile:
        return Profile(explicit=Preferences(liked_styles=["简约", "通勤"], avoid_tags=[]),
                       learned_weights={"简约": 0.2}, inspiration_image_urls=[], version=1)


class DemoTrends:
    async def search(self, scene: str, season: str, limit: int = 3,
                     user_id: str | None = None) -> list[Trend]:
        return [Trend(summary=f"{season} {scene}：简洁的中性色搭配", tags=["简约", "中性色"],
                      source_url="https://example.invalid/demo-trend")][:limit]

    async def search_with_warnings(self, scene: str, season: str, limit: int = 3,
                                   *, user_id: str) -> tuple[list[Trend], list[str]]:
        """Normalize trend results and warnings for the recommendation workflow."""
        try:
            items = await self.search(scene, season, limit=limit, user_id=user_id)
            return items, []
        except Exception:
            return [], ["TREND_UNAVAILABLE：趋势查询失败，本次使用衣橱和用户偏好继续推荐。"]


class RuleBasedStylist:
    """Local deterministic fallback; replace with the team's multimodal model adapter."""
    async def propose(self, context: dict, previous_issues: list[dict] | None = None) -> RecommendationDraft:
        items: list[Garment] = context["garments"]
        excluded = {issue.get("garment_id") for issue in (previous_issues or [])}
        grouped: dict[str, list[Garment]] = {}
        for item in items:
            if item.garment_id not in excluded:
                grouped.setdefault(item.tags.category, []).append(item)
        tops = grouped.get("top", [])
        bottoms = grouped.get("bottom", [])
        if not tops or not bottoms:
            # A one-piece dress may form a complete outfit by itself.
            dresses = grouped.get("dress", [])
            if dresses:
                return RecommendationDraft(outfits=[{"garment_ids": [dresses[0].garment_id],
                    "reason": "以衣橱中的连衣裙作为完整搭配，兼顾当前场景与天气。"}])
            return RecommendationDraft(outfits=[])
        liked = set(context["profile"].explicit.liked_styles)
        ranked = sorted(tops, key=lambda g: sum(tag in liked for tag in g.tags.style_tags), reverse=True)
        pairings = [(ranked[0], bottoms[0]), (ranked[-1], bottoms[-1])]
        output = []
        for top, bottom in pairings:
            ids = [top.garment_id, bottom.garment_id]
            coat = next(iter(grouped.get("outerwear", [])), None)
            if coat and context["weather"].apparent_temp_c < 14:
                ids.append(coat.garment_id)
            if len(output) and ids == output[0]["garment_ids"]:
                continue
            style = "、".join(top.tags.style_tags[:2] + bottom.tags.style_tags[:1]) or "协调"
            output.append({"garment_ids": ids, "reason": f"{style}风格组合；结合{context['weather'].apparent_temp_c:g}℃天气选择衣物。"})
        return RecommendationDraft(outfits=output[:2])


class LocalReview:
    """Demo checker using the same four-argument contract as the review module."""
    def __init__(self, wardrobe: WardrobePort):
        self.wardrobe = wardrobe

    async def check(self, user_id: str, draft: RecommendationDraft, weather: Weather,
                    profile: Profile) -> ReviewResult:
        confirmed = await self.wardrobe.list_confirmed(user_id)
        confirmed_ids = {garment.garment_id for garment in confirmed}
        issues = []
        for outfit in draft.outfits:
            for garment_id in outfit.garment_ids:
                if garment_id not in confirmed_ids:
                    issues.append({"garment_id": garment_id, "reason": "当前用户的已确认衣橱中没有这件单品，请从该衣橱选择替代单品。"})
        return ReviewResult(passed=not issues, issues=issues)


class WardrobeServiceAdapter:
    """Translate the team's wardrobe.list / closet.search APIs to recommendation ports."""

    def __init__(self, wardrobe, closet):
        self.wardrobe = wardrobe
        self.closet = closet

    async def list_confirmed(self, user_id: str) -> list[Garment]:
        items = await self.wardrobe.list(user_id)
        return [Garment.model_validate(item.model_dump()) for item in items]

    async def search(self, user_id: str, image_id: str, limit: int = 10) -> list[Candidate]:
        items = await self.closet.search(user_id, image_id, limit=limit)
        return [Candidate.model_validate(item.model_dump()) for item in items]


class MemoryProfileAdapter:
    """Translate memory.get(user_id) to the recommendation Profile port."""

    def __init__(self, memory):
        self.memory = memory

    async def get(self, user_id: str) -> Profile:
        profile = await self.memory.get(user_id)
        return Profile.model_validate(profile.model_dump())


class TrendsServiceAdapter:
    """Normalize the team's trends search result and warning tuple."""

    def __init__(self, trends):
        self.trends = trends

    async def search(self, scene: str, season: str, limit: int = 3,
                     user_id: str | None = None) -> list[Trend]:
        items = await self.trends.search(scene, season, limit=limit, user_id=user_id)
        return [Trend.model_validate(item.model_dump()) for item in items]

    async def search_with_warnings(self, scene: str, season: str, limit: int = 3,
                                   *, user_id: str) -> tuple[list[Trend], list[str]]:
        items, warnings = await self.trends.search_with_warnings(
            scene, season, limit=limit, user_id=user_id
        )
        return [Trend.model_validate(item.model_dump()) for item in items], list(warnings)


def season_for(day: date) -> str:
    return {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}[day.month]
