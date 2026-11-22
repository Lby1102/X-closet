"""趋势采集与检索验证（刘博远）。

覆盖 02/03/04 的验收点：
    - 基于 TREND_SOURCES 的采集（环境变量 / 配置对象）
    - 抓取 -> 抽取 -> 风格+季节筛选 -> 去重 -> added_count
    - 同一来源重复采集去重（added_count=0）
    - 采集成功刷新喜好列表（inspirations 可读、倒序）
    - trends.search(scene, season, limit=3) 的场景/季节过滤、limit 与排序
    - 失败分支：单来源失败带警告、全失败 502、未配置来源、非法 scene/season/limit
    - 人工导入初始案例；模型抽取接入点（ModelStyleExtractor）
    - 空数据：无案例时检索返回 []

可直接用 pytest 收集，也可用 python -m unittest 运行（当前环境未安装 pytest）。
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx

from app.liuboyuan import trends
from app.liuboyuan.inspirations import (
    InMemoryInspirationRepository,
    InspirationsService,
)
from app.liuboyuan.trends import (
    CollectResult,
    ExtractedTrend,
    FetchedPage,
    HeuristicStyleExtractor,
    HttpFetcher,
    InMemoryTrendRepository,
    ModelStyleExtractor,
    TrendCase,
    TrendConfigError,
    TrendError,
    TrendUpstreamError,
    TrendsService,
    bootstrap_demo,
    load_trend_sources,
)


def make_page(title: str, body: str, images: list[str]) -> str:
    """构造来源页面：首图放 og:image，其余放 <img>。"""
    metas = "".join(
        f'<meta property="og:image" content="{url}">' for url in images[:1]
    )
    imgs = "".join(f'<img src="{url}">' for url in images[1:])
    return (
        f"<html><head><title>{title}</title>{metas}</head>"
        f"<body><p>{body}</p>{imgs}</body></html>"
    )


MINIMAL_SPRING = make_page(
    "Spring Minimal Commute",
    "Minimal commute looks for spring with clean lines and light layers.",
    [
        "https://cdn.example.com/minimal-1.jpg",
        "https://cdn.example.com/minimal-2.jpg",
    ],
)
STREETWEAR_SPRING = make_page(
    "Streetwear Spring Drop",
    "Streetwear street style for spring weekends.",
    ["https://cdn.example.com/street-1.jpg"],
)
MINIMAL_AUTUMN = make_page(
    "Autumn Minimal Layers",
    "Minimal layers for autumn commuting.",
    ["https://cdn.example.com/minimal-autumn-1.jpg"],
)


def handler_for(pages: dict[str, str], broken: tuple[str, ...] = ()):
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url in broken:
            return httpx.Response(500, text="upstream error")
        if url in pages:
            return httpx.Response(200, text=pages[url])
        return httpx.Response(404, text="not found")

    return handler


class TrendsTestCase(unittest.IsolatedAsyncioTestCase):
    """公共装置：内存存储 + Mock HTTPX。"""

    def setUp(self) -> None:
        self._sources_snapshot = trends.TREND_SOURCES

    def tearDown(self) -> None:
        trends.TREND_SOURCES = self._sources_snapshot

    async def asyncSetUp(self) -> None:
        self._clients: list[httpx.AsyncClient] = []

    async def asyncTearDown(self) -> None:
        for client in self._clients:
            await client.aclose()

    def build(
        self,
        handler,
        *,
        sources: list[str] | None = None,
        extractor=None,
        repository=None,
        inspirations: InspirationsService | None = None,
    ) -> tuple[TrendsService, InspirationsService]:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self._clients.append(client)
        self.inspirations = inspirations or InspirationsService(
            InMemoryInspirationRepository()
        )
        service = TrendsService(
            repository=repository or InMemoryTrendRepository(),
            inspirations=self.inspirations,
            fetcher=HttpFetcher(client=client),
            extractor=extractor or HeuristicStyleExtractor(),
            sources=sources,
        )
        return service, self.inspirations


# --------------------------------------------------------------------------- #
# TREND_SOURCES 配置
# --------------------------------------------------------------------------- #
class TestTrendSources(TrendsTestCase):
    def test_env_var_parsing_dedupes_and_drops_invalid(self) -> None:
        environ = {
            "TREND_SOURCES": (
                "https://a.example.com/looks, https://b.example.com/looks\n"
                "https://a.example.com/looks/, ftp://bad.example.com/x, not-a-url"
            )
        }
        sources = load_trend_sources(environ=environ)
        self.assertEqual(
            sources,
            ["https://a.example.com/looks", "https://b.example.com/looks"],
        )

    def test_settings_object_wins_over_env(self) -> None:
        environ = {"TREND_SOURCES": "https://env.example.com/x"}
        settings = SimpleNamespace(
            TREND_SOURCES=["https://settings.example.com/a", "https://settings.example.com/b"]
        )
        self.assertEqual(
            load_trend_sources(settings, environ=environ),
            ["https://settings.example.com/a", "https://settings.example.com/b"],
        )
        self.assertEqual(
            load_trend_sources({"TREND_SOURCES": "https://dict.example.com/a"}, environ=environ),
            ["https://dict.example.com/a"],
        )

    def test_reload_updates_module_snapshot(self) -> None:
        snapshot = trends.reload_trend_sources(
            environ={"TREND_SOURCES": "https://one.example.com/x"}
        )
        self.assertEqual(snapshot, ["https://one.example.com/x"])
        self.assertEqual(list(trends.TREND_SOURCES), ["https://one.example.com/x"])

    def test_empty_config_is_empty_list(self) -> None:
        self.assertEqual(load_trend_sources(environ={}), [])


# --------------------------------------------------------------------------- #
# 页面解析
# --------------------------------------------------------------------------- #
class TestParsePage(unittest.TestCase):
    def test_extracts_title_text_images_and_resolves_relative(self) -> None:
        body = (
            "<html><head><title>Minimal 春季通勤</title>"
            '<meta property="og:image" content="https://cdn.example.com/og.jpg">'
            "</head><body><script>var x = 1;</script>"
            '<p>Clean minimal layers.</p><img src="/img/rel.jpg">'
            '<img src="data:image/png;base64,AAA"></body></html>'
        )
        page = trends.parse_page("https://blog.example.com/look/1", body)
        self.assertEqual(page.title, "Minimal 春季通勤")
        self.assertIn("Clean minimal layers.", page.text)
        self.assertNotIn("var x", page.text)
        self.assertEqual(
            page.image_urls,
            (
                "https://cdn.example.com/og.jpg",
                "https://blog.example.com/img/rel.jpg",
            ),
        )

    def test_extracted_trend_model_accepts_page_fields(self) -> None:
        trend = ExtractedTrend(
            image_url="https://cdn.example.com/a.jpg",
            summary="minimal",
            tags=["minimal"],
        )
        self.assertEqual(trend.season, None)


# --------------------------------------------------------------------------- #
# 采集：成功 / 去重 / added_count / 刷新喜好列表
# --------------------------------------------------------------------------- #
class TestCollect(TrendsTestCase):
    async def test_collect_adds_cases_and_reports_added_count(self) -> None:
        source = "https://blog.example.com/spring-minimal"
        service, inspirations = self.build(
            handler_for({source: MINIMAL_SPRING}), sources=[source]
        )

        result = await service.collect("u1", "minimal", "spring")

        self.assertIsInstance(result, CollectResult)
        self.assertEqual(result.added_count, 2)
        self.assertEqual(result.duplicates, 0)
        self.assertEqual(result.warnings, [])
        self.assertEqual(result.as_job_result(), {"added_count": 2})

        items = (await inspirations.list_items("u1"))["items"]
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].source_url, source)
        # 喜好列表倒序：后写入的在前
        self.assertEqual(
            [item.image_url for item in items],
            [
                "https://cdn.example.com/minimal-2.jpg",
                "https://cdn.example.com/minimal-1.jpg",
            ],
        )

    async def test_recollect_same_source_adds_nothing(self) -> None:
        source = "https://blog.example.com/spring-minimal"
        service, inspirations = self.build(
            handler_for({source: MINIMAL_SPRING}), sources=[source]
        )

        first = await service.collect("u1", "minimal", "spring")
        second = await service.collect("u1", "minimal", "spring")

        self.assertEqual(first.added_count, 2)
        self.assertEqual(second.added_count, 0)
        self.assertEqual(second.duplicates, 2)
        self.assertEqual(len((await inspirations.list_items("u1"))["items"]), 2)

    async def test_style_filter_drops_other_styles(self) -> None:
        minimal_url = "https://blog.example.com/spring-minimal"
        street_url = "https://blog.example.com/spring-street"
        service, _ = self.build(
            handler_for(
                {minimal_url: MINIMAL_SPRING, street_url: STREETWEAR_SPRING}
            ),
            sources=[minimal_url, street_url],
        )

        result = await service.collect("u1", "minimal", "spring")

        self.assertEqual(result.added_count, 2)
        self.assertTrue(
            all("minimal" in case.image_url for case in result.created),
            [case.image_url for case in result.created],
        )

    async def test_season_filter_drops_other_seasons(self) -> None:
        autumn_url = "https://blog.example.com/autumn-minimal"
        service, _ = self.build(
            handler_for({autumn_url: MINIMAL_AUTUMN}), sources=[autumn_url]
        )

        result = await service.collect("u1", "minimal", "spring")

        self.assertEqual(result.added_count, 0)
        self.assertEqual(result.duplicates, 0)

    async def test_unknown_season_inherits_requested_season(self) -> None:
        source = "https://blog.example.com/minimal-guide"
        page = make_page(
            "Clean Minimal Guide",
            "Minimal looks with clean lines, no seasonal words here.",
            ["https://cdn.example.com/guide-1.jpg"],
        )
        service, _ = self.build(handler_for({source: page}), sources=[source])

        result = await service.collect("u1", "minimal", "winter")

        self.assertEqual(result.added_count, 1)
        self.assertEqual(result.created[0].season, "winter")

    async def test_partial_source_failure_returns_warnings(self) -> None:
        good = "https://blog.example.com/spring-minimal"
        broken = "https://broken.example.com/feeds"
        service, _ = self.build(
            handler_for({good: MINIMAL_SPRING}, broken=(broken,)),
            sources=[good, broken],
        )

        result = await service.collect("u1", "minimal", "spring")

        self.assertEqual(result.added_count, 2)
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("TREND_SOURCE_FAILED", result.warnings[0])
        self.assertIn(broken, result.warnings[0])

    async def test_all_sources_fail_raises_upstream_error(self) -> None:
        broken = ("https://broken-a.example.com/feed", "https://broken-b.example.com/feed")
        service, _ = self.build(
            handler_for({}, broken=broken), sources=list(broken)
        )

        with self.assertRaises(TrendUpstreamError) as ctx:
            await service.collect("u1", "minimal", "spring")

        self.assertEqual(ctx.exception.code, "UPSTREAM_FAILED")
        self.assertEqual(ctx.exception.status_code, 502)
        self.assertTrue(ctx.exception.retryable)
        self.assertIn("TREND_SOURCE_FAILED", ctx.exception.message)

    async def test_missing_sources_config_raises(self) -> None:
        service, _ = self.build(handler_for({}), sources=[])
        with self.assertRaises(TrendConfigError) as ctx:
            await service.collect("u1", "minimal", "spring")
        self.assertEqual(ctx.exception.code, "TREND_SOURCES_MISSING")

    async def test_collect_uses_env_trend_sources(self) -> None:
        source = "https://blog.example.com/env-source"
        service, inspirations = self.build(
            handler_for({source: MINIMAL_SPRING}), sources=None
        )
        with mock.patch.dict(os.environ, {"TREND_SOURCES": source}, clear=False):
            result = await service.collect("u1", "minimal", "spring")

        self.assertEqual(result.added_count, 2)
        self.assertEqual(len((await inspirations.list_items("u1"))["items"]), 2)

    async def test_invalid_season_rejected(self) -> None:
        source = "https://blog.example.com/spring-minimal"
        service, _ = self.build(handler_for({source: MINIMAL_SPRING}), sources=[source])

        with self.assertRaises(TrendError) as ctx:
            await service.collect("u1", "minimal", "monsoon")
        self.assertEqual(ctx.exception.code, "INVALID_SEASON")

    async def test_blank_style_rejected(self) -> None:
        service, _ = self.build(handler_for({}), sources=["https://a.example.com/x"])
        with self.assertRaises(TrendError) as ctx:
            await service.collect("u1", "  ", "spring")
        self.assertEqual(ctx.exception.code, "INVALID_FIELD")


# --------------------------------------------------------------------------- #
# 模型抽取接入点
# --------------------------------------------------------------------------- #
class TestModelStyleExtractor(TrendsTestCase):
    async def test_model_output_parsed_and_guarded(self) -> None:
        page = FetchedPage(
            url="https://blog.example.com/model",
            title="Model page",
            text="text",
            image_urls=("https://cdn.example.com/real.jpg",),
        )
        payload = {
            "cases": [
                {
                    "image_url": "https://cdn.example.com/hallucinated.jpg",
                    "summary": "minimal commute",
                    "style": "Minimal",
                    "season": "spring",
                    "scene": "commute",
                    "tags": ["minimal"],
                },
                {
                    "image_url": "https://cdn.example.com/real.jpg",
                    "summary": "bad season",
                    "style": "minimal",
                    "season": "monsoon",
                    "scene": "party",
                    "tags": [],
                },
            ]
        }

        async def complete_json(prompt: str) -> str:
            self.assertIn("时尚趋势标注器", prompt)
            return json.dumps(payload, ensure_ascii=False)

        extractor = ModelStyleExtractor(complete_json)
        cases = await extractor.extract(page)

        self.assertEqual(len(cases), 2)
        # 模型编造的图片被回退到页面首图
        self.assertEqual(cases[0].image_url, "https://cdn.example.com/real.jpg")
        self.assertEqual(cases[0].style, "minimal")
        self.assertEqual(cases[0].season, "spring")
        # 非法枚举值被置空
        self.assertIsNone(cases[1].season)
        self.assertIsNone(cases[1].scene)


# --------------------------------------------------------------------------- #
# 检索：search(scene, season, limit=3)
# --------------------------------------------------------------------------- #
class TestSearch(TrendsTestCase):
    def case(
        self,
        case_id: str,
        *,
        scene: str | None,
        season: str | None,
        image: str | None = None,
    ) -> TrendCase:
        return TrendCase(
            case_id=case_id,
            image_url=image or f"https://cdn.example.com/{case_id}.jpg",
            summary=f"{scene or 'untyped'} {season or 'any'} look",
            source_url=f"https://demo.example.com/{case_id}",
            style="minimal",
            season=season,
            scene=scene,
            tags=["minimal"],
        )

    async def test_three_arg_signature_reads_default_scope(self) -> None:
        service, _ = self.build(handler_for({}), sources=[])
        await bootstrap_demo(service)

        cases = await service.search("commute", "spring", 3)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].scene, "commute")
        self.assertEqual(cases[0].season, "spring")

    async def test_filters_by_scene_and_season_with_limit(self) -> None:
        service, _ = self.build(handler_for({}), sources=[])
        for index in range(4):
            await service.add_manual_cases(
                "u1",
                [self.case(f"commute_spring_{index}", scene="commute", season="spring")],
            )
        await service.add_manual_cases(
            "u1", [self.case("work_winter", scene="commute", season="winter")]
        )
        await service.add_manual_cases(
            "u1", [self.case("weekend_spring", scene="casual", season="spring")]
        )

        cases = await service.search("commute", "spring", 3, user_id="u1")

        self.assertEqual(len(cases), 3)
        for case in cases:
            self.assertEqual(case.scene, "commute")
            self.assertEqual(case.season, "spring")
        # 倒序：后插入的在前
        self.assertEqual(cases[0].case_id, "commute_spring_3")
        self.assertEqual(cases[2].case_id, "commute_spring_1")

    async def test_limit_is_clamped_but_not_silently_zero(self) -> None:
        service, _ = self.build(handler_for({}), sources=[])
        await service.add_manual_cases(
            "u1", [self.case("a", scene="commute", season="spring")]
        )
        self.assertEqual(len(await service.search("commute", "spring", 999, user_id="u1")), 1)

        with self.assertRaises(TrendError) as ctx:
            await service.search("commute", "spring", 0, user_id="u1")
        self.assertEqual(ctx.exception.code, "INVALID_LIMIT")

    async def test_invalid_scene_rejected(self) -> None:
        service, _ = self.build(handler_for({}), sources=[])
        with self.assertRaises(TrendError) as ctx:
            await service.search("party", "spring", 3, user_id="u1")
        self.assertEqual(ctx.exception.code, "INVALID_SCENE")

    async def test_untyped_scene_cases_fill_after_exact_matches(self) -> None:
        service, _ = self.build(handler_for({}), sources=[])
        await service.add_manual_cases(
            "u1", [self.case("exact", scene="business", season="spring")]
        )
        await service.add_manual_cases(
            "u1", [self.case("untyped", scene=None, season="spring")]
        )
        await service.add_manual_cases(
            "u1", [self.case("other_scene", scene="casual", season="spring")]
        )

        cases = await service.search("business", "spring", 3, user_id="u1")

        self.assertEqual([case.case_id for case in cases], ["exact", "untyped"])
        self.assertIsNone(cases[1].scene)

    async def test_empty_repository_returns_empty_list(self) -> None:
        service, _ = self.build(handler_for({}), sources=[])
        self.assertEqual(await service.search("commute", "spring", 3, user_id="u1"), [])

    async def test_search_is_user_scoped(self) -> None:
        service, _ = self.build(handler_for({}), sources=[])
        await service.add_manual_cases(
            "u1", [self.case("only_u1", scene="commute", season="spring")]
        )
        self.assertEqual(await service.search("commute", "spring", 3, user_id="u2"), [])
        self.assertEqual(
            len(await service.search("commute", "spring", 3, user_id="u1")), 1
        )

    async def test_search_failure_returns_warning_for_recommender(self) -> None:
        class BrokenRepository(InMemoryTrendRepository):
            async def search(self, *args, **kwargs):
                raise RuntimeError("db down")

        service, _ = self.build(
            handler_for({}), sources=[], repository=BrokenRepository()
        )
        cases, warnings = await service.search_with_warnings(
            "commute", "spring", 3, user_id="u1"
        )
        self.assertEqual(cases, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("TREND_SEARCH_FAILED", warnings[0])
        self.assertIn("db down", warnings[0])


# --------------------------------------------------------------------------- #
# 喜好图片：新增 + 倒序列表（02 POST/GET /inspirations）
# --------------------------------------------------------------------------- #
class TestInspirationList(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.inspirations = InspirationsService(InMemoryInspirationRepository())

    async def test_create_returns_inspiration_and_list_is_reverse_chronological(self) -> None:
        first = await self.inspirations.create(
            "u1",
            "img_1",
            tags=["minimal"],
            source_url="https://blog.example.com/look-a",
        )
        await self.inspirations.create("u1", "img_2", tags=["casual"])
        await self.inspirations.create("u1", "img_3")

        self.assertTrue(first.inspiration_id.startswith("insp_"))
        self.assertEqual(first.image_url, "/media/img_1.jpg")
        self.assertEqual(first.source_url, "https://blog.example.com/look-a")

        items = (await self.inspirations.list_items("u1"))["items"]
        self.assertEqual(
            [item.image_url for item in items],
            ["/media/img_3.jpg", "/media/img_2.jpg", "/media/img_1.jpg"],
        )
        self.assertEqual(
            list(items[0].model_dump().keys()),
            ["inspiration_id", "image_id", "image_url", "tags", "source_url"],
        )

    async def test_tags_are_normalized(self) -> None:
        created = await self.inspirations.create(
            "u1", "img_1", tags=[" minimal ", "minimal", "", "clean"]
        )
        self.assertEqual(created.tags, ["minimal", "clean"])

    async def test_blank_image_id_rejected(self) -> None:
        from app.liuboyuan.inspirations import InspirationError

        with self.assertRaises(InspirationError) as ctx:
            await self.inspirations.create("u1", "   ")
        self.assertEqual(ctx.exception.code, "INVALID_FIELD")

    async def test_invalid_source_url_rejected(self) -> None:
        from app.liuboyuan.inspirations import InspirationError

        with self.assertRaises(InspirationError) as ctx:
            await self.inspirations.create(
                "u1", "img_1", source_url="ftp://bad.example.com/look"
            )
        self.assertEqual(ctx.exception.code, "INVALID_SOURCE_URL")

    async def test_list_is_user_scoped(self) -> None:
        await self.inspirations.create("u1", "img_1")
        await self.inspirations.create("u2", "img_2")

        self.assertEqual(len((await self.inspirations.list_items("u1"))["items"]), 1)
        self.assertEqual(
            (await self.inspirations.list_items("u1"))["items"][0].image_url,
            "/media/img_1.jpg",
        )


# --------------------------------------------------------------------------- #
# 人工导入初始案例
# --------------------------------------------------------------------------- #
class TestManualImport(TrendsTestCase):
    async def test_manual_import_dedupes_and_assigns_ids(self) -> None:
        service, _ = self.build(handler_for({}), sources=[])
        cases = [
            {
                "case_id": "",
                "image_url": "https://cdn.example.com/manual-1.jpg",
                "summary": "manual",
                "source_url": "https://manual.example.com/1",
                "season": "spring",
                "scene": "commute",
            }
        ]

        created = await service.add_manual_cases("u1", cases)
        again = await service.add_manual_cases("u1", cases)

        self.assertEqual(len(created), 1)
        self.assertTrue(created[0].case_id.startswith("trend_"))
        self.assertEqual(again, [])
        self.assertEqual(len(await service.search("commute", "spring", 3, user_id="u1")), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
