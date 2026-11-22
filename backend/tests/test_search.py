"""整合验证：HTTP 全流程、真实 socket、趋势检索隔离、画像 30 天窗口。

覆盖 02 / 03 / 04 的交叉验收点：
    * 上传（multipart）-> 识别（202 Job）-> 轮询 succeeded -> 修正标签入库
      -> 查询到修正值 -> PATCH 整体替换 -> 以参考图搜衣，全链路走 02 的包装与状态码；
    * 同一套服务同时经 ``Api.dispatch`` 与真实 HTTP（stdlib 适配器）验证；
    * 推荐调用趋势检索时的**用户隔离**（谁的案例只给谁，3 参检索走独立演示 scope）；
    * 画像读取时**过期反馈退出 30 天窗口**（权重按窗口重算，不是累加历史）；
    * 三个任务处理函数（garment_analysis / inspiration_collection / memory_update）
      经任务框架跑通，失败写 Job.error。

郭子骏的上传/任务框架、何梓森的反馈/偏好仍为 Mock（见各文件顶部接入点），
趋势抓取用离线 StubFetcher，不依赖外网。
可直接 pytest 收集，也可用 python -m unittest 运行（当前环境未安装 pytest）。
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.liuboyuan.container import build_services
from app.liuboyuan.http_server import make_server
from app.liuboyuan.jobs import JOB_TYPES
from app.liuboyuan.memory import MockFeedbackSource, MockPreferencesSource
from app.liuboyuan.trends import FetchedPage, HeuristicStyleExtractor

USER = "u_1"
OTHER = "u_2"
DEMO = "u_demo"


def jpeg(payload: bytes) -> bytes:
    return b"\xff\xd8\xff\xe0" + payload


def corrected_tags(**overrides) -> dict:
    data = {"category": "top", "color": "off-white", "style_tags": ["minimal"]}
    data.update(overrides)
    return data


async def upload(services, *, user: str = USER, payload: bytes = b"photo") -> dict:
    response = await services.api.upload_image(user, jpeg(payload))
    assert response.status_code == 201, response.body
    return response.body["data"]


async def poll_job(services, user: str, job_id: str, timeout: float = 5.0):
    return await services.jobs.wait(user, job_id, timeout=timeout)


# --------------------------------------------------------------------------- #
# 1. 02 契约层全流程（Api.dispatch，不经过 socket）
# --------------------------------------------------------------------------- #
class HttpContractFlowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.services = build_services(users=(USER, OTHER))
        self.api = self.services.api

    async def asyncTearDown(self) -> None:
        await self.services.jobs.wait_all(timeout=5)

    async def test_upload_recognize_correct_query_patch_then_search(self) -> None:
        api = self.api

        uploaded = await api.dispatch(
            "POST", "/api/v1/images", user_id=USER, upload=(jpeg(b"white-tee"), "image/jpeg")
        )
        self.assertEqual(uploaded.status_code, 201)
        self.assertEqual(set(uploaded.body), {"data", "request_id"})
        image = uploaded.body["data"]

        analyzed = await api.dispatch(
            "POST",
            "/api/v1/garments:analyze",
            user_id=USER,
            idempotency_key="flow-1",
            body={"image_id": image["image_id"]},
        )
        self.assertEqual(analyzed.status_code, 202)
        job = analyzed.body["data"]
        self.assertEqual(job["type"], "garment_analysis")
        self.assertIsNone(job["result"])
        self.assertEqual(job["warnings"], [])

        fetched = await api.dispatch(
            "GET", f"/api/v1/jobs/{job['job_id']}", user_id=USER
        )
        self.assertEqual(fetched.status_code, 200)
        done = await poll_job(self.services, USER, job["job_id"])
        self.assertEqual(done.status, "succeeded")
        self.assertEqual(done.result["image_id"], image["image_id"])

        created = await api.dispatch(
            "POST",
            "/api/v1/garments",
            user_id=USER,
            body={"image_id": image["image_id"], "tags": corrected_tags()},
        )
        self.assertEqual(created.status_code, 201)
        garment = created.body["data"]

        listed = await api.dispatch("GET", "/api/v1/garments", user_id=USER)
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(len(listed.body["data"]["items"]), 1)
        self.assertEqual(
            listed.body["data"]["items"][0]["tags"]["color"], "off-white"
        )

        patched = await api.dispatch(
            "PATCH",
            f"/api/v1/garments/{garment['garment_id']}",
            user_id=USER,
            body={"tags": {"category": "top", "color": "black"}},
        )
        self.assertEqual(patched.status_code, 200)
        self.assertEqual(patched.body["data"]["tags"]["style_tags"], [])

        after = await api.dispatch("GET", "/api/v1/garments", user_id=USER)
        self.assertEqual(
            after.body["data"]["items"][0]["tags"]["color"], "black"
        )

        # 以参考图搜衣：同内容 -> 1.0，且只召回刚确认的单品
        reference = await upload(self.services, payload=b"white-tee")
        candidates = await self.services.closet.search(
            USER, reference["image_id"], limit=10
        )
        self.assertEqual([item.garment_id for item in candidates], [garment["garment_id"]])
        self.assertEqual(candidates[0].score, 1.0)

        # request_id 每条响应都有且各不相同
        self.assertNotEqual(
            uploaded.body["request_id"], listed.body["request_id"]
        )

    async def test_error_envelopes(self) -> None:
        api = self.api

        unauthorized = await api.dispatch("GET", "/api/v1/garments")
        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(unauthorized.body["error"]["code"], "UNAUTHORIZED")

        unknown_path = await api.dispatch("GET", "/api/v1/nope", user_id=USER)
        self.assertEqual(unknown_path.status_code, 404)

        missing_image = await api.dispatch(
            "POST",
            "/api/v1/garments:analyze",
            user_id=USER,
            idempotency_key="k-404",
            body={"image_id": "img_missing"},
        )
        self.assertEqual(missing_image.status_code, 404)
        self.assertEqual(missing_image.body["error"]["code"], "IMAGE_NOT_FOUND")

        missing_key = await api.dispatch(
            "POST",
            "/api/v1/garments:analyze",
            user_id=USER,
            body={"image_id": "img_missing"},
        )
        self.assertEqual(missing_key.status_code, 422)
        self.assertEqual(missing_key.body["error"]["code"], "INVALID_INPUT")

        image = await upload(self.services)
        first = await api.dispatch(
            "POST",
            "/api/v1/garments:analyze",
            user_id=USER,
            idempotency_key="k-conflict",
            body={"image_id": image["image_id"]},
        )
        self.assertEqual(first.status_code, 202)
        second = await upload(self.services, payload=b"another")
        conflict = await api.dispatch(
            "POST",
            "/api/v1/garments:analyze",
            user_id=USER,
            idempotency_key="k-conflict",
            body={"image_id": second["image_id"]},
        )
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(conflict.body["error"]["code"], "IDEMPOTENCY_CONFLICT")

        invalid_tags = await api.dispatch(
            "POST",
            "/api/v1/garments",
            user_id=USER,
            body={"image_id": image["image_id"], "tags": {"category": "hat", "color": "x"}},
        )
        self.assertEqual(invalid_tags.status_code, 422)
        self.assertEqual(invalid_tags.body["error"]["code"], "INVALID_TAGS")

        unknown_field = await api.dispatch(
            "POST",
            "/api/v1/garments",
            user_id=USER,
            body={
                "image_id": image["image_id"],
                "tags": corrected_tags(),
                "extra": 1,
            },
        )
        self.assertEqual(unknown_field.status_code, 422)
        self.assertEqual(unknown_field.body["error"]["code"], "INVALID_INPUT")


# --------------------------------------------------------------------------- #
# 2. 真实 HTTP（stdlib 适配器，socket 上的 multipart 与状态码）
# --------------------------------------------------------------------------- #
def multipart(
    payload: bytes,
    content_type: str = "image/jpeg",
    filename: str = "photo.jpg",
    boundary: str = "----XClosetBoundary",
) -> tuple[bytes, str]:
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
    return head + payload + tail, f"multipart/form-data; boundary={boundary}"


def http_call(
    port: int,
    method: str,
    path: str,
    *,
    user: str | None = None,
    key: str | None = None,
    body: dict | None = None,
    upload_payload: bytes | None = None,
) -> tuple[int, dict]:
    data: bytes | None = None
    content_type: str | None = None
    if upload_payload is not None:
        data, content_type = multipart(upload_payload)
    elif body is not None:
        data = json.dumps(body).encode("utf-8")
        content_type = "application/json"

    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data, method=method
    )
    if content_type:
        request.add_header("Content-Type", content_type)
    if user:
        request.add_header("X-User-Id", user)
    if key:
        request.add_header("Idempotency-Key", key)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode("utf-8"))
        finally:
            exc.close()


class RealHttpServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.services = build_services(users=(USER, OTHER))
        self.server = make_server(self.services, port=0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(
            target=self.server.serve_forever, name="x-closet-test-http", daemon=True
        )
        self.thread.start()

    async def asyncTearDown(self) -> None:
        await asyncio.to_thread(self.server.shutdown)
        self.thread.join(timeout=5)
        self.server.server_close()

    async def call(self, *args, **kwargs) -> tuple[int, dict]:
        return await asyncio.to_thread(http_call, self.port, *args, **kwargs)

    async def test_full_flow_over_real_http(self) -> None:
        status, body = await self.call(
            "POST", "/api/v1/images", user=USER, upload_payload=jpeg(b"white-tee")
        )
        self.assertEqual(status, 201)
        image_id = body["data"]["image_id"]

        status, body = await self.call(
            "POST",
            "/api/v1/garments:analyze",
            user=USER,
            key="http-1",
            body={"image_id": image_id},
        )
        self.assertEqual(status, 202)
        job_id = body["data"]["job_id"]

        deadline = asyncio.get_running_loop().time() + 5.0
        status, body = await self.call("GET", f"/api/v1/jobs/{job_id}", user=USER)
        while body["data"]["status"] not in ("succeeded", "failed"):
            self.assertLess(asyncio.get_running_loop().time(), deadline)
            await asyncio.sleep(0.02)
            status, body = await self.call("GET", f"/api/v1/jobs/{job_id}", user=USER)
        self.assertEqual(status, 200)
        self.assertEqual(body["data"]["status"], "succeeded")
        self.assertEqual(body["data"]["result"]["image_id"], image_id)

        status, body = await self.call(
            "POST",
            "/api/v1/garments",
            user=USER,
            body={"image_id": image_id, "tags": corrected_tags(color="black")},
        )
        self.assertEqual(status, 201)
        garment_id = body["data"]["garment_id"]

        status, body = await self.call("GET", "/api/v1/garments", user=USER)
        self.assertEqual(status, 200)
        self.assertEqual(body["data"]["items"][0]["tags"]["color"], "black")

        status, body = await self.call(
            "PATCH",
            f"/api/v1/garments/{garment_id}",
            user=USER,
            body={"tags": {"category": "top", "color": "navy"}},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["data"]["tags"]["color"], "navy")
        self.assertEqual(body["data"]["tags"]["style_tags"], [])

        status, body = await self.call(
            "GET", "/api/v1/garments?category=top", user=USER
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(body["data"]["items"]), 1)
        self.assertEqual(body["data"]["items"][0]["tags"]["color"], "navy")

        # 真实 HTTP 上的 401 / 404 / 422 / 409
        status, body = await self.call("GET", "/api/v1/garments")
        self.assertEqual((status, body["error"]["code"]), (401, "UNAUTHORIZED"))

        status, body = await self.call(
            "POST",
            "/api/v1/garments:analyze",
            user=USER,
            key="http-404",
            body={"image_id": "img_missing"},
        )
        self.assertEqual((status, body["error"]["code"]), (404, "IMAGE_NOT_FOUND"))

        status, body = await self.call(
            "POST",
            "/api/v1/garments",
            user=USER,
            body={"image_id": image_id, "tags": {"category": "hat", "color": "x"}},
        )
        self.assertEqual((status, body["error"]["code"]), (422, "INVALID_TAGS"))

        status, body = await self.call(
            "POST",
            "/api/v1/garments:analyze",
            user=USER,
            key="http-409",
            body={"image_id": image_id},
        )
        self.assertEqual(status, 202)
        status, body = await self.call(
            "POST", "/api/v1/images", user=USER, upload_payload=jpeg(b"second-tee")
        )
        self.assertEqual(status, 201)
        other_image_id = body["data"]["image_id"]
        status, body = await self.call(
            "POST",
            "/api/v1/garments:analyze",
            user=USER,
            key="http-409",
            body={"image_id": other_image_id},
        )
        self.assertEqual(status, 409)
        self.assertEqual(body["error"]["code"], "IDEMPOTENCY_CONFLICT")

    async def test_upload_validates_media_type_over_http(self) -> None:
        status, body = await self.call(
            "POST", "/api/v1/images", user=USER, upload_payload=b"plain text"
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error"]["code"], "INVALID_IMAGE")


# --------------------------------------------------------------------------- #
# 3. 趋势检索：用户隔离 + 采集任务 + 失败转 warning
# --------------------------------------------------------------------------- #
PAGE = FetchedPage(
    url="https://trends.example.com/minimal-spring",
    title="Spring Minimal Commute",
    text="Minimal commute looks for spring with clean lines and light layers.",
    image_urls=(
        "https://cdn.example.com/look-1.jpg",
        "https://cdn.example.com/look-2.jpg",
    ),
)


class StubFetcher:
    """离线抓取替身：不访问外网，返回固定 FetchedPage。"""

    def __init__(self, pages: tuple[FetchedPage, ...] = (PAGE,)) -> None:
        self._pages = {page.url: page for page in pages}
        self.calls: list[str] = []

    async def fetch(self, url: str) -> FetchedPage:
        self.calls.append(url)
        page = self._pages.get(url)
        if page is None:  # pragma: no cover - 未配置的来源
            raise RuntimeError(f"来源不可达：{url}")
        return page


class TrendIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.services = build_services(
            users=(USER, OTHER),
            trend_sources=[PAGE.url],
            fetcher=StubFetcher(),
            extractor=HeuristicStyleExtractor(),
        )
        self.api = self.services.api

    async def asyncTearDown(self) -> None:
        await self.services.jobs.wait_all(timeout=5)

    async def test_collect_job_refreshes_inspirations_and_is_isolated(self) -> None:
        response = await self.api.collect_inspirations(
            USER, "collect-1", {"style": "minimal", "season": "spring"}
        )
        self.assertEqual(response.status_code, 202)
        job = response.body["data"]
        self.assertEqual(job["type"], "inspiration_collection")

        done = await poll_job(self.services, USER, job["job_id"])
        self.assertEqual(done.status, "succeeded")
        self.assertEqual(done.result, {"added_count": 2})

        # 采集成功后刷新喜好列表（02：GET /inspirations 倒序返回全部）
        listed = await self.api.list_inspirations(USER)
        self.assertEqual(len(listed.body["data"]["items"]), 2)

        # 同一来源重复采集：只新增 0 条
        replay = await self.api.collect_inspirations(
            USER, "collect-2", {"style": "minimal", "season": "spring"}
        )
        replay_done = await poll_job(self.services, USER, replay.body["data"]["job_id"])
        self.assertEqual(replay_done.result, {"added_count": 0})

        # 推荐调用 trends.search(scene, season, limit) 时的用户隔离
        mine = await self.services.trends.search(
            "commute", "spring", 3, user_id=USER
        )
        self.assertEqual(len(mine), 2)
        self.assertTrue(all(case.scene == "commute" for case in mine))

        theirs = await self.services.trends.search(
            "commute", "spring", 3, user_id=OTHER
        )
        self.assertEqual(theirs, [])
        demo = await self.services.trends.search("commute", "spring", 3)
        self.assertEqual(demo, [])  # 3 参调用走独立演示 scope，不串用户

        self.assertEqual(DEMO, self.services.trends.default_user_id)

    async def test_trend_failure_becomes_warning_not_crash(self) -> None:
        cases, warnings = await self.services.trends.search_with_warnings(
            "party", "spring", 3, user_id=USER
        )
        self.assertEqual(cases, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("TREND_SEARCH_FAILED", warnings[0])


# --------------------------------------------------------------------------- #
# 4. 画像：最近三张喜好图 + 过期反馈退出 30 天窗口 + memory_update 任务
# --------------------------------------------------------------------------- #
class MemoryWindowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.clock = {"now": datetime(2025, 5, 1, 12, 0, tzinfo=timezone.utc)}
        self.feedback = MockFeedbackSource()
        self.preferences = MockPreferencesSource()
        self.preferences.set(USER, liked_styles=["minimal"], avoid_tags=["sexy"])
        self.services = build_services(
            users=(USER,),
            feedback=self.feedback,
            preferences=self.preferences,
            now=lambda: self.clock["now"],
        )

    async def asyncTearDown(self) -> None:
        await self.services.jobs.wait_all(timeout=5)

    async def test_profile_shape_and_recent_three_inspirations(self) -> None:
        for index in range(4):
            await self.services.inspirations.create(
                USER,
                f"img_{index}",
                tags=["minimal"],
                source_url=f"https://looks.example.com/{index}",
            )
        profile = await self.services.memory.get(USER)
        self.assertEqual(profile.version, 0)
        self.assertEqual(profile.explicit.liked_styles, ["minimal"])
        self.assertEqual(profile.explicit.avoid_tags, ["sexy"])
        self.assertEqual(profile.learned_weights, {})
        self.assertEqual(len(profile.inspiration_image_urls), 3)  # 最近三张
        self.assertEqual(profile.inspiration_image_urls[0], "/media/img_3.jpg")

    async def test_expired_feedback_leaves_the_window(self) -> None:
        self.feedback.add_feedback(
            "fb_old",
            USER,
            action="like",
            style_tags=["minimal"],
            created_at=self.clock["now"],
        )
        first = await self.services.memory.update(USER, "fb_old")
        self.assertEqual(first, {"profile_version": 1})
        profile = await self.services.memory.get(USER)
        self.assertEqual(profile.learned_weights, {"minimal": 1.0})

        # 31 天后：旧反馈已出 30 天窗口，权重按窗口重算而不是累加
        self.clock["now"] += timedelta(days=31)
        self.feedback.add_feedback(
            "fb_new",
            USER,
            action="like",
            style_tags=["minimal"],
            created_at=self.clock["now"],
        )
        second = await self.services.memory.update(USER, "fb_new")
        self.assertEqual(second, {"profile_version": 2})
        profile = await self.services.memory.get(USER)
        self.assertEqual(profile.learned_weights, {"minimal": 1.0})
        self.assertEqual(profile.version, 2)

        # 重放同一反馈：不重复计权、版本不变
        replay = await self.services.memory.update(USER, "fb_new")
        self.assertEqual(replay, {"profile_version": 2})

    async def test_explicit_avoid_tags_win_over_learned_likes(self) -> None:
        self.feedback.add_feedback(
            "fb_sexy",
            USER,
            action="like",
            style_tags=["sexy", "minimal"],
            created_at=self.clock["now"],
        )
        await self.services.memory.update(USER, "fb_sexy")
        profile = await self.services.memory.get(USER)
        self.assertEqual(profile.learned_weights.get("sexy"), 0.0)
        self.assertEqual(profile.learned_weights.get("minimal"), 1.0)

    async def test_memory_update_job_and_failure(self) -> None:
        self.feedback.add_feedback(
            "fb_job",
            USER,
            action="like",
            style_tags=["minimal"],
            created_at=self.clock["now"],
        )
        job = await self.services.jobs.submit(
            user_id=USER,
            path="/api/v1/feedback",
            job_type="memory_update",
            params={"feedback_id": "fb_job"},
            idempotency_key="mem-1",
        )
        done = await self.services.jobs.wait(USER, job.job_id, timeout=5)
        self.assertEqual(done.status, "succeeded")
        self.assertEqual(done.result, {"profile_version": 1})

        replay = await self.services.jobs.submit(
            user_id=USER,
            path="/api/v1/feedback",
            job_type="memory_update",
            params={"feedback_id": "fb_job"},
            idempotency_key="mem-1",
        )
        self.assertEqual(replay.job_id, job.job_id)

        missing = await self.services.jobs.submit(
            user_id=USER,
            path="/api/v1/feedback",
            job_type="memory_update",
            params={"feedback_id": "fb_missing"},
            idempotency_key="mem-2",
        )
        failed = await self.services.jobs.wait(USER, missing.job_id, timeout=5)
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.error.code, "FEEDBACK_NOT_FOUND")


# --------------------------------------------------------------------------- #
# 5. 任务框架与装配
# --------------------------------------------------------------------------- #
class ContainerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.services = build_services(users=(USER,))

    async def asyncTearDown(self) -> None:
        await self.services.jobs.wait_all(timeout=5)

    async def test_three_handlers_registered(self) -> None:
        self.assertEqual(
            self.services.jobs.handler_types(),
            ("garment_analysis", "inspiration_collection", "memory_update"),
        )
        for job_type in (
            "garment_analysis",
            "inspiration_collection",
            "memory_update",
            "recommendation",
            "tryon",
            "plan",
        ):
            self.assertIn(job_type, JOB_TYPES)

    async def test_handlers_share_one_wardrobe_and_embedder(self) -> None:
        # 装配只建一套实例：识别/衣橱/以图搜衣共用图片仓与向量仓
        self.assertIs(self.services.wardrobe.embedder, self.services.embedder)
        self.assertIs(self.services.wardrobe.vectors, self.services.vectors)
        self.assertIs(self.services.recognition.images, self.services.images)
        self.assertIs(self.services.closet.images, self.services.images)
        self.assertIs(self.services.closet.wardrobe, self.services.wardrobe)

    async def test_recommendation_composition_intersection(self) -> None:
        """03 的 ②：参考图候选与（天气/禁忌）过滤结果取交集，不足时补入。"""
        top = await upload(self.services, payload=b"warm-top")
        await self.services.api.create_garment(
            USER,
            {
                "image_id": top["image_id"],
                "tags": {"category": "top", "color": "beige", "warmth": "warm"},
            },
        )
        bottom = await upload(self.services, payload=b"light-bottom")
        await self.services.api.create_garment(
            USER,
            {
                "image_id": bottom["image_id"],
                "tags": {"category": "bottom", "color": "navy", "warmth": "light"},
            },
        )

        reference = await upload(self.services, payload=b"warm-top")
        candidates = await self.services.closet.search(
            USER, reference["image_id"], limit=10
        )
        self.assertEqual(len(candidates), 2)

        allowed = {
            garment.garment_id
            for garment in await self.services.wardrobe.list(USER)
            if garment.tags.warmth != "light"
        }
        intersection = [item for item in candidates if item.garment_id in allowed]
        self.assertEqual(len(intersection), 1)
        self.assertEqual(intersection[0].score, 1.0)
        # 交集不足时的兜底：补入已过滤的衣物（由编排层完成，这里验证数据可用）
        fallback = [
            garment
            for garment in await self.services.wardrobe.list(USER)
            if garment.garment_id in allowed
        ]
        self.assertEqual(len(fallback), 1)


if __name__ == "__main__":
    unittest.main()
