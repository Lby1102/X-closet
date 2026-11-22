"""识别 -> 修正标签确认入库 -> 查询 -> 改标签（04 最小验收「录入一件上衣」）。

覆盖 02/03/04 的验收点：
    * 上传后识别：202 Job（type=garment_analysis）-> succeeded result={image_id,tags}；
    * 用户修正标签后 POST /garments 入库；GET /garments 查到**修正后的值**；
    * PATCH 提交的是**完整** Tags：省略的可选字段被清空（整体替换，不是合并）；
    * 用户隔离：列表 / 改标签 / 他人图片 / 他人 Job 全部 404 或空；
    * 失败分支：缺 Idempotency-Key 422、未知图片 404、非法标签 422、
      幂等键冲突 409、视觉失败 -> Job.error=UPSTREAM_FAILED（可重试）；
    * 上传校验：非 JPEG/PNG 422、超过 10 MB 413。

视觉模型与图片上传当前用 Mock（郭子骏的模型网关 / 上传模块未落地），
真实接入点见 recognition.py 与 images.py 顶部。
可直接 pytest 收集，也可用 python -m unittest 运行（当前环境未安装 pytest）。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.liuboyuan.container import build_services
from app.liuboyuan.images import MAX_IMAGE_BYTES
from app.liuboyuan.recognition import (
    FailingVisionAnalyzer,
    HeuristicVisionAnalyzer,
    InvalidTags,
    normalize_tags,
)

USER = "u_1"
OTHER = "u_2"


def jpeg(payload: bytes) -> bytes:
    """最小 JPEG：文件头 + 区分内容的载荷（HashImageEmbedder 用内容做向量）。"""
    return b"\xff\xd8\xff\xe0" + payload


def tags(**overrides) -> dict:
    data = {"category": "top", "color": "white"}
    data.update(overrides)
    return data


async def upload(services, *, user: str = USER, payload: bytes = b"photo") -> dict:
    response = await services.api.upload_image(user, jpeg(payload))
    assert response.status_code == 201, response.body
    assert set(response.body) == {"data", "request_id"}
    return response.body["data"]


async def analyze(services, image_id: str, *, user: str = USER, key: str = "k-1"):
    response = await services.api.analyze_garment(
        user, key, {"image_id": image_id}
    )
    assert response.status_code == 202, response.body
    job_id = response.body["data"]["job_id"]
    done = await services.jobs.wait(user, job_id, timeout=5)
    return response.body["data"], done


class RecognitionToWardrobeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.services = build_services(users=(USER, OTHER))
        self.api = self.services.api

    async def asyncTearDown(self) -> None:
        await self.services.jobs.wait_all(timeout=5)

    async def test_upload_recognize_correct_and_query(self) -> None:
        image = await upload(self.services, payload=b"white-tee")
        self.assertEqual(image["image_id"], "img_1")
        self.assertTrue(image["image_url"].startswith("/media/img_1"))

        submitted, done = await analyze(self.services, image["image_id"])

        # 02：202 时只有 Job，没有 result；查任务 200
        self.assertEqual(set(submitted), {"job_id", "type", "status", "stage", "result", "error", "warnings"})
        self.assertEqual(submitted["type"], "garment_analysis")
        self.assertIsNone(submitted["result"])
        self.assertIsNone(submitted["error"])
        self.assertEqual(submitted["warnings"], [])

        self.assertEqual(done.status, "succeeded")
        self.assertEqual(done.result["image_id"], image["image_id"])
        self.assertIn(done.result["tags"]["category"], ("top", "bottom", "dress", "shoes", "outerwear"))
        self.assertTrue(done.result["tags"]["color"])

        # 用户把识别结果改成自己的标签后确认入库
        created = await self.api.create_garment(
            USER,
            {
                "image_id": image["image_id"],
                "tags": tags(color="off-white", style_tags=["minimal", "soft"]),
            },
        )
        self.assertEqual(created.status_code, 201)
        garment = created.body["data"]
        self.assertEqual(garment["image_id"], image["image_id"])
        self.assertEqual(garment["image_url"], image["image_url"])
        self.assertEqual(garment["tags"]["color"], "off-white")
        self.assertEqual(garment["tags"]["style_tags"], ["minimal", "soft"])

        listed = await self.api.list_garments(USER, {})
        self.assertEqual(listed.status_code, 200)
        items = listed.body["data"]["items"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["garment_id"], garment["garment_id"])
        self.assertEqual(items[0]["tags"]["color"], "off-white")

    async def test_patch_replaces_the_whole_tags(self) -> None:
        image = await upload(self.services)
        created = await self.api.create_garment(
            USER,
            {
                "image_id": image["image_id"],
                "tags": tags(
                    color="white",
                    pattern="stripe",
                    fit="slim",
                    warmth="warm",
                    style_tags=["minimal", "soft"],
                ),
            },
        )
        garment_id = created.body["data"]["garment_id"]

        # 02：PATCH 提交完整标签；未提交的可选字段必须被清空（整体替换）
        patched = await self.api.update_garment(
            USER, garment_id, {"tags": {"category": "top", "color": "black"}}
        )
        self.assertEqual(patched.status_code, 200)
        updated = patched.body["data"]["tags"]
        self.assertEqual(updated["color"], "black")
        self.assertIsNone(updated["pattern"])
        self.assertIsNone(updated["fit"])
        self.assertIsNone(updated["warmth"])
        self.assertEqual(updated["style_tags"], [])
        # 不可变字段不受影响
        self.assertEqual(patched.body["data"]["garment_id"], garment_id)
        self.assertEqual(patched.body["data"]["image_id"], image["image_id"])

        again = await self.api.list_garments(USER, {})
        self.assertEqual(again.body["data"]["items"][0]["tags"]["color"], "black")
        self.assertEqual(again.body["data"]["items"][0]["tags"]["style_tags"], [])

    async def test_patch_requires_complete_tags(self) -> None:
        image = await upload(self.services)
        created = await self.api.create_garment(
            USER, {"image_id": image["image_id"], "tags": tags()}
        )
        garment_id = created.body["data"]["garment_id"]

        missing_tags = await self.api.update_garment(USER, garment_id, {})
        self.assertEqual(missing_tags.status_code, 422)
        self.assertEqual(missing_tags.body["error"]["code"], "INVALID_INPUT")

        missing_category = await self.api.update_garment(
            USER, garment_id, {"tags": {"color": "black"}}
        )
        self.assertEqual(missing_category.status_code, 422)
        self.assertEqual(missing_category.body["error"]["code"], "INVALID_TAGS")

        missing_color = await self.api.update_garment(
            USER, garment_id, {"tags": {"category": "top"}}
        )
        self.assertEqual(missing_color.status_code, 422)
        self.assertEqual(missing_color.body["error"]["code"], "INVALID_TAGS")

        unknown_field = await self.api.update_garment(
            USER,
            garment_id,
            {"tags": tags(brand="uniqlo")},
        )
        self.assertEqual(unknown_field.status_code, 422)
        self.assertEqual(unknown_field.body["error"]["code"], "INVALID_TAGS")

        # 失败的 PATCH 不改变已存标签
        listed = await self.api.list_garments(USER, {})
        self.assertEqual(listed.body["data"]["items"][0]["tags"]["color"], "white")

    async def test_create_requires_own_existing_image(self) -> None:
        missing = await self.api.create_garment(
            USER, {"image_id": "img_missing", "tags": tags()}
        )
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.body["error"]["code"], "IMAGE_NOT_FOUND")

        foreign_image = await upload(self.services, user=OTHER)
        stolen = await self.api.create_garment(
            USER, {"image_id": foreign_image["image_id"], "tags": tags()}
        )
        self.assertEqual(stolen.status_code, 404)
        self.assertEqual(stolen.body["error"]["code"], "IMAGE_NOT_FOUND")

    async def test_invalid_tags_are_rejected(self) -> None:
        image = await upload(self.services)
        for bad_tags in (
            {"category": "hat", "color": "red"},
            {"category": "top", "color": "   "},
            {"category": "top", "color": "red", "warmth": "hot"},
            {"category": "top", "color": "red", "style_tags": 5},
            {"color": "red"},
        ):
            response = await self.api.create_garment(
                USER, {"image_id": image["image_id"], "tags": bad_tags}
            )
            self.assertEqual(response.status_code, 422, bad_tags)
            self.assertEqual(response.body["error"]["code"], "INVALID_TAGS", bad_tags)

        # 一个都没入库
        listed = await self.api.list_garments(USER, {})
        self.assertEqual(listed.body["data"]["items"], [])

    async def test_category_filter_and_invalid_category(self) -> None:
        top_image = await upload(self.services, payload=b"top")
        bottom_image = await upload(self.services, payload=b"bottom")
        await self.api.create_garment(
            USER, {"image_id": top_image["image_id"], "tags": tags(category="top")}
        )
        await self.api.create_garment(
            USER,
            {
                "image_id": bottom_image["image_id"],
                "tags": tags(category="bottom", color="navy"),
            },
        )

        tops = await self.api.list_garments(USER, {"category": "top"})
        self.assertEqual(len(tops.body["data"]["items"]), 1)
        self.assertEqual(tops.body["data"]["items"][0]["tags"]["category"], "top")

        everything = await self.api.list_garments(USER, {})
        self.assertEqual(len(everything.body["data"]["items"]), 2)

        invalid = await self.api.list_garments(USER, {"category": "hat"})
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(invalid.body["error"]["code"], "INVALID_INPUT")

    async def test_user_isolation_across_wardrobe(self) -> None:
        image = await upload(self.services)
        created = await self.api.create_garment(
            USER, {"image_id": image["image_id"], "tags": tags()}
        )
        garment_id = created.body["data"]["garment_id"]

        other_list = await self.api.list_garments(OTHER, {})
        self.assertEqual(other_list.body["data"]["items"], [])

        other_patch = await self.api.update_garment(
            OTHER, garment_id, {"tags": tags(color="red")}
        )
        self.assertEqual(other_patch.status_code, 404)
        self.assertEqual(other_patch.body["error"]["code"], "GARMENT_NOT_FOUND")

        unknown_patch = await self.api.update_garment(
            USER, "gar_missing", {"tags": tags()}
        )
        self.assertEqual(unknown_patch.status_code, 404)
        self.assertEqual(unknown_patch.body["error"]["code"], "GARMENT_NOT_FOUND")

    async def test_analyze_validation_and_idempotency(self) -> None:
        image_a = await upload(self.services, payload=b"a")
        image_b = await upload(self.services, payload=b"b")

        missing_key = await self.api.analyze_garment(
            USER, None, {"image_id": image_a["image_id"]}
        )
        self.assertEqual(missing_key.status_code, 422)
        self.assertEqual(missing_key.body["error"]["code"], "INVALID_INPUT")

        unknown_image = await self.api.analyze_garment(
            USER, "k-404", {"image_id": "img_missing"}
        )
        self.assertEqual(unknown_image.status_code, 404)
        self.assertEqual(unknown_image.body["error"]["code"], "IMAGE_NOT_FOUND")

        foreign = await upload(self.services, user=OTHER)
        foreign_analyze = await self.api.analyze_garment(
            USER, "k-foreign", {"image_id": foreign["image_id"]}
        )
        self.assertEqual(foreign_analyze.status_code, 404)

        first = await self.api.analyze_garment(
            USER, "same-key", {"image_id": image_a["image_id"]}
        )
        replay = await self.api.analyze_garment(
            USER, "same-key", {"image_id": image_a["image_id"]}
        )
        self.assertEqual(first.status_code, 202)
        self.assertEqual(first.body["data"]["job_id"], replay.body["data"]["job_id"])

        conflict = await self.api.analyze_garment(
            USER, "same-key", {"image_id": image_b["image_id"]}
        )
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(conflict.body["error"]["code"], "IDEMPOTENCY_CONFLICT")

        # 幂等键按用户隔离：别人用同一个键不受影响
        others = await self.api.analyze_garment(
            OTHER, "same-key", {"image_id": foreign["image_id"]}
        )
        self.assertEqual(others.status_code, 202)
        self.assertNotEqual(
            others.body["data"]["job_id"], first.body["data"]["job_id"]
        )

    async def test_job_isolation_and_failed_job_readable(self) -> None:
        image = await upload(self.services)
        submitted, _done = await analyze(self.services, image["image_id"], key="k-job")
        job_id = submitted["job_id"]

        mine = await self.api.get_job(USER, job_id)
        self.assertEqual(mine.status_code, 200)
        self.assertEqual(mine.body["data"]["status"], "succeeded")

        other = await self.api.get_job(OTHER, job_id)
        self.assertEqual(other.status_code, 404)
        self.assertEqual(other.body["error"]["code"], "JOB_NOT_FOUND")

        unknown = await self.api.get_job(USER, "job_missing")
        self.assertEqual(unknown.status_code, 404)

    async def test_missing_or_unknown_user_is_401(self) -> None:
        missing = await self.api.list_garments(None, {})
        self.assertEqual(missing.status_code, 401)
        self.assertEqual(missing.body["error"]["code"], "UNAUTHORIZED")

        unknown = await self.api.list_garments("u_stranger", {})
        self.assertEqual(unknown.status_code, 401)

    async def test_upload_rejects_non_image_and_oversized(self) -> None:
        not_image = await self.api.upload_image(USER, b"plain text, not a photo")
        self.assertEqual(not_image.status_code, 422)
        self.assertEqual(not_image.body["error"]["code"], "INVALID_IMAGE")

        too_large = await self.api.upload_image(
            USER, b"\xff\xd8\xff" + b"0" * MAX_IMAGE_BYTES
        )
        self.assertEqual(too_large.status_code, 413)
        self.assertEqual(too_large.body["error"]["code"], "IMAGE_TOO_LARGE")

        png = await self.api.upload_image(
            USER, b"\x89PNG\r\n\x1a\n" + b"payload", content_type="image/png"
        )
        self.assertEqual(png.status_code, 201)
        self.assertTrue(png.body["data"]["image_url"].endswith(".png"))


class VisionFailureTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.services = build_services(
            users=(USER,), vision=FailingVisionAnalyzer("视觉模型 504")
        )
        self.api = self.services.api

    async def asyncTearDown(self) -> None:
        await self.services.jobs.wait_all(timeout=5)

    async def test_upstream_failure_marks_job_failed(self) -> None:
        image = await upload(self.services)
        submitted, done = await analyze(self.services, image["image_id"], key="k-fail")
        self.assertEqual(submitted["status"] in ("queued", "running", "succeeded"), True)
        self.assertEqual(done.status, "failed")
        self.assertIsNone(done.result)
        self.assertEqual(done.error.code, "UPSTREAM_FAILED")
        self.assertTrue(done.error.retryable)
        self.assertEqual(done.stage, "failed")

        # 02：失败的 Job 查询仍返回 200，错误从 Job.error 读
        fetched = await self.api.get_job(USER, done.job_id)
        self.assertEqual(fetched.status_code, 200)
        self.assertEqual(fetched.body["data"]["status"], "failed")
        self.assertEqual(fetched.body["data"]["error"]["code"], "UPSTREAM_FAILED")

        # 失败任务不写入衣橱
        listed = await self.api.list_garments(USER, {})
        self.assertEqual(listed.body["data"]["items"], [])


class InvalidModelOutputTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.services = build_services(users=(USER,), vision=HeuristicVisionAnalyzer())
        self.api = self.services.api

    async def asyncTearDown(self) -> None:
        await self.services.jobs.wait_all(timeout=5)

    async def test_unusable_model_tags_fail_the_job(self) -> None:
        image = await upload(self.services)
        # 模型返回 02 之外的品类：属于外部服务问题，不改写用户的衣橱
        self.services.vision.hint(image["image_id"], {"category": "hat", "color": "red"})
        _submitted, done = await analyze(self.services, image["image_id"], key="k-bad")
        self.assertEqual(done.status, "failed")
        self.assertEqual(done.error.code, "UPSTREAM_FAILED")
        self.assertIn("识别结果不可用", done.error.message)
        self.assertTrue(done.error.retryable)


class NormalizeTagsTests(unittest.TestCase):
    """Tags 是识别与衣橱共用的唯一校验口，单测锁死 02 的规则。"""

    def test_minimal_tags_get_defaults(self) -> None:
        parsed = normalize_tags({"category": "top", "color": " White "})
        self.assertEqual(parsed.category, "top")
        self.assertEqual(parsed.color, "White")
        self.assertIsNone(parsed.pattern)
        self.assertIsNone(parsed.fit)
        self.assertIsNone(parsed.warmth)
        self.assertEqual(parsed.style_tags, [])

    def test_style_tags_are_trimmed_and_deduped(self) -> None:
        parsed = normalize_tags(
            {"category": "dress", "color": "red", "style_tags": [" minimal ", "minimal", "", "soft"]}
        )
        self.assertEqual(parsed.style_tags, ["minimal", "soft"])

    def test_string_style_tag_is_accepted(self) -> None:
        parsed = normalize_tags({"category": "shoes", "color": "black", "style_tags": "minimal"})
        self.assertEqual(parsed.style_tags, ["minimal"])

    def test_long_text_is_truncated(self) -> None:
        parsed = normalize_tags({"category": "top", "color": "c" * 120})
        self.assertEqual(len(parsed.color), 40)

    def test_rejects_bad_input(self) -> None:
        for bad in (
            None,
            "top",
            ["top", "white"],
            {"category": "top"},
            {"color": "white"},
            {"category": "top", "color": "white", "unknown": 1},
            {"category": "top", "color": "white", "warmth": "hot"},
            {"category": "top", "color": "white", "style_tags": 5},
            {"category": "TOP", "color": "white"},
        ):
            with self.assertRaises(InvalidTags, msg=repr(bad)):
                normalize_tags(bad)

    def test_tags_instance_is_revalidated(self) -> None:
        parsed = normalize_tags({"category": "top", "color": "white"})
        self.assertEqual(normalize_tags(parsed).color, "white")


if __name__ == "__main__":
    unittest.main()
