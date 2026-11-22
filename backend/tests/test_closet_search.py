"""以图搜衣 ``closet.search(user_id, image_id, limit=10)`` 的契约与边界。

覆盖 02/03 的验收点：
    * 服装图与参考图用**同一个**图像 Embedding 实例（同一模型空间）；
    * 只召回当前用户**已确认入库**的单品（识别过但没确认的图片不算）；
    * 按品类召回，每品类最多 limit 件，分数降序、同分稳定；
    * 参考图跨用户 -> 404；衣橱为空 -> 空候选且无 warnings；
    * 向量缺失 -> 懒补建；补建失败 -> 从候选剔除并给 warnings（search 契约不变）；
    * 参考图向量化 / 向量检索不可用 -> 502 UPSTREAM_FAILED（retryable）；
    * 改品类后向量按新品类重建；只改颜色不重建（向量同步接入点 2）；
    * InMemoryGarmentVectorStore 的用户/品类隔离与余弦相似度边界。

真实接入点：pgvector 实现 GarmentVectorStore（见 wardrobe.py 顶部）。
可直接 pytest 收集，也可用 python -m unittest 运行。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.liuboyuan.closet_search import (
    Candidate,
    ClosetSearchService,
    SearchUpstreamError,
)
from app.liuboyuan.container import build_services
from app.liuboyuan.wardrobe import (
    HashImageEmbedder,
    InMemoryGarmentVectorStore,
    WardrobeService,
    cosine_similarity,
)

USER = "u_1"
OTHER = "u_2"


def jpeg(payload: bytes) -> bytes:
    return b"\xff\xd8\xff\xe0" + payload


async def upload(services, *, user: str = USER, payload: bytes = b"photo") -> dict:
    response = await services.api.upload_image(user, jpeg(payload))
    assert response.status_code == 201, response.body
    return response.body["data"]


async def add_garment(
    services,
    *,
    user: str = USER,
    payload: bytes = b"photo",
    category: str = "top",
    color: str = "white",
    **tag_overrides,
) -> dict:
    """走真实入库链路（上传 -> POST /garments），保证衣橱里是“已确认”数据。"""
    image = await upload(services, user=user, payload=payload)
    body = {"category": category, "color": color, **tag_overrides}
    response = await services.api.create_garment(
        user, {"image_id": image["image_id"], "tags": body}
    )
    assert response.status_code == 201, response.body
    return response.body["data"]


class SpyEmbedder:
    """记录调用顺序，验证服装图与参考图确实走同一个实例。"""

    def __init__(self) -> None:
        self.inner = HashImageEmbedder()
        self.calls: list[str] = []

    @property
    def model(self) -> str:
        return self.inner.model

    @property
    def dimension(self) -> int:
        return self.inner.dimension

    async def embed_image(self, image, content):  # noqa: ANN001
        self.calls.append(image.image_id)
        return await self.inner.embed_image(image, content)


class SpyVectorStore(InMemoryGarmentVectorStore):
    def __init__(self) -> None:
        super().__init__()
        self.upserts: list[tuple[str, str]] = []

    async def upsert_garment(self, user_id, *, garment_id, category, vector, model):
        self.upserts.append((garment_id, category))
        await super().upsert_garment(
            user_id,
            garment_id=garment_id,
            category=category,
            vector=vector,
            model=model,
        )


class BrokenUpsertStore(InMemoryGarmentVectorStore):
    """读取正常、只有写入失败（模拟 pgvector 只读/故障）。"""

    async def upsert_garment(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("pgvector 写入失败")


class BrokenSearchStore(InMemoryGarmentVectorStore):
    async def search_garments(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("pgvector 查询失败")


class BrokenIndexedStore(InMemoryGarmentVectorStore):
    async def indexed_garments(self, user_id: str) -> set[str]:
        raise RuntimeError("pgvector 索引不可用")


class BrokenWardrobeRepository:
    """衣橱仓储故障（数据库不可用）。"""

    async def add(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("db down")

    async def list(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("db down")

    async def get(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("db down")

    async def replace(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("db down")


class BoomEmbedder:
    model = "boom-embed"
    dimension = 8

    async def embed_image(self, image, content):  # noqa: ANN001
        raise RuntimeError("embedding 服务 502")


class SameEmbedderTests(unittest.IsolatedAsyncioTestCase):
    async def test_garment_and_reference_share_one_embedder(self) -> None:
        spy = SpyEmbedder()
        services = build_services(users=(USER,), embedder=spy)
        garment = await add_garment(services, payload=b"photo-white-tee")
        reference = await upload(services, payload=b"photo-white-tee")

        candidates = await services.closet.search(USER, reference["image_id"], limit=10)

        # 入库时用该实例生成服装向量，检索时用**同一实例**生成参考图向量
        self.assertEqual(spy.calls, [garment["image_id"], reference["image_id"]])
        self.assertEqual([item.garment_id for item in candidates], [garment["garment_id"]])
        self.assertEqual(candidates[0].score, 1.0)

    async def test_different_photo_scores_lower(self) -> None:
        services = build_services(users=(USER,))
        await add_garment(services, payload=b"a-white-tee")
        reference = await upload(services, payload=b"a-completely-different-photo")
        candidates = await services.closet.search(USER, reference["image_id"])
        self.assertEqual(len(candidates), 1)
        self.assertLess(candidates[0].score, 0.99)


class PerCategoryLimitTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.services = build_services(users=(USER,))
        self.tops = [
            await add_garment(
                self.services, payload=f"top-{index}".encode(), category="top"
            )
            for index in range(3)
        ]
        self.bottoms = [
            await add_garment(
                self.services,
                payload=f"bottom-{index}".encode(),
                category="bottom",
                color="navy",
            )
            for index in range(3)
        ]
        self.reference = await upload(self.services, payload=b"top-0")
        self.top_ids = {item["garment_id"] for item in self.tops}
        self.bottom_ids = {item["garment_id"] for item in self.bottoms}

    async def asyncTearDown(self) -> None:
        await self.services.jobs.wait_all(timeout=5)

    async def test_limit_applies_per_category(self) -> None:
        candidates = await self.services.closet.search(
            USER, self.reference["image_id"], limit=2
        )
        self.assertEqual(len(candidates), 4)  # 2 个品类 x 每品类 2 件
        ids = [item.garment_id for item in candidates]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len([i for i in ids if i in self.top_ids]), 2)
        self.assertEqual(len([i for i in ids if i in self.bottom_ids]), 2)

    async def test_reference_itself_ranks_first(self) -> None:
        candidates = await self.services.closet.search(
            USER, self.reference["image_id"], limit=1
        )
        self.assertEqual(len(candidates), 2)  # 每个品类各 1 件
        tops = [c for c in candidates if c.garment_id in self.top_ids]
        self.assertEqual(len(tops), 1)
        self.assertEqual(tops[0].garment_id, self.tops[0]["garment_id"])
        self.assertEqual(tops[0].score, 1.0)

    async def test_limit_is_clamped(self) -> None:
        zero = await self.services.closet.search(USER, self.reference["image_id"], limit=0)
        self.assertEqual(len(zero), 2)  # 至少 1 件/品类
        huge = await self.services.closet.search(
            USER, self.reference["image_id"], limit=999
        )
        self.assertEqual(len(huge), 6)  # 受衣橱数量限制
        broken = await self.services.closet.search(
            USER, self.reference["image_id"], limit="not-a-number"  # type: ignore[arg-type]
        )
        self.assertEqual(len(broken), 6)  # 非法 limit 退回默认 10

    async def test_default_limit_is_ten_and_positional_call_works(self) -> None:
        candidates = await self.services.closet.search(USER, self.reference["image_id"])
        self.assertEqual(len(candidates), 6)

    async def test_scores_descend_within_category(self) -> None:
        candidates = await self.services.closet.search(
            USER, self.reference["image_id"], limit=3
        )
        top_scores = [c.score for c in candidates if c.garment_id in self.top_ids]
        self.assertEqual(top_scores, sorted(top_scores, reverse=True))
        bottom_scores = [c.score for c in candidates if c.garment_id in self.bottom_ids]
        self.assertEqual(bottom_scores, sorted(bottom_scores, reverse=True))

    async def test_candidate_shape(self) -> None:
        candidates = await self.services.closet.search(USER, self.reference["image_id"])
        for item in candidates:
            self.assertIsInstance(item, Candidate)
            self.assertEqual(set(item.model_dump()), {"garment_id", "image_url", "score"})
            self.assertIsInstance(item.score, float)
            self.assertGreaterEqual(item.score, -1.0)
            self.assertLessEqual(item.score, 1.0)
            self.assertTrue(item.image_url.startswith("/media/"))


class CategoryChangeTests(unittest.IsolatedAsyncioTestCase):
    async def test_changing_category_rebuilds_the_vector(self) -> None:
        store = SpyVectorStore()
        services = build_services(users=(USER,), vectors=store)
        garment = await add_garment(services, payload=b"photo-x", category="top")
        reference = await upload(services, payload=b"photo-x")
        self.assertEqual(len(store.upserts), 1)  # 入库即同步（接入点 1）

        patched = await services.api.update_garment(
            USER,
            garment["garment_id"],
            {"tags": {"category": "outerwear", "color": "white"}},
        )
        self.assertEqual(patched.status_code, 200)
        self.assertEqual(len(store.upserts), 2)  # 换品类 -> 重建（接入点 2）
        self.assertEqual(store.upserts[-1][1], "outerwear")

        candidates = await services.closet.search(USER, reference["image_id"])
        self.assertEqual([item.garment_id for item in candidates], [garment["garment_id"]])
        self.assertEqual(candidates[0].score, 1.0)

        # 旧品类不再命中，新品类命中（向量按品类召回）
        tops = await store.search_garments(
            USER, category="top", vector=[0.0], limit=10
        )
        self.assertEqual(tops, [])
        self.assertEqual(await store.indexed_garments(USER), {garment["garment_id"]})

    async def test_color_only_patch_keeps_the_vector(self) -> None:
        store = SpyVectorStore()
        services = build_services(users=(USER,), vectors=store)
        garment = await add_garment(services, payload=b"photo-y")
        reference = await upload(services, payload=b"photo-y")

        patched = await services.api.update_garment(
            USER,
            garment["garment_id"],
            {"tags": {"category": "top", "color": "black"}},
        )
        self.assertEqual(patched.status_code, 200)
        self.assertEqual(len(store.upserts), 1)  # 未重建

        candidates = await services.closet.search(USER, reference["image_id"])
        self.assertEqual(candidates[0].score, 1.0)


class VectorAvailabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_vector_triggers_lazy_rebuild(self) -> None:
        store = InMemoryGarmentVectorStore()
        services = build_services(users=(USER,), vectors=store)
        garment = await add_garment(services, payload=b"lazy-photo")
        await store.clear(USER)  # 模拟 pgvector 数据丢失
        reference = await upload(services, payload=b"lazy-photo")

        candidates, warnings = await services.closet.search_with_warnings(
            USER, reference["image_id"]
        )
        self.assertEqual(warnings, [])
        self.assertEqual([item.garment_id for item in candidates], [garment["garment_id"]])
        self.assertEqual(candidates[0].score, 1.0)  # 补建后仍指向同一张服装图

    async def test_rebuild_failure_reports_warning_and_skips(self) -> None:
        services = build_services(users=(USER,), vectors=BrokenUpsertStore())
        missing = await add_garment(services, payload=b"missing-photo")
        reference = await upload(services, payload=b"missing-photo")

        candidates, warnings = await services.closet.search_with_warnings(
            USER, reference["image_id"]
        )
        self.assertEqual(candidates, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("尚未生成图像向量", warnings[0])
        self.assertIn(missing["garment_id"], warnings[0])

        # 契约方法结构不变（不因 warnings 改变返回类型）
        plain = await services.closet.search(USER, reference["image_id"])
        self.assertEqual(plain, [])

    async def test_reference_embedding_failure_is_502(self) -> None:
        services = build_services(users=(USER,), embedder=BoomEmbedder())
        garment = await add_garment(services, payload=b"photo")
        self.assertTrue(garment["garment_id"])
        reference = await upload(services, payload=b"photo")

        with self.assertRaises(SearchUpstreamError) as raised:
            await services.closet.search(USER, reference["image_id"])
        self.assertEqual(raised.exception.status_code, 502)
        self.assertEqual(raised.exception.code, "UPSTREAM_FAILED")
        self.assertTrue(raised.exception.retryable)

    async def test_vector_search_failure_is_502(self) -> None:
        services = build_services(users=(USER,), vectors=BrokenSearchStore())
        await add_garment(services, payload=b"photo")
        reference = await upload(services, payload=b"photo")

        with self.assertRaises(SearchUpstreamError) as raised:
            await services.closet.search(USER, reference["image_id"])
        self.assertEqual(raised.exception.status_code, 502)
        self.assertIn("向量检索失败", raised.exception.message)

    async def test_index_lookup_failure_is_502(self) -> None:
        services = build_services(users=(USER,), vectors=BrokenIndexedStore())
        await add_garment(services, payload=b"photo")
        reference = await upload(services, payload=b"photo")

        with self.assertRaises(SearchUpstreamError) as raised:
            await services.closet.search(USER, reference["image_id"])
        self.assertEqual(raised.exception.status_code, 502)
        self.assertIn("向量索引查询失败", raised.exception.message)

    async def test_wardrobe_query_failure_is_502(self) -> None:
        services = build_services(users=(USER,))
        wardrobe = WardrobeService(
            images=services.images,
            repository=BrokenWardrobeRepository(),
            embedder=services.embedder,
            vectors=services.vectors,
        )
        closet = ClosetSearchService(wardrobe=wardrobe, images=services.images)
        reference = await upload(services, payload=b"photo")

        with self.assertRaises(SearchUpstreamError) as raised:
            await closet.search(USER, reference["image_id"])
        self.assertEqual(raised.exception.status_code, 502)
        self.assertIn("衣橱查询失败", raised.exception.message)


class IsolationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.services = build_services(users=(USER, OTHER))
        self.mine = await add_garment(
            self.services, user=USER, payload=b"mine-tee", category="top"
        )
        self.theirs = await add_garment(
            self.services, user=OTHER, payload=b"their-tee", category="top", color="red"
        )

    async def asyncTearDown(self) -> None:
        await self.services.jobs.wait_all(timeout=5)

    async def test_foreign_reference_image_is_404(self) -> None:
        foreign = await upload(self.services, user=OTHER, payload=b"their-ref")

        from app.liuboyuan.images import ImageNotFound

        with self.assertRaises(ImageNotFound) as raised:
            await self.services.closet.search(USER, foreign["image_id"])
        self.assertEqual(raised.exception.status_code, 404)
        self.assertEqual(raised.exception.code, "IMAGE_NOT_FOUND")

    async def test_only_own_confirmed_garments_are_recalled(self) -> None:
        reference = await upload(self.services, user=USER, payload=b"mine-tee")
        candidates = await self.services.closet.search(USER, reference["image_id"])
        ids = [item.garment_id for item in candidates]
        self.assertEqual(ids, [self.mine["garment_id"]])
        self.assertNotIn(self.theirs["garment_id"], ids)

        their_reference = await upload(self.services, user=OTHER, payload=b"their-tee")
        their_candidates = await self.services.closet.search(
            OTHER, their_reference["image_id"]
        )
        self.assertEqual(
            [item.garment_id for item in their_candidates],
            [self.theirs["garment_id"]],
        )

    async def test_empty_wardrobe_returns_empty_without_warning(self) -> None:
        services = build_services(users=(USER,))
        reference = await upload(services, payload=b"no-clothes-yet")
        candidates, warnings = await services.closet.search_with_warnings(
            USER, reference["image_id"]
        )
        self.assertEqual(candidates, [])
        self.assertEqual(warnings, [])

    async def test_analyzed_but_unconfirmed_photo_is_not_recalled(self) -> None:
        image = await upload(self.services, user=USER, payload=b"never-confirmed")
        response = await self.services.api.analyze_garment(
            USER, "k-unconfirmed", {"image_id": image["image_id"]}
        )
        self.assertEqual(response.status_code, 202)
        await self.services.jobs.wait(
            USER, response.body["data"]["job_id"], timeout=5
        )

        candidates = await self.services.closet.search(USER, image["image_id"])
        self.assertEqual(
            [item.garment_id for item in candidates], [self.mine["garment_id"]]
        )
        self.assertNotIn(
            self.theirs["garment_id"], [item.garment_id for item in candidates]
        )


class VectorStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_user_and_category_isolation(self) -> None:
        store = InMemoryGarmentVectorStore()
        vector = [1.0, 0.0]
        await store.upsert_garment(
            USER, garment_id="g_1", category="top", vector=vector, model="m"
        )
        await store.upsert_garment(
            OTHER, garment_id="g_2", category="top", vector=vector, model="m"
        )

        mine = await store.search_garments(
            USER, category="top", vector=vector, limit=10
        )
        self.assertEqual([item[0] for item in mine], ["g_1"])
        self.assertEqual(mine[0][1], 1.0)

        bottoms = await store.search_garments(
            USER, category="bottom", vector=vector, limit=10
        )
        self.assertEqual(bottoms, [])
        self.assertEqual(await store.indexed_garments(USER), {"g_1"})

    async def test_upsert_overwrites_category(self) -> None:
        store = InMemoryGarmentVectorStore()
        vector = [0.0, 1.0]
        await store.upsert_garment(
            USER, garment_id="g_1", category="top", vector=vector, model="m"
        )
        await store.upsert_garment(
            USER, garment_id="g_1", category="outerwear", vector=vector, model="m"
        )
        self.assertEqual(
            await store.search_garments(USER, category="top", vector=vector, limit=10),
            [],
        )
        moved = await store.search_garments(
            USER, category="outerwear", vector=vector, limit=10
        )
        self.assertEqual([item[0] for item in moved], ["g_1"])

    async def test_dimension_mismatch_scores_zero(self) -> None:
        store = InMemoryGarmentVectorStore()
        await store.upsert_garment(
            USER, garment_id="g_1", category="top", vector=[1.0, 0.0], model="m"
        )
        scored = await store.search_garments(
            USER, category="top", vector=[1.0, 0.0, 0.0], limit=10
        )
        self.assertEqual(scored, [("g_1", 0.0)])

    def test_cosine_similarity_edges(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)
        self.assertEqual(cosine_similarity([1.0], [1.0, 0.0]), 0.0)
        self.assertEqual(cosine_similarity([], []), 0.0)
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 1.0]), 0.0)


if __name__ == "__main__":
    unittest.main()
