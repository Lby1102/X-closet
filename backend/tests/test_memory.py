"""偏好记忆验证（刘博远）。

覆盖 02/03/04 的验收点：
    - memory.get(user_id) 返回 03 规定的 Profile（explicit / learned_weights /
      inspiration_image_urls / version），画像取最近三张个人喜好图
    - memory.update(user_id, feedback_id) 返回 {"profile_version": int}
    - 反馈未入库时报 FEEDBACK_NOT_FOUND（03：更新须等 Feedback 入库）
    - feedback_id 去重（重复点击 / 任务重放只生效一次）
    - 按 user_id 串行；近 30 天 style_tags 计权（like +1 / dislike -1）
    - 权重与版本同一事务提交（提交失败不产生半写）
    - 显式禁忌优先于学习到的偏好（写入时与读取时都生效）

反馈/偏好来源当前用 Mock 驱动（何梓森模块未落地），真实接入点见 memory.py 顶部。
可直接用 pytest 收集，也可用 python -m unittest 运行。
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.liuboyuan.inspirations import (
    InMemoryInspirationRepository,
    InspirationsService,
)
from app.liuboyuan.memory import (
    FeedbackNotReady,
    InMemoryProfileStore,
    MemoryError,
    MemoryService,
    MockFeedbackSource,
    MockPreferencesSource,
    Preferences,
    StoreConflict,
    compute_learned_weights,
)

BASE = datetime(2025, 5, 1, 12, 0, tzinfo=timezone.utc)


class MemoryTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.clock = BASE
        self.feedback = MockFeedbackSource()
        self.preferences = MockPreferencesSource()
        self.store = InMemoryProfileStore()
        self.inspirations = InspirationsService(
            InMemoryInspirationRepository(), now=lambda: self.clock
        )
        self.memory = MemoryService(
            store=self.store,
            feedback=self.feedback,
            preferences=self.preferences,
            inspirations=self.inspirations,
            now=lambda: self.clock,
        )

    def add_feedback(
        self,
        feedback_id: str,
        *,
        action: str = "like",
        style_tags: list[str] | None = None,
        user_id: str = "u1",
        created_at: datetime | None = None,
    ):
        return self.feedback.add_feedback(
            feedback_id,
            user_id,
            action=action,  # type: ignore[arg-type]
            style_tags=style_tags or [],
            created_at=created_at or self.clock,
        )


# --------------------------------------------------------------------------- #
# memory.get
# --------------------------------------------------------------------------- #
class TestGetProfile(MemoryTestCase):
    async def test_empty_profile_matches_v1_shape(self) -> None:
        profile = await self.memory.get("u1")

        self.assertEqual(
            set(profile.model_dump().keys()),
            {"explicit", "learned_weights", "inspiration_image_urls", "version"},
        )
        self.assertEqual(profile.learned_weights, {})
        self.assertEqual(profile.inspiration_image_urls, [])
        self.assertEqual(profile.version, 0)
        self.assertEqual(profile.explicit.liked_styles, [])
        self.assertEqual(profile.explicit.avoid_tags, [])

    async def test_explicit_preferences_passed_through(self) -> None:
        self.preferences.set("u1", liked_styles=["minimal"], avoid_tags=["sexy"])

        profile = await self.memory.get("u1")

        self.assertEqual(profile.explicit.liked_styles, ["minimal"])
        self.assertEqual(profile.explicit.avoid_tags, ["sexy"])

    async def test_profile_uses_latest_three_inspirations_desc(self) -> None:
        for index in range(4):
            self.clock = BASE + timedelta(minutes=index + 1)
            await self.inspirations.create("u1", f"img_{index}", tags=["minimal"])

        profile = await self.memory.get("u1")

        self.assertEqual(
            profile.inspiration_image_urls,
            ["/media/img_3.jpg", "/media/img_2.jpg", "/media/img_1.jpg"],
        )

    async def test_profile_without_inspirations_service_is_empty(self) -> None:
        memory = MemoryService(
            store=self.store,
            feedback=self.feedback,
            preferences=self.preferences,
            inspirations=None,
        )
        profile = await memory.get("u2")
        self.assertEqual(profile.inspiration_image_urls, [])

    async def test_blank_user_id_rejected(self) -> None:
        with self.assertRaises(MemoryError) as ctx:
            await self.memory.get("  ")
        self.assertEqual(ctx.exception.code, "INVALID_FIELD")


# --------------------------------------------------------------------------- #
# memory.update：计权与版本
# --------------------------------------------------------------------------- #
class TestUpdateWeights(MemoryTestCase):
    async def test_like_adds_one_and_bumps_version(self) -> None:
        self.add_feedback("fb_1", style_tags=["minimal"])

        result = await self.memory.update("u1", "fb_1")

        self.assertEqual(result, {"profile_version": 1})
        profile = await self.memory.get("u1")
        self.assertEqual(profile.version, 1)
        self.assertEqual(profile.learned_weights, {"minimal": 1.0})

    async def test_dislike_subtracts_one(self) -> None:
        self.add_feedback("fb_1", action="like", style_tags=["minimal"])
        self.add_feedback("fb_2", action="dislike", style_tags=["minimal"])

        await self.memory.update("u1", "fb_1")
        result = await self.memory.update("u1", "fb_2")

        self.assertEqual(result, {"profile_version": 2})
        profile = await self.memory.get("u1")
        # +1 与 -1 抵消后不再保留该标签
        self.assertEqual(profile.learned_weights, {})

    async def test_multiple_tags_and_accumulation(self) -> None:
        self.add_feedback("fb_1", style_tags=["minimal", "clean"])
        self.add_feedback("fb_2", style_tags=["minimal"])

        await self.memory.update("u1", "fb_1")
        await self.memory.update("u1", "fb_2")

        profile = await self.memory.get("u1")
        self.assertEqual(profile.learned_weights, {"minimal": 2.0, "clean": 1.0})
        self.assertEqual(profile.version, 2)

    def test_duplicate_tags_in_one_feedback_count_once(self) -> None:
        from app.liuboyuan.memory import FeedbackRecord

        record = FeedbackRecord(
            feedback_id="fb_x",
            user_id="u1",
            outfit_id="outfit_1",
            action="like",
            style_tags=["Minimal", "minimal", "MINIMAL", "clean"],
            created_at=BASE,
        )

        self.assertEqual(
            compute_learned_weights([record]), {"minimal": 1.0, "clean": 1.0}
        )


# --------------------------------------------------------------------------- #
# 近 30 天窗口
# --------------------------------------------------------------------------- #
class TestFeedbackWindow(MemoryTestCase):
    async def test_feedback_older_than_30_days_excluded(self) -> None:
        self.add_feedback(
            "fb_old",
            style_tags=["vintage"],
            created_at=BASE - timedelta(days=31),
        )
        self.add_feedback(
            "fb_new",
            style_tags=["minimal"],
            created_at=BASE - timedelta(days=1),
        )

        await self.memory.update("u1", "fb_new")

        profile = await self.memory.get("u1")
        self.assertEqual(profile.learned_weights, {"minimal": 1.0})

    async def test_feedback_inside_window_included(self) -> None:
        self.add_feedback(
            "fb_edge",
            style_tags=["vintage"],
            created_at=BASE - timedelta(days=29),
        )

        await self.memory.update("u1", "fb_edge")

        profile = await self.memory.get("u1")
        self.assertEqual(profile.learned_weights, {"vintage": 1.0})

    async def test_old_feedback_dropped_when_window_recomputed(self) -> None:
        self.add_feedback(
            "fb_old",
            style_tags=["vintage"],
            created_at=BASE - timedelta(days=40),
        )
        self.add_feedback("fb_new", style_tags=["minimal"], created_at=BASE)

        await self.memory.update("u1", "fb_old")
        await self.memory.update("u1", "fb_new")

        profile = await self.memory.get("u1")
        self.assertEqual(profile.learned_weights, {"minimal": 1.0})
        self.assertEqual(profile.version, 2)


# --------------------------------------------------------------------------- #
# feedback_id 去重
# --------------------------------------------------------------------------- #
class TestFeedbackIdempotency(MemoryTestCase):
    async def test_replay_same_feedback_id_changes_nothing(self) -> None:
        self.add_feedback("fb_1", style_tags=["minimal"])

        first = await self.memory.update("u1", "fb_1")
        second = await self.memory.update("u1", "fb_1")
        third = await self.memory.update("u1", "fb_1")

        self.assertEqual(first, {"profile_version": 1})
        self.assertEqual(second, {"profile_version": 1})
        self.assertEqual(third, {"profile_version": 1})
        profile = await self.memory.get("u1")
        self.assertEqual(profile.learned_weights, {"minimal": 1.0})
        self.assertEqual(profile.version, 1)

    async def test_missing_feedback_is_not_ready(self) -> None:
        with self.assertRaises(FeedbackNotReady) as ctx:
            await self.memory.update("u1", "fb_missing")

        error = ctx.exception.to_error()
        self.assertEqual(error["code"], "FEEDBACK_NOT_FOUND")
        self.assertFalse(error["retryable"])
        self.assertEqual(ctx.exception.status_code, 422)

    async def test_other_users_feedback_rejected(self) -> None:
        self.add_feedback("fb_other", style_tags=["minimal"], user_id="u2")

        with self.assertRaises(FeedbackNotReady):
            await self.memory.update("u1", "fb_other")
        # 未产生任何画像写入
        self.assertEqual((await self.memory.get("u1")).version, 0)

    async def test_blank_feedback_id_rejected(self) -> None:
        with self.assertRaises(MemoryError) as ctx:
            await self.memory.update("u1", "   ")
        self.assertEqual(ctx.exception.code, "INVALID_FIELD")


# --------------------------------------------------------------------------- #
# 显式禁忌优先
# --------------------------------------------------------------------------- #
class TestExplicitPriority(MemoryTestCase):
    async def test_avoid_tag_cannot_become_positive(self) -> None:
        self.preferences.set("u1", liked_styles=["minimal"], avoid_tags=["sexy"])
        self.add_feedback("fb_1", style_tags=["minimal", "sexy"])

        await self.memory.update("u1", "fb_1")

        profile = await self.memory.get("u1")
        self.assertEqual(profile.learned_weights["minimal"], 1.0)
        self.assertEqual(profile.learned_weights["sexy"], 0.0)

    async def test_avoid_tag_negative_is_kept(self) -> None:
        self.preferences.set("u1", avoid_tags=["sexy"])
        self.add_feedback("fb_1", action="dislike", style_tags=["sexy"])

        await self.memory.update("u1", "fb_1")

        profile = await self.memory.get("u1")
        self.assertEqual(profile.learned_weights["sexy"], -1.0)

    async def test_preference_change_applies_immediately_on_read(self) -> None:
        self.add_feedback("fb_1", style_tags=["minimal"])
        await self.memory.update("u1", "fb_1")
        self.assertEqual((await self.memory.get("u1")).learned_weights["minimal"], 1.0)

        # 用户中途把 minimal 设为明确禁忌：读取时立即压制正权重（版本不变）
        self.preferences.set("u1", avoid_tags=["minimal"])

        profile = await self.memory.get("u1")
        self.assertEqual(profile.learned_weights["minimal"], 0.0)
        self.assertEqual(profile.version, 1)


# --------------------------------------------------------------------------- #
# 按用户串行
# --------------------------------------------------------------------------- #
class TestSerialUpdates(MemoryTestCase):
    async def test_concurrent_updates_for_same_user_are_serialized(self) -> None:
        for index in range(5):
            self.add_feedback(f"fb_{index}", style_tags=["minimal"])

        results = await asyncio.gather(
            *(self.memory.update("u1", f"fb_{index}") for index in range(5))
        )

        versions = sorted(item["profile_version"] for item in results)
        self.assertEqual(versions, [1, 2, 3, 4, 5])
        profile = await self.memory.get("u1")
        self.assertEqual(profile.version, 5)
        self.assertEqual(profile.learned_weights, {"minimal": 5.0})

    async def test_users_are_independent(self) -> None:
        self.add_feedback("fb_u1", style_tags=["minimal"], user_id="u1")
        self.add_feedback("fb_u2", style_tags=["streetwear"], user_id="u2")

        await asyncio.gather(
            self.memory.update("u1", "fb_u1"),
            self.memory.update("u2", "fb_u2"),
        )

        profile_u1 = await self.memory.get("u1")
        profile_u2 = await self.memory.get("u2")
        self.assertEqual(profile_u1.learned_weights, {"minimal": 1.0})
        self.assertEqual(profile_u2.learned_weights, {"streetwear": 1.0})
        self.assertEqual(profile_u1.version, 1)
        self.assertEqual(profile_u2.version, 1)


# --------------------------------------------------------------------------- #
# 权重与版本同事务提交
# --------------------------------------------------------------------------- #
class TestTransaction(MemoryTestCase):
    async def test_failed_commit_leaves_no_partial_write(self) -> None:
        class ExplodingStore(InMemoryProfileStore):
            async def commit(self, user_id, state, expected_version):
                raise RuntimeError("commit failed")

        self.store = ExplodingStore()
        memory = MemoryService(
            store=self.store,
            feedback=self.feedback,
            preferences=self.preferences,
            inspirations=self.inspirations,
            now=lambda: self.clock,
        )
        self.add_feedback("fb_1", style_tags=["minimal"])

        with self.assertRaises(RuntimeError):
            await memory.update("u1", "fb_1")

        state = await self.store.load("u1")
        self.assertEqual(state.version, 0)
        self.assertEqual(state.learned_weights, {})
        self.assertEqual(state.applied_feedback_ids, [])
        profile = await memory.get("u1")
        self.assertEqual(profile.version, 0)

    async def test_version_conflict_is_retried(self) -> None:
        class RacingStore(InMemoryProfileStore):
            """第一次提交模拟版本竞争，之后正常。"""

            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            async def commit(self, user_id, state, expected_version):
                self.calls += 1
                if self.calls == 1:
                    raise StoreConflict("racing writer")
                return await super().commit(user_id, state, expected_version)

        self.store = RacingStore()
        memory = MemoryService(
            store=self.store,
            feedback=self.feedback,
            preferences=self.preferences,
            inspirations=self.inspirations,
            now=lambda: self.clock,
        )
        self.add_feedback("fb_1", style_tags=["minimal"])

        result = await memory.update("u1", "fb_1")

        self.assertEqual(result, {"profile_version": 1})
        self.assertEqual(self.store.calls, 2)


# --------------------------------------------------------------------------- #
# 与喜好图片的联动（画像取最近三张）
# --------------------------------------------------------------------------- #
class TestProfileInspirationLink(MemoryTestCase):
    async def test_collected_inspirations_feed_profile(self) -> None:
        self.clock = BASE
        await self.inspirations.add_collected(
            "u1",
            image_url="https://cdn.example.com/trend-1.jpg",
            tags=["minimal"],
            source_url="https://blog.example.com/post-1",
        )
        await self.inspirations.add_collected(
            "u1",
            image_url="https://cdn.example.com/trend-2.jpg",
            tags=["minimal"],
            source_url="https://blog.example.com/post-1",
        )
        # 重复图片不再写入
        again = await self.inspirations.add_collected(
            "u1",
            image_url="https://cdn.example.com/trend-1.jpg",
            tags=["minimal"],
            source_url="https://blog.example.com/post-1",
        )
        self.assertIsNone(again)

        profile = await self.memory.get("u1")

        self.assertEqual(
            profile.inspiration_image_urls,
            [
                "https://cdn.example.com/trend-2.jpg",
                "https://cdn.example.com/trend-1.jpg",
            ],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
