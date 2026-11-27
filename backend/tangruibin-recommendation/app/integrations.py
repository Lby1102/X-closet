"""Composition helpers for connecting recommendation ports to shared services."""
from __future__ import annotations

from app.adapters import MemoryProfileAdapter, TrendsServiceAdapter, WardrobeServiceAdapter
from app.services import RecommendationService


def build_recommendation_service(*, weather, wardrobe, closet, memory, trends,
                                 stylist=None, review=None) -> RecommendationService:
    """Build the recommendation workflow around the shared service interfaces.

    Expected shared interfaces:
    - wardrobe.list(user_id)
    - closet.search(user_id, image_id, limit=10)
    - memory.get(user_id)
    - trends.search_with_warnings(scene, season, limit=3, user_id=...)

    This module contains only interface adapters; shared service implementations are
    provided by the caller and are not copied into this project.
    """
    return RecommendationService(
        weather=weather,
        wardrobe=WardrobeServiceAdapter(wardrobe, closet),
        profile=MemoryProfileAdapter(memory),
        trends=TrendsServiceAdapter(trends),
        stylist=stylist,
        review=review,
    )
