"""Default stylist entry point; production can pass an API-backed provider."""
from app.adapters import RuleBasedStylist
from app.schemas import RecommendationDraft

_demo_provider = RuleBasedStylist()


async def propose(context: dict, previous_issues: list[dict] | None = None,
                  provider=None) -> RecommendationDraft:
    """Generate at most two outfit drafts from structured context and image references."""
    active_provider = provider or _demo_provider
    return await active_provider.propose(context, previous_issues)
