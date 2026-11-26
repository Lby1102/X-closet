from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class Scene(str, Enum):
    commute = "commute"
    casual = "casual"
    business = "business"


class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"


class ChatMessage(BaseModel):
    role: MessageRole
    text: str = Field(min_length=1, max_length=12000)


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    city: str = Field(min_length=1, max_length=100)
    date: date
    scene: Scene
    request: str | None = Field(default=None, max_length=4000)
    reference_image_id: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list, max_length=100)


class Tags(BaseModel):
    category: str
    color: str
    pattern: str | None = None
    fit: str | None = None
    warmth: str | None = None
    style_tags: list[str] = Field(default_factory=list)

    @field_validator("style_tags", mode="before")
    @classmethod
    def normalize_style_tags(cls, value):
        # Liu Boyuan's Mock accepts one style tag as a string for compatibility.
        return [value] if isinstance(value, str) else value


class Garment(BaseModel):
    garment_id: str
    image_id: str
    image_url: str
    tags: Tags


class Candidate(BaseModel):
    garment_id: str
    image_url: str
    score: float = 0.0


class Preferences(BaseModel):
    liked_styles: list[str] = Field(default_factory=list)
    avoid_tags: list[str] = Field(default_factory=list)


class Profile(BaseModel):
    explicit: Preferences = Field(default_factory=Preferences)
    learned_weights: dict[str, float] = Field(default_factory=dict)
    inspiration_image_urls: list[str] = Field(default_factory=list)
    version: int = 0


class Weather(BaseModel):
    apparent_temp_c: float
    rain_probability: int = Field(ge=0, le=100)
    wind_kph: float = Field(ge=0)
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Trend(BaseModel):
    image_url: str | None = None
    summary: str
    source_url: str | None = None
    tags: list[str] = Field(default_factory=list)


class CompressedContext(BaseModel):
    scene: str | None = None
    preferences: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    summary: str = ""


class OutfitDraft(BaseModel):
    garment_ids: list[str] = Field(min_length=1)
    reason: str


class RecommendationDraft(BaseModel):
    outfits: list[OutfitDraft] = Field(max_length=2)


class ReviewIssue(BaseModel):
    garment_id: str
    reason: str


class ReviewResult(BaseModel):
    passed: bool
    issues: list[ReviewIssue] = Field(default_factory=list)


class Outfit(OutfitDraft):
    outfit_id: str = Field(default_factory=lambda: new_id("outfit"))


class Recommendation(BaseModel):
    recommendation_id: str = Field(default_factory=lambda: new_id("rec"))
    outfits: list[Outfit]
    warnings: list[str] = Field(default_factory=list)


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class JobError(BaseModel):
    code: str
    message: str
    retryable: bool = False


class Job(BaseModel):
    job_id: str = Field(default_factory=lambda: new_id("job"))
    type: str = "recommendation"
    status: JobStatus = JobStatus.queued
    stage: str = "queued"
    result: dict[str, Any] | None = None
    error: JobError | None = None
    warnings: list[str] = Field(default_factory=list)
    request_id: str | None = None


class ApiEnvelope(BaseModel):
    data: Any
    request_id: str


class ApiErrorBody(BaseModel):
    code: str
    message: str
    retryable: bool = False


class ApiErrorEnvelope(BaseModel):
    error: ApiErrorBody
    request_id: str
