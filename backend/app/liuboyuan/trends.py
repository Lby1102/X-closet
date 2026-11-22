"""X Closet v1 · 趋势采集与检索（trends）— 负责人：刘博远

对外契约（见 02 接口协议）
    POST /inspirations:collect  {style*:string, season?:spring|summer|autumn|winter}
                                -> 202 Job；成功 result 为 {"added_count": int}
    “采集工具读取配置 TREND_SOURCES 中的来源 URL 列表，抓取图片与正文，模型抽取风格，
     再筛选并去重。” 采集成功后刷新喜好列表（inspirations）。

内部契约（见 03 内部接口与运行流程）
    collect(user_id, style, season)      -> {"added_count": int}
    search(scene, season, limit=3)       -> 案例列表，含图片、摘要与来源

链路
    TREND_SOURCES -> Fetcher(HTTPX) -> StyleExtractor(模型) -> 过滤 -> 按来源/图片去重
    -> TrendRepository(PostgreSQL) -> InspirationsService(喜好列表刷新) -> added_count

可注入端口（真实接入点见各 Protocol 注释）：Fetcher / StyleExtractor / TrendRepository。
默认给 HTTPX + 规则抽取 + 内存存储，保证无骨架时也能端到端跑通。
"""

from __future__ import annotations

import asyncio
import html as html_lib
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Iterable, Mapping, Protocol, Sequence
from urllib.parse import urljoin

from pydantic import BaseModel, ConfigDict, Field

if __package__:  # pragma: no cover - 包内导入
    from .inspirations import (
        InspirationsService,
        get_inspiration_service,
        utcnow,
    )
else:  # pragma: no cover - 兼容 python trends.py 直接加载
    from inspirations import InspirationsService, get_inspiration_service, utcnow

__all__ = [
    "SCENES",
    "SEASONS",
    "TREND_SOURCES",
    "CollectResult",
    "ExtractedTrend",
    "FetchedPage",
    "Fetcher",
    "HeuristicStyleExtractor",
    "HttpFetcher",
    "InMemoryTrendRepository",
    "ModelStyleExtractor",
    "StyleExtractor",
    "TrendCase",
    "TrendError",
    "TrendRepository",
    "TrendsService",
    "bootstrap_demo",
    "load_trend_sources",
    "parse_page",
    "reload_trend_sources",
]

SCENES: tuple[str, ...] = ("commute", "casual", "business")
SEASONS: tuple[str, ...] = ("spring", "summer", "autumn", "winter")

_ENV_TREND_SOURCES = "TREND_SOURCES"
_MAX_SEARCH_LIMIT = 20
_DEFAULT_SEARCH_LIMIT = 3
_MAX_CASES_PER_PAGE = 3


# --------------------------------------------------------------------------- #
# 错误（02：422 字段/业务条件无效，502 外部服务失败）
# --------------------------------------------------------------------------- #
class TrendError(Exception):
    code = "INVALID_REQUEST"
    status_code = 422
    retryable = False

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code:
            self.code = code

    def to_error(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "retryable": self.retryable}


class TrendConfigError(TrendError):
    code = "TREND_SOURCES_MISSING"


class TrendUpstreamError(TrendError):
    code = "UPSTREAM_FAILED"
    status_code = 502
    retryable = True


# --------------------------------------------------------------------------- #
# TREND_SOURCES 配置
# --------------------------------------------------------------------------- #
def _split_sources(raw: str) -> list[str]:
    parts = re.split(r"[,，;\n\r\t]+", raw)
    return [part.strip() for part in parts if part.strip()]


def _settings_sources(settings: Any) -> list[str] | None:
    """从配置对象/字典读取 TREND_SOURCES（兼容 pydantic settings 与 dict）。"""
    if settings is None:
        return None
    value = None
    if isinstance(settings, Mapping):
        value = settings.get("TREND_SOURCES")
    else:
        value = getattr(settings, "TREND_SOURCES", None)
    if value is None:
        return None
    if isinstance(value, str):
        return _split_sources(value)
    return [str(item).strip() for item in value if str(item).strip()]


def load_trend_sources(
    settings: Any = None,
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    """读取 TREND_SOURCES 来源 URL 列表；只保留 http(s) 地址并去重保序。

    读取顺序：显式 settings（郭子骏的配置对象）-> 环境变量 TREND_SOURCES。
    真实接入点：把 pydantic-settings 的 Settings 实例传进来即可。
    """
    env = environ if environ is not None else os.environ
    sources = _settings_sources(settings)
    if sources is None:
        sources = _split_sources(env.get(_ENV_TREND_SOURCES, "") or "")

    result: list[str] = []
    seen: set[str] = set()
    for url in sources:
        if not re.match(r"^https?://[^\s]+$", url, re.IGNORECASE):
            continue
        key = url.rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(url)
    return result


#: 导入时的来源快照；采集在未显式传参时读取实时配置（见 TrendsService.collect）。
TREND_SOURCES: tuple[str, ...] = ()


def reload_trend_sources(
    settings: Any = None, environ: Mapping[str, str] | None = None
) -> list[str]:
    """刷新 TREND_SOURCES 快照（配置热更新或测试设置环境变量后调用）。"""
    global TREND_SOURCES
    TREND_SOURCES = tuple(load_trend_sources(settings, environ))
    return list(TREND_SOURCES)


reload_trend_sources()


# --------------------------------------------------------------------------- #
# 抓取
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FetchedPage:
    """一次来源抓取的结果：正文文本 + 候选图片 URL。"""

    url: str
    title: str = ""
    text: str = ""
    image_urls: tuple[str, ...] = ()


class Fetcher(Protocol):
    """抓取端口。真实实现：HTTPX（03 趋势 RAG 技术栈）。"""

    async def fetch(self, url: str) -> FetchedPage: ...


_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_OG_IMAGE_RE = re.compile(
    r"""<meta[^>]+(?:property|name)=["']og:image["'][^>]*>""", re.IGNORECASE
)
_CONTENT_ATTR_RE = re.compile(r"""content=["']([^"']+)["']""", re.IGNORECASE)
_IMG_SRC_RE = re.compile(r"""<img\b[^>]*?\bsrc=["']([^"']+)["']""", re.IGNORECASE)
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
_BLOCK_TAG_RE = re.compile(
    r"</?(?:p|div|br|li|h[1-6]|section|article|tr|td)\b[^>]*>", re.IGNORECASE
)
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"[ \t\u00a0]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def _html_to_text(source: str) -> str:
    cleaned = _SCRIPT_STYLE_RE.sub(" ", source)
    cleaned = _BLOCK_TAG_RE.sub("\n", cleaned)
    cleaned = _TAG_RE.sub(" ", cleaned)
    cleaned = html_lib.unescape(cleaned)
    cleaned = _SPACE_RE.sub(" ", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.splitlines())
    cleaned = _BLANK_LINES_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def _collect_image_urls(source: str, base_url: str) -> tuple[str, ...]:
    found: list[str] = []
    for block in _OG_IMAGE_RE.findall(source):
        match = _CONTENT_ATTR_RE.search(block)
        if match:
            found.append(match.group(1))
    found.extend(_IMG_SRC_RE.findall(source))

    result: list[str] = []
    seen: set[str] = set()
    for raw in found:
        candidate = urljoin(base_url, html_lib.unescape(raw).strip())
        if not re.match(r"^https?://", candidate, re.IGNORECASE):
            continue
        key = candidate.rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return tuple(result)


def parse_page(url: str, body: str) -> FetchedPage:
    """把 HTML 解析成 FetchedPage（正文 + 图片）。纯函数，便于单测。"""
    title_match = _TITLE_RE.search(body)
    title = html_lib.unescape(title_match.group(1)).strip() if title_match else ""
    return FetchedPage(
        url=url,
        title=title,
        text=_html_to_text(body),
        image_urls=_collect_image_urls(body, url),
    )


class HttpFetcher:
    """HTTPX 实现。client 可注入（测试用 httpx.MockTransport 验证真实抓取路径）。"""

    def __init__(
        self,
        *,
        timeout: float = 20.0,
        client: Any = None,
        user_agent: str = "X-Closet-TrendBot/1.0 (+https://x-closet.local)",
    ) -> None:
        self._timeout = timeout
        self._client = client
        self._user_agent = user_agent

    async def fetch(self, url: str) -> FetchedPage:
        try:
            import httpx  # 局部导入：无 HTTPX 时只有真正抓取才失败
        except Exception as exc:  # pragma: no cover - 环境缺依赖
            raise TrendUpstreamError(f"HTTPX 不可用：{exc}") from exc

        headers = {"User-Agent": self._user_agent}
        try:
            if self._client is not None:
                response = await self._client.get(url, headers=headers)
            else:
                async with httpx.AsyncClient(
                    timeout=self._timeout,
                    follow_redirects=True,
                    headers=headers,
                ) as client:
                    response = await client.get(url)
            response.raise_for_status()
        except TrendError:
            raise
        except Exception as exc:
            raise TrendUpstreamError(f"抓取失败 {url}：{exc}") from exc

        return parse_page(str(response.url), response.text)


# --------------------------------------------------------------------------- #
# 风格抽取
# --------------------------------------------------------------------------- #
class ExtractedTrend(BaseModel):
    """模型抽取出的单条趋势案例（未入库）。"""

    model_config = ConfigDict(extra="ignore")

    image_url: str
    summary: str
    style: str = ""
    season: str | None = None
    scene: str | None = None
    tags: list[str] = Field(default_factory=list)


class StyleExtractor(Protocol):
    """风格抽取端口。真实实现：模型 API（见 ModelStyleExtractor）。"""

    async def extract(self, page: FetchedPage) -> list[ExtractedTrend]: ...


STYLE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "minimal": ("minimal", "clean", "简约", "极简"),
    "streetwear": ("streetwear", "street style", "潮流", "街头"),
    "business": ("business", "formal", "商务", "正装"),
    "casual": ("casual", "休闲", "日常"),
    "sporty": ("sporty", "athleisure", "运动"),
    "elegant": ("elegant", "romantic", "优雅", "法式"),
    "vintage": ("vintage", "retro", "复古"),
    "y2k": ("y2k", "千禧"),
}
SEASON_KEYWORDS: dict[str, tuple[str, ...]] = {
    "spring": ("spring", "春季", "春天"),
    "summer": ("summer", "夏季", "夏天", "hot weather"),
    "autumn": ("autumn", "fall", "秋季", "秋天"),
    "winter": ("winter", "冬季", "冬天", "cold weather"),
}
SCENE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "commute": ("commute", "commuting", "通勤", "上下班"),
    "business": ("business", "office", "商务", "办公"),
    "casual": ("weekend", "休闲", "日常", "casual"),
}
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？\n])\s*")


def _first_match(text: str, table: Mapping[str, tuple[str, ...]]) -> str | None:
    lowered = text.lower()
    for key, keywords in table.items():
        if any(keyword in lowered for keyword in keywords):
            return key
    return None


def _matched_keys(text: str, table: Mapping[str, tuple[str, ...]]) -> list[str]:
    lowered = text.lower()
    return [key for key, keywords in table.items() if any(k in lowered for k in keywords)]


def _pick_summary(text: str, keywords: Iterable[str], fallback: str) -> str:
    keywords = list(keywords)
    matches: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        candidate = sentence.strip()
        if len(candidate) < 8:
            continue
        lowered = candidate.lower()
        if any(keyword in lowered for keyword in keywords):
            matches.append(candidate[:240])
    if matches:
        # 多个命中句时取信息量最大的一句，避免只拿到标题
        return max(matches, key=len)
    compact = " ".join(text.split())
    if compact:
        return compact[:240]
    return fallback[:240]


class HeuristicStyleExtractor:
    """规则抽取（无模型时的可验证实现）。

    关键词识别风格/季节/场景，每页最多产出 max_items 条案例。
    真实接入点：换成 ModelStyleExtractor（模型抽取风格），接口完全一致。
    """

    def __init__(self, *, max_items: int = _MAX_CASES_PER_PAGE) -> None:
        self._max_items = max(1, max_items)

    async def extract(self, page: FetchedPage) -> list[ExtractedTrend]:
        corpus = f"{page.title}\n{page.text}".strip()
        if not corpus:
            return []
        styles = _matched_keys(corpus, STYLE_KEYWORDS)
        season = _first_match(corpus, SEASON_KEYWORDS)
        scene = _first_match(corpus, SCENE_KEYWORDS)
        keywords = [
            keyword
            for style in styles
            for keyword in STYLE_KEYWORDS.get(style, ())
        ]
        summary = _pick_summary(corpus, keywords or [corpus[:8]], page.title)
        tags = styles + ([season] if season else []) + ([scene] if scene else [])

        result: list[ExtractedTrend] = []
        for image_url in page.image_urls[: self._max_items]:
            result.append(
                ExtractedTrend(
                    image_url=image_url,
                    summary=summary,
                    style=styles[0] if styles else "",
                    season=season,
                    scene=scene,
                    tags=list(tags),
                )
            )
        return result


MODEL_PROMPT = (
    "你是时尚趋势标注器。阅读给定页面（标题、正文、图片 URL），抽取穿搭风格案例。"
    "只返回 JSON：{\"cases\":[{\"image_url\":str,\"summary\":str,"
    "\"style\":str,\"season\":spring|summer|autumn|winter|null,"
    "\"scene\":commute|casual|business|null,\"tags\":[str]}]}。"
    "style 用英文小写词（如 minimal/streetwear/athleisure），summary 不超过 240 字。"
)


class ModelStyleExtractor:
    """模型抽取（真实接入点）。

    complete_json(prompt) 由模型网关提供（唐睿彬的模型接入 / 郭子骏的后端配置），
    输入 prompt 返回 JSON 字符串。未注入时请使用 HeuristicStyleExtractor。
    """

    def __init__(
        self,
        complete_json: Callable[[str], Awaitable[str]],
        *,
        max_items: int = _MAX_CASES_PER_PAGE,
        page_chars: int = 4000,
    ) -> None:
        self._complete = complete_json
        self._max_items = max(1, max_items)
        self._page_chars = page_chars

    async def extract(self, page: FetchedPage) -> list[ExtractedTrend]:
        payload = json.dumps(
            {
                "url": page.url,
                "title": page.title,
                "text": page.text[: self._page_chars],
                "image_urls": list(page.image_urls)[: self._max_items],
            },
            ensure_ascii=False,
        )
        raw = await self._complete(f"{MODEL_PROMPT}\n\n页面：{payload}")
        data = json.loads(raw)
        items = data.get("cases") if isinstance(data, dict) else data
        if not isinstance(items, list):
            return []

        allowed_images = set(page.image_urls)
        result: list[ExtractedTrend] = []
        for item in items[: self._max_items]:
            if not isinstance(item, dict):
                continue
            image_url = str(item.get("image_url") or "").strip()
            if not image_url:
                continue
            if page.image_urls and image_url not in allowed_images:
                # 模型不得编造页面里不存在的图片（防幻觉，回退到页面首图）
                image_url = page.image_urls[0]
            season = str(item.get("season") or "").strip().lower() or None
            scene = str(item.get("scene") or "").strip().lower() or None
            tags = [str(tag).strip() for tag in item.get("tags") or [] if str(tag).strip()]
            result.append(
                ExtractedTrend(
                    image_url=image_url,
                    summary=str(item.get("summary") or item.get("title") or "")[:240],
                    style=str(item.get("style") or "").strip().lower(),
                    season=season if season in SEASONS else None,
                    scene=scene if scene in SCENES else None,
                    tags=tags,
                )
            )
        return result


# --------------------------------------------------------------------------- #
# 案例模型与存储
# --------------------------------------------------------------------------- #
class TrendCase(BaseModel):
    """已入库趋势案例：含图片、摘要与来源（03 检索输出要求）。"""

    model_config = ConfigDict(extra="ignore")

    case_id: str
    image_url: str
    summary: str = ""
    source_url: str = ""
    style: str = ""
    season: str | None = None
    scene: str | None = None
    tags: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _CaseRecord:
    case: TrendCase
    created_at: datetime
    seq: int


def _dedupe_key(case: TrendCase) -> tuple[str, str]:
    return (
        (case.source_url or "").rstrip("/").lower(),
        (case.image_url or "").rstrip("/").lower(),
    )


class TrendRepository(Protocol):
    """趋势案例存储端口。真实实现：PostgreSQL + pgvector（按来源去重、按场景/季节检索）。"""

    async def insert_new(
        self, user_id: str, cases: Sequence[TrendCase]
    ) -> list[TrendCase]: ...

    async def list_desc(
        self, user_id: str, limit: int | None = None
    ) -> list[TrendCase]: ...

    async def search(
        self,
        user_id: str,
        scene: str,
        season: str | None,
        limit: int,
        *,
        include_untyped: bool = True,
    ) -> list[TrendCase]: ...


class InMemoryTrendRepository:
    """进程内实现：来源/图片去重 + 场景季节检索 + 创建时间倒序。"""

    def __init__(self) -> None:
        self._records: dict[str, list[_CaseRecord]] = {}
        self._keys: dict[str, set[tuple[str, str]]] = {}
        self._seq = 0
        self._lock = asyncio.Lock()

    async def insert_new(
        self, user_id: str, cases: Sequence[TrendCase]
    ) -> list[TrendCase]:
        created: list[TrendCase] = []
        async with self._lock:
            keys = self._keys.setdefault(user_id, set())
            for case in cases:
                key = _dedupe_key(case)
                if key in keys:
                    continue
                keys.add(key)
                self._seq += 1
                self._records.setdefault(user_id, []).append(
                    _CaseRecord(case=case, created_at=_case_time(), seq=self._seq)
                )
                created.append(case)
        return created

    async def list_desc(
        self, user_id: str, limit: int | None = None
    ) -> list[TrendCase]:
        if limit is not None and limit <= 0:
            return []
        async with self._lock:
            rows = sorted(
                self._records.get(user_id, []),
                key=lambda record: (record.created_at, record.seq),
                reverse=True,
            )
            items = [record.case for record in rows]
        return items[:limit] if limit is not None else items

    async def search(
        self,
        user_id: str,
        scene: str,
        season: str | None,
        limit: int,
        *,
        include_untyped: bool = True,
    ) -> list[TrendCase]:
        if limit <= 0:
            return []
        async with self._lock:
            rows = sorted(
                self._records.get(user_id, []),
                key=lambda record: (record.created_at, record.seq),
                reverse=True,
            )
        exact: list[TrendCase] = []
        untyped: list[TrendCase] = []
        for record in rows:
            case = record.case
            if season is not None and case.season not in (season, None):
                continue
            if case.scene == scene:
                exact.append(case)
            elif case.scene is None and include_untyped:
                untyped.append(case)
        merged = exact + untyped
        return merged[:limit]


def _case_time() -> datetime:
    """案例入库时间。真实实现由 SQL created_at 提供；内存实现用当前时间。"""
    return utcnow()


# --------------------------------------------------------------------------- #
# 采集结果
# --------------------------------------------------------------------------- #
@dataclass
class CollectResult:
    added_count: int = 0
    duplicates: int = 0
    warnings: list[str] = field(default_factory=list)
    created: list[TrendCase] = field(default_factory=list)

    def as_job_result(self) -> dict[str, int]:
        """Job.result，v1 只暴露 added_count（02）。"""
        return {"added_count": self.added_count}


# --------------------------------------------------------------------------- #
# 业务服务
# --------------------------------------------------------------------------- #
class TrendsService:
    """趋势采集与检索。"""

    def __init__(
        self,
        *,
        repository: TrendRepository | None = None,
        inspirations: InspirationsService | None = None,
        fetcher: Fetcher | None = None,
        extractor: StyleExtractor | None = None,
        sources: Sequence[str] | None = None,
        settings: Any = None,
        per_source_timeout: float = 20.0,
        id_factory: Callable[[], str] | None = None,
        default_user_id: str = "u_demo",
    ) -> None:
        self._repository = repository or InMemoryTrendRepository()
        self._inspirations = inspirations
        self._fetcher = fetcher or HttpFetcher()
        self._extractor = extractor or HeuristicStyleExtractor()
        self._sources = list(sources) if sources is not None else None
        self._settings = settings
        self._timeout = per_source_timeout
        self._id_factory = id_factory or (lambda: f"trend_{uuid.uuid4().hex[:12]}")
        #: search() 未显式传 user_id 时使用的演示 scope（不与其他用户共享数据）。
        self._default_user_id = default_user_id

    # ---- 配置 ------------------------------------------------------------- #
    @property
    def default_user_id(self) -> str:
        return self._default_user_id

    def resolve_sources(self, sources: Sequence[str] | None = None) -> list[str]:
        if sources is not None:
            return [str(url).strip() for url in sources if str(url).strip()]
        if self._sources is not None:
            return list(self._sources)
        live = load_trend_sources(self._settings)
        return live or list(TREND_SOURCES)

    def _inspiration_service(self) -> InspirationsService | None:
        if self._inspirations is not None:
            return self._inspirations
        try:
            return get_inspiration_service()
        except Exception:  # pragma: no cover - 默认单例不可用
            return None

    # ---- 采集：POST /inspirations:collect --------------------------------- #
    async def collect(
        self,
        user_id: str,
        style: str,
        season: str | None = None,
        sources: Sequence[str] | None = None,
    ) -> CollectResult:
        user = (user_id or "").strip()
        if not user:
            raise TrendError("user_id 不能为空", code="INVALID_FIELD")
        target_style = (style or "").strip()
        if not target_style:
            raise TrendError("style 不能为空", code="INVALID_FIELD")
        normalized_season = _normalize_season(season)

        urls = self.resolve_sources(sources)
        if not urls:
            raise TrendConfigError(
                "TREND_SOURCES 未配置来源 URL，无法采集趋势"
            )

        pages, warnings = await self._fetch_all(urls)
        if not pages:
            detail = f"；{warnings[0]}" if warnings else ""
            raise TrendUpstreamError(f"全部趋势来源抓取失败{detail}")

        candidates: list[ExtractedTrend] = []
        for page in pages:
            try:
                extracted = await self._extractor.extract(page)
            except Exception as exc:
                warnings.append(
                    f"TREND_EXTRACT_FAILED: {page.url} ({_brief(exc)})"
                )
                continue
            candidates.extend(extracted)

        filtered = [
            case
            for case in candidates
            if _matches_style(case, target_style)
            and _matches_season(case, normalized_season)
        ]

        cases: list[TrendCase] = []
        for item in filtered:
            case_season = item.season or normalized_season
            cases.append(
                TrendCase(
                    case_id=self._id_factory(),
                    image_url=item.image_url,
                    summary=item.summary,
                    source_url=_source_for(item.image_url, pages),
                    style=item.style or target_style.lower(),
                    season=case_season,
                    scene=item.scene,
                    tags=list(item.tags),
                )
            )

        created = await self._repository.insert_new(user, cases)
        await self._mirror_to_inspirations(user, created, warnings)

        return CollectResult(
            added_count=len(created),
            duplicates=max(0, len(cases) - len(created)),
            warnings=warnings,
            created=created,
        )

    async def _fetch_all(
        self, urls: Sequence[str]
    ) -> tuple[list[FetchedPage], list[str]]:
        async def one(url: str) -> FetchedPage | Exception:
            try:
                return await asyncio.wait_for(
                    self._fetcher.fetch(url), timeout=self._timeout
                )
            except Exception as exc:
                return exc

        results = await asyncio.gather(*(one(url) for url in urls))
        pages: list[FetchedPage] = []
        warnings: list[str] = []
        for url, outcome in zip(urls, results):
            if isinstance(outcome, FetchedPage):
                pages.append(outcome)
            else:
                warnings.append(f"TREND_SOURCE_FAILED: {url} ({_brief(outcome)})")
        return pages, warnings

    async def _mirror_to_inspirations(
        self, user_id: str, cases: Sequence[TrendCase], warnings: list[str]
    ) -> None:
        service = self._inspiration_service()
        if service is None:
            warnings.append("INSPIRATION_WRITE_SKIPPED: 喜好图片服务未接线")
            return
        for case in cases:
            try:
                await service.add_collected(
                    user_id,
                    image_url=case.image_url,
                    tags=case.tags,
                    source_url=case.source_url or None,
                )
            except Exception as exc:
                warnings.append(
                    f"INSPIRATION_WRITE_FAILED: {case.image_url} ({_brief(exc)})"
                )

    # ---- 检索：trends.search(scene, season, limit=3) ---------------------- #
    async def search(
        self,
        scene: str,
        season: str | None = None,
        limit: int = _DEFAULT_SEARCH_LIMIT,
        *,
        user_id: str | None = None,
    ) -> list[TrendCase]:
        """按场景和季节检索已入库案例（03：trends.search(scene,season,limit=3)）。

        签名与 03 的写法一致；user_id 作为可选关键字参数做资源隔离
        （03 “user_id 由后端传入”）。未传时使用 default_user_id 这个演示 scope，
        不会读到其他用户采集的数据。
        """
        user = (user_id or self._default_user_id or "").strip()
        if not user:
            raise TrendError("user_id 不能为空", code="INVALID_FIELD")
        normalized_scene = (scene or "").strip().lower()
        if normalized_scene not in SCENES:
            raise TrendError(
                f"scene 必须是 {'/'.join(SCENES)} 之一", code="INVALID_SCENE"
            )
        normalized_season = _normalize_season(season)
        normalized_limit = _normalize_limit(limit)
        return await self._repository.search(
            user, normalized_scene, normalized_season, normalized_limit
        )

    async def search_with_warnings(
        self,
        scene: str,
        season: str | None = None,
        limit: int = _DEFAULT_SEARCH_LIMIT,
        *,
        user_id: str | None = None,
    ) -> tuple[list[TrendCase], list[str]]:
        """推荐编排用：趋势查询失败可带警告继续（03）。"""
        try:
            cases = await self.search(scene, season, limit, user_id=user_id)
        except Exception as exc:
            return [], [f"TREND_SEARCH_FAILED: {exc}"]
        return cases, []

    # ---- 初始案例人工导入（02：初始案例允许人工导入） --------------------- #
    async def add_manual_cases(
        self, user_id: str, cases: Sequence[TrendCase | Mapping[str, Any]]
    ) -> list[TrendCase]:
        user = (user_id or "").strip()
        if not user:
            raise TrendError("user_id 不能为空", code="INVALID_FIELD")
        normalized: list[TrendCase] = []
        for item in cases:
            case = item if isinstance(item, TrendCase) else TrendCase(**dict(item))
            normalized.append(
                case.model_copy(
                    update={"case_id": case.case_id or self._id_factory()}
                )
            )
        return await self._repository.insert_new(user, normalized)


def _normalize_season(season: str | None) -> str | None:
    if season is None:
        return None
    value = str(season).strip().lower()
    if not value:
        return None
    if value not in SEASONS:
        raise TrendError(
            f"season 必须是 {'/'.join(SEASONS)} 之一", code="INVALID_SEASON"
        )
    return value


def _normalize_limit(limit: int) -> int:
    try:
        value = int(limit)
    except (TypeError, ValueError) as exc:
        raise TrendError("limit 必须是整数", code="INVALID_LIMIT") from exc
    if value <= 0:
        raise TrendError("limit 必须大于 0", code="INVALID_LIMIT")
    return min(value, _MAX_SEARCH_LIMIT)


def _matches_style(case: ExtractedTrend, target_style: str) -> bool:
    wanted = target_style.strip().lower()
    haystack = {tag.strip().lower() for tag in case.tags}
    haystack.add((case.style or "").strip().lower())
    return wanted in haystack


def _matches_season(case: ExtractedTrend, season: str | None) -> bool:
    if season is None:
        return True
    return case.season in (season, None)


def _brief(exc: BaseException, limit: int = 160) -> str:
    """把外部异常压成一行短描述，避免 warnings/error.message 过长。"""
    text = " ".join(str(exc).split())
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _source_for(image_url: str, pages: Sequence[FetchedPage]) -> str:
    for page in pages:
        if image_url in page.image_urls:
            return page.url
    return pages[0].url if pages else ""

# --------------------------------------------------------------------------- #
# 默认单例 / 演示数据
# --------------------------------------------------------------------------- #
_default_service: TrendsService | None = None


def get_trends_service() -> TrendsService:
    global _default_service
    if _default_service is None:
        _default_service = TrendsService()
    return _default_service


def reset_default_service() -> None:
    global _default_service
    _default_service = None


DEMO_CASES: tuple[dict[str, Any], ...] = (
    {
        "case_id": "trend_demo_commute_spring_1",
        "image_url": "https://cdn.x-closet.local/trends/commute-spring-1.jpg",
        "summary": "轻薄西装 + 直筒裤的通勤组合，浅色系适合春季空调房。",
        "source_url": "https://demo.x-closet.local/trends/commute-spring",
        "style": "minimal",
        "season": "spring",
        "scene": "commute",
        "tags": ["minimal", "spring", "commute"],
    },
    {
        "case_id": "trend_demo_business_winter_1",
        "image_url": "https://cdn.x-closet.local/trends/business-winter-1.jpg",
        "summary": "深色羊毛大衣配高领针织，商务冬季外搭。",
        "source_url": "https://demo.x-closet.local/trends/business-winter",
        "style": "business",
        "season": "winter",
        "scene": "business",
        "tags": ["business", "winter"],
    },
    {
        "case_id": "trend_demo_casual_summer_1",
        "image_url": "https://cdn.x-closet.local/trends/casual-summer-1.jpg",
        "summary": "宽松亚麻衬衫与短裤的周末休闲搭配。",
        "source_url": "https://demo.x-closet.local/trends/casual-summer",
        "style": "casual",
        "season": "summer",
        "scene": "casual",
        "tags": ["casual", "summer"],
    },
)


async def bootstrap_demo(
    service: TrendsService,
    user_id: str | None = None,
    cases: Sequence[Mapping[str, Any]] = DEMO_CASES,
) -> list[TrendCase]:
    """人工导入初始案例（02：初始案例允许人工导入）。

    user_id 省略时导入到 service.default_user_id（与 3 参 search() 同一 scope）。
    """
    target = user_id or service.default_user_id
    return await service.add_manual_cases(target, cases)
