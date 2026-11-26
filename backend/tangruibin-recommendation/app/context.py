from __future__ import annotations

import re

from app.schemas import ChatMessage, CompressedContext, Scene


SCENES = {
    "通勤": Scene.commute.value,
    "上班": Scene.commute.value,
    "工作": Scene.business.value,
    "商务": Scene.business.value,
    "休闲": Scene.casual.value,
    "约会": Scene.casual.value,
    "casual": Scene.casual.value,
    "business": Scene.business.value,
    "commute": Scene.commute.value,
}
PREFERENCES = ("简约", "休闲", "正式", "通勤", "显瘦", "舒适", "优雅", "运动", "复古", "宽松", "低调", "亮色")
CONSTRAINTS = ("不要", "别", "避免", "不穿", "禁忌", "过敏", "不能", "不喜欢")


def compress(messages: list[ChatMessage] | list[dict], fallback_scene: str | None = None) -> CompressedContext:
    """Compress recent text only; image/weather/garment IDs stay in typed fields."""
    normalized = [m if isinstance(m, ChatMessage) else ChatMessage.model_validate(m) for m in messages]
    # Recent user messages carry current intent; keep enough dialogue for corrections.
    recent = normalized[-20:]
    user_text = "\n".join(m.text for m in recent if m.role.value == "user")
    all_text = user_text
    scene = fallback_scene
    for message in reversed(recent):
        if message.role.value != "user":
            continue
        mentions = [(match.start(), value) for token, value in SCENES.items()
                    for match in re.finditer(re.escape(token), message.text, re.IGNORECASE)]
        if mentions:
            scene = max(mentions, key=lambda item: item[0])[1]
            break
    prefs = list(dict.fromkeys(tag for tag in PREFERENCES if tag.lower() in all_text.lower()))
    constraints = []
    for line in user_text.splitlines():
        if any(marker in line for marker in CONSTRAINTS):
            cleaned = re.sub(r"\s+", " ", line).strip()
            if cleaned and cleaned not in constraints:
                constraints.append(cleaned[:180])
    summary = "；".join(re.sub(r"\s+", " ", m.text).strip()[:180] for m in recent[-6:])
    return CompressedContext(scene=scene, preferences=prefs, constraints=constraints, summary=summary[:900])
