"""X Closet v1 · 刘博远模块（识别 / 衣橱 / 以图搜衣 / 喜好图 / 趋势 / 记忆）。

对外只暴露装配入口，业务实现按文件拆分：

    container.build_services()   一次装配全部服务（全 Mock 可直接跑）
    api.Api                     02 的 HTTP 契约层（框架无关）
    http_server.make_server()   stdlib HTTP 适配器（Mock，替代 FastAPI 应用）
    wardrobe.WardrobeService    确认入库 / 查询 / 改标签 + 向量同步
    recognition.GarmentRecognitionService   识别（Job: garment_analysis）
    closet_search.ClosetSearchService       closet.search(user_id, image_id, limit=10)

窗口 B 的实现（inspirations / trends / memory）原样保留，未做改动。
"""

from __future__ import annotations

from .container import Services, build_services, drain

__all__ = ["Services", "build_services", "drain"]
