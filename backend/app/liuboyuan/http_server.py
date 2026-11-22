"""stdlib HTTP 适配器（Mock，替代郭子骏的 FastAPI 应用）。

用途：
    * 前端 / 冒烟脚本按 02 的真实路径与状态码联调，不依赖 FastAPI；
    * 验证 multipart 上传、X-User-Id、Idempotency-Key 与错误码。

真实接入点：郭子骏的 FastAPI 应用。把 ``Api`` 的九个方法挂到 APIRouter 上即可，
本文件不包含业务规则。

启动：
    cd backend && python -m app.liuboyuan.http_server --port 8010
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .api import Api, InvalidInput, error_response
from .container import DEFAULT_USERS, build_services

__all__ = ["MockHTTPServer", "make_server", "parse_multipart"]


def parse_multipart(content_type: str, body: bytes) -> list[tuple[str, bytes, str]]:
    """解析 ``multipart/form-data``，返回 [(filename, content, media_type)]。"""
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode(
        "utf-8"
    )
    message = BytesParser(policy=policy.default).parsebytes(header + body)
    files: list[tuple[str, bytes, str]] = []
    if not message.is_multipart():  # pragma: no cover - 客户端未按规范发送
        return files
    for part in message.iter_parts():
        filename = part.get_filename()
        if not filename:
            continue
        files.append(
            (filename, part.get_payload(decode=True) or b"", part.get_content_type())
        )
    return files


class ClosetRequestHandler(BaseHTTPRequestHandler):
    """把一次 HTTP 请求翻译成 ``Api.dispatch`` 调用。"""

    protocol_version = "HTTP/1.1"
    server_version = "X-Closet-Mock/1.0"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        self._handle("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._handle("POST")

    def do_PATCH(self) -> None:  # noqa: N802
        self._handle("PATCH")

    def _handle(self, method: str) -> None:
        api: Api = self.server.api  # type: ignore[attr-defined]
        parsed = urlsplit(self.path)
        query = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length > 0 else b""
        content_type = self.headers.get("Content-Type") or ""

        body: Any = None
        upload: tuple[bytes, str | None] | None = None
        try:
            if raw and content_type.lower().startswith("multipart/form-data"):
                files = parse_multipart(content_type, raw)
                if not files:
                    raise InvalidInput("缺少上传文件 file")
                _, payload, media_type = files[0]
                upload = (payload, media_type)
            elif raw:
                body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            self._write(422, error_response(InvalidInput(f"请求体不是合法 JSON：{exc.msg}")))
            return
        except InvalidInput as exc:
            self._write(exc.status_code, error_response(exc))
            return

        response = self.server.call(
            api.dispatch(
                method,
                parsed.path,
                user_id=self.headers.get("X-User-Id"),
                idempotency_key=self.headers.get("Idempotency-Key"),
                body=body,
                query=query,
                upload=upload,
            )
        )
        self._write(response.status_code, response.body)

    def _write(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        if getattr(self.server, "verbose", False):  # pragma: no cover - 调试用
            super().log_message(format, *args)


class MockHTTPServer(ThreadingHTTPServer):
    """带 ``api`` 引用的线程化 HTTP 服务（测试里可 port=0 随机端口）。

    后台任务（202 Job）需要跨请求存活，所以请求协程统一跑在一个**常驻**事件循环
    线程里；每个请求线程通过 ``call()`` 把协程提交进去并等结果。
    """

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        api: Api,
        verbose: bool = False,
    ) -> None:
        self.api = api
        self.verbose = verbose
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, name="x-closet-mock-loop", daemon=True
        )
        super().__init__(address, handler_class)
        self._loop_thread.start()

    def call(self, coroutine: Any, timeout: float = 30.0) -> Any:
        """线程安全地在本服务的常驻事件循环里执行协程。"""
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        return future.result(timeout=timeout)

    def server_close(self) -> None:
        super().server_close()
        if self._loop_thread.is_alive():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=5.0)


def make_server(
    services: Any = None,
    host: str = "127.0.0.1",
    port: int = 0,
    *,
    verbose: bool = False,
) -> MockHTTPServer:
    """构造（不启动）HTTP 服务；``services`` 省略时装配一套全 Mock 服务。"""
    bundle = services or build_services()
    return MockHTTPServer(
        (host, port), ClosetRequestHandler, api=bundle.api, verbose=verbose
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="X-Closet 刘博远模块 Mock HTTP 服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--users", default="u_1,u_demo")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    users = [item.strip() for item in args.users.split(",") if item.strip()]
    services = build_services(users=users or DEFAULT_USERS)
    server = make_server(services, args.host, args.port, verbose=args.verbose)
    host, port = server.server_address[:2]
    print(f"X-Closet Mock HTTP 已启动: http://{host}:{port}", flush=True)
    print(f"预置用户: {', '.join(users)}（请求头 X-User-Id）", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - 手工停止
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover - 手工启动
    raise SystemExit(main())
