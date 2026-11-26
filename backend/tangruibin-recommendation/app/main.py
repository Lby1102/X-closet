from __future__ import annotations

import hashlib
import json
import os
from contextvars import ContextVar
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from app.schemas import ApiEnvelope, ApiErrorBody, ApiErrorEnvelope, JobStatus, RecommendationRequest
from app.services import InMemoryJobStore, RecommendationService, WorkflowError


request_id_var: ContextVar[str] = ContextVar("request_id", default="unknown")
load_dotenv()
app = FastAPI(title="X-Closet 多模态推荐模块", version="1.0.0",
              description="唐睿彬负责的推荐编排与上下文压缩演示模块。")
service = RecommendationService()
jobs = InMemoryJobStore()
ALLOWED_USERS = {value.strip() for value in os.getenv("ALLOWED_USER_IDS", "demo-user").split(",") if value.strip()}


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-Id") or f"req_{uuid4().hex[:12]}"
    token = request_id_var.set(request_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response
    finally:
        request_id_var.reset(token)


def user_from_header(x_user_id: str | None) -> str:
    if not x_user_id or x_user_id not in ALLOWED_USERS:
        raise HTTPException(status_code=401, detail={"code": "UNAUTHORIZED", "message": "缺少或未知的 X-User-Id。"})
    return x_user_id


def envelope(data, status_code: int = 200) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=ApiEnvelope(data=data, request_id=request_id_var.get()).model_dump(mode="json"))


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {"code": "HTTP_ERROR", "message": str(exc.detail)}
    body = ApiErrorEnvelope(error=ApiErrorBody(code=detail.get("code", "HTTP_ERROR"),
                                                message=detail.get("message", "请求失败")),
                            request_id=request_id_var.get())
    return JSONResponse(status_code=exc.status_code, content=body.model_dump(mode="json"))


@app.exception_handler(WorkflowError)
async def workflow_error_handler(request: Request, exc: WorkflowError):
    status = 409 if exc.code == "IDEMPOTENCY_CONFLICT" else 422 if exc.code in {"WARDROBE_EMPTY", "NO_VALID_OUTFIT"} else 502
    body = ApiErrorEnvelope(error=ApiErrorBody(code=exc.code, message=exc.message, retryable=exc.retryable),
                            request_id=request_id_var.get())
    return JSONResponse(status_code=status, content=body.model_dump(mode="json"))


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    body = ApiErrorEnvelope(error=ApiErrorBody(code="INVALID_REQUEST", message="请求字段或业务条件无效。"),
                            request_id=request_id_var.get())
    return JSONResponse(status_code=422, content=body.model_dump(mode="json"))


@app.post("/api/v1/recommendations", status_code=202)
async def create_recommendation(payload: RecommendationRequest,
                                x_user_id: str | None = Header(default=None, alias="X-User-Id"),
                                idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    user_id = user_from_header(x_user_id)
    if not idempotency_key:
        raise HTTPException(status_code=422, detail={"code": "MISSING_IDEMPOTENCY_KEY", "message": "必须提供 Idempotency-Key。"})
    canonical = json.dumps(payload.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    job, is_new = jobs.create(user_id, idempotency_key, fingerprint, request_id_var.get())
    if is_new:
        task = __import__("asyncio").create_task(run_job(user_id, job.job_id, payload))
        jobs.add_task(task)
    return envelope(job, 202)


async def run_job(user_id: str, job_id: str, payload: RecommendationRequest):
    job = jobs.get(user_id, job_id)
    if not job or job.status != JobStatus.queued:
        return
    job.status = JobStatus.running
    job.stage = "loading_context"
    try:
        job.stage = "generating_and_reviewing"
        result = await service.generate(user_id, payload)
        job.result = result.model_dump(mode="json")
        job.warnings = result.warnings
        job.error = None
        job.status = JobStatus.succeeded
        job.stage = "completed"
    except WorkflowError as exc:
        job.error = {"code": exc.code, "message": exc.message, "retryable": exc.retryable}
        job.result = None
        job.status = JobStatus.failed
        job.stage = "failed"
    except Exception:
        job.error = {"code": "UPSTREAM_FAILED", "message": "推荐任务执行失败。", "retryable": True}
        job.result = None
        job.status = JobStatus.failed
        job.stage = "failed"


@app.get("/api/v1/jobs/{job_id}")
async def get_job(job_id: str, x_user_id: str | None = Header(default=None, alias="X-User-Id")):
    user_id = user_from_header(x_user_id)
    job = jobs.get(user_id, job_id)
    if not job:
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "任务不存在。"})
    return envelope(job)


@app.get("/api/v1/recommendations/{recommendation_id}")
async def get_recommendation(recommendation_id: str, x_user_id: str | None = Header(default=None, alias="X-User-Id")):
    user_id = user_from_header(x_user_id)
    result = service.recommendations.get((user_id, recommendation_id))
    if not result:
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "推荐记录不存在。"})
    return envelope(result)


@app.get("/health")
async def health():
    return envelope({"status": "ok", "mode": "demo"})

