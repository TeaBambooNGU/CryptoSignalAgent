"""FastAPI 应用入口。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request

from app.api.routes import router as api_router
from app.config.logging import get_logger, log_context, setup_logging
from app.config.settings import settings
from app.runtime import AppRuntime

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化运行时容器，关闭时释放资源。"""

    setup_logging(
        settings.log_level,
        log_to_file=settings.log_to_file,
        log_file_path=settings.log_file_path,
        log_file_max_mb=settings.log_file_max_mb,
        log_file_backup_days=settings.log_file_backup_days,
    )
    runtime = AppRuntime(settings=settings)
    app.state.runtime = runtime
    try:
        yield
    finally:
        runtime.close()


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例。"""

    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def trace_logging_middleware(request: Request, call_next):
        trace_id = request.headers.get("X-Trace-Id", "").strip() or str(uuid4())
        request.state.trace_id = trace_id
        start = perf_counter()

        with log_context(trace_id=trace_id, component="api.request"):
            logger.info(
                "请求开始 method=%s path=%s",
                request.method,
                request.url.path,
            )
            try:
                response = await call_next(request)
            except Exception:
                logger.exception("请求失败 method=%s path=%s", request.method, request.url.path)
                raise

            response.headers["X-Trace-Id"] = trace_id
            duration_ms = (perf_counter() - start) * 1000
            logger.info(
                "请求结束 method=%s path=%s status=%s duration_ms=%.2f",
                request.method,
                request.url.path,
                response.status_code,
                duration_ms,
            )
            return response

    app.include_router(api_router)
    return app


app = create_app()
