"""FastAPI 应用入口。"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.config.logging import setup_logging
from app.config.settings import settings
from app.runtime import AppRuntime


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化运行时容器，关闭时释放资源。"""

    setup_logging(settings.log_level)
    runtime = AppRuntime(settings=settings)
    app.state.runtime = runtime
    try:
        yield
    finally:
        runtime.close()


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例。"""

    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.include_router(api_router)
    return app


app = create_app()
