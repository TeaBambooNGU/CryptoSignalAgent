"""FastAPI 依赖注入定义。"""

from __future__ import annotations

from fastapi import Request

from app.runtime import AppRuntime


def get_runtime(request: Request) -> AppRuntime:
    """获取应用运行时容器。"""

    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise RuntimeError("应用运行时未初始化")
    return runtime
