"""本地启动脚本。"""

from __future__ import annotations

import uvicorn

from app.config.settings import settings


def main() -> None:
    """启动 FastAPI 服务。"""

    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "dev",
    )


if __name__ == "__main__":
    main()
