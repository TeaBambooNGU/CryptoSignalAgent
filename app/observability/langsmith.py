"""LangSmith 可观测性配置。

职责：在应用启动时设置 LangSmith 环境变量，保证 LangChain/LangGraph 链路自动上报。
"""

from __future__ import annotations

import os

from app.config.settings import Settings


LANGSMITH_ENV_MAPPING = {
    "LANGCHAIN_TRACING_V2": "false",
    "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
    "LANGCHAIN_PROJECT": "crypto-signal-agent",
}


def configure_langsmith(settings: Settings) -> None:
    """根据配置启用或关闭 LangSmith 跟踪。"""

    if settings.langsmith_tracing:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        if settings.langsmith_api_key:
            os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        return

    for key, default in LANGSMITH_ENV_MAPPING.items():
        os.environ.setdefault(key, default)
