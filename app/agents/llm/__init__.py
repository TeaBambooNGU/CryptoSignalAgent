"""LLM 客户端工厂。"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.agents.llm.base import BaseLLMClient
from app.agents.llm.langchain_client import LangChainLLMClient
from app.config.settings import Settings


def create_llm_client(settings: Settings) -> BaseLLMClient:
    """根据配置构建可替换 LLM 客户端。

    当前仅使用 LangChain 框架，默认 `LLM_PROVIDER=minimax`。
    """

    provider = settings.llm_provider.strip().lower()

    if provider == "minimax":
        if not settings.minimax_api_key:
            raise ValueError("MINIMAX_API_KEY 未配置，无法初始化 LLM 客户端")

        minimax_llm = ChatOpenAI(
            model_name=settings.llm_model,
            temperature=settings.llm_temperature,
            request_timeout=settings.llm_timeout_seconds,
            openai_api_key=settings.minimax_api_key,
            openai_api_base=f"{settings.minimax_api_host.rstrip('/')}/v1",
        )
        return LangChainLLMClient(llm=minimax_llm, provider_name="minimax")

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY 未配置，无法初始化 LLM 客户端")

        openai_llm = ChatOpenAI(
            model_name=settings.llm_model,
            temperature=settings.llm_temperature,
            request_timeout=settings.llm_timeout_seconds,
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.openai_base_url or None,
        )
        return LangChainLLMClient(llm=openai_llm, provider_name="openai")

    raise ValueError(f"未知 LLM_PROVIDER={provider}")
