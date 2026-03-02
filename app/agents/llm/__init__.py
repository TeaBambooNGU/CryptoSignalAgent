"""LLM 客户端工厂。"""

from __future__ import annotations

from app.agents.llm.base import BaseLLMClient
from app.agents.llm.openai_compatible import OpenAICompatibleLLMClient
from app.config.settings import Settings


def create_llm_client(settings: Settings) -> BaseLLMClient:
    """根据配置构建可替换 LLM 客户端。

    当前默认 `LLM_PROVIDER=minimax`，通过 OpenAI-compatible 协议接入。
    """

    provider = settings.llm_provider.strip().lower()

    if provider == "minimax":
        if not settings.minimax_api_key:
            raise ValueError("MINIMAX_API_KEY 未配置，无法初始化 LLM 客户端")

        return OpenAICompatibleLLMClient(
            api_key=settings.minimax_api_key,
            base_url=settings.minimax_base_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            timeout_seconds=settings.llm_timeout_seconds,
            provider_name="minimax",
        )

    if provider in {"openai_compatible", "openai-compatible"}:
        if not settings.openai_compatible_api_key or not settings.openai_compatible_base_url:
            raise ValueError("OPENAI_COMPATIBLE_API_KEY 或 OPENAI_COMPATIBLE_BASE_URL 未配置，无法初始化 LLM 客户端")

        return OpenAICompatibleLLMClient(
            api_key=settings.openai_compatible_api_key,
            base_url=settings.openai_compatible_base_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            timeout_seconds=settings.llm_timeout_seconds,
            provider_name="openai_compatible",
        )

    raise ValueError(f"未知 LLM_PROVIDER={provider}")
