"""LLM 客户端工厂。"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.config.settings import Settings


def create_llm_client(settings: Settings) -> BaseChatModel:
    """根据配置构建 LangChain 原生 ChatModel。

    当前支持 `minimax` 与 `openai`（均走 OpenAI-compatible 协议）。
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
        return minimax_llm

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
        return openai_llm

    raise ValueError(f"未知 LLM_PROVIDER={provider}")
