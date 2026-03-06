"""LLM 客户端工厂。"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.config.settings import Settings


def _build_openai_compatible_client(
    *,
    model_name: str,
    temperature: float,
    timeout_seconds: int,
    api_key: str,
    base_url: str | None,
    missing_api_key_error: str,
) -> BaseChatModel:
    """构造 OpenAI-compatible ChatModel。"""

    resolved_api_key = api_key.strip()
    if not resolved_api_key:
        raise ValueError(missing_api_key_error)

    resolved_base_url = (base_url or "").strip() or None
    return ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        request_timeout=timeout_seconds,
        openai_api_key=resolved_api_key,
        openai_api_base=resolved_base_url,
    )


def create_deepseek_client(
    settings: Settings,
    *,
    model_name: str,
    timeout_seconds: int,
    temperature: float = 0,
) -> BaseChatModel:
    """构造 DeepSeek Chat 客户端。"""

    return _build_openai_compatible_client(
        model_name=model_name,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        missing_api_key_error="DEEPSEEK_API_KEY 未配置，无法初始化 DeepSeek 客户端",
    )


def create_llm_client(settings: Settings) -> BaseChatModel:
    """根据配置构建 LangChain 原生 ChatModel。

    当前支持 `minimax` 与 `openai`（均走 OpenAI-compatible 协议）。
    """

    provider = settings.llm_provider.strip().lower()

    if provider == "minimax":
        return _build_openai_compatible_client(
            model_name=settings.llm_model,
            temperature=settings.llm_temperature,
            timeout_seconds=settings.llm_timeout_seconds,
            api_key=settings.minimax_api_key,
            base_url=f"{settings.minimax_api_host.rstrip('/')}/v1",
            missing_api_key_error="MINIMAX_API_KEY 未配置，无法初始化 LLM 客户端",
        )

    if provider == "openai":
        return _build_openai_compatible_client(
            model_name=settings.llm_model,
            temperature=settings.llm_temperature,
            timeout_seconds=settings.llm_timeout_seconds,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            missing_api_key_error="OPENAI_API_KEY 未配置，无法初始化 LLM 客户端",
        )

    raise ValueError(f"未知 LLM_PROVIDER={provider}")
