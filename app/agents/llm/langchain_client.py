"""LangChain LLM 客户端实现。"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.agents.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


class LangChainLLMClient(BaseLLMClient):
    """基于 LangChain 模型对象的统一客户端。"""

    def __init__(self, llm: Any, provider_name: str) -> None:
        self.llm = llm
        self.provider_name = provider_name

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def generate(self, system_prompt: str, user_prompt: str, metadata: dict[str, Any] | None = None) -> str:
        """调用 LangChain 模型并返回文本。"""

        del metadata
        if isinstance(self.llm, BaseChatModel):
            response = self.llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            content = self._extract_text(response.content)
        else:
            prompt = f"{system_prompt}\n\n{user_prompt}".strip()
            response = self.llm.invoke(prompt)
            content = response if isinstance(response, str) else self._extract_text(getattr(response, "content", response))

        if not content:
            logger.warning("%s 返回空内容", self.provider_name)
            return ""
        return content.strip()

    @staticmethod
    def _extract_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
                        continue
                chunks.append(str(item))
            return "".join(chunks)
        return str(content)
