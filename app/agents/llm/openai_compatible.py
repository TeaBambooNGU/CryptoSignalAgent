"""OpenAI-compatible LLM 客户端实现。

当前默认用于 MiniMax M2.5，也可用于任意 OpenAI 兼容接口。
"""

from __future__ import annotations

import logging
from typing import Any

from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.agents.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


class OpenAICompatibleLLMClient(BaseLLMClient):
    """基于 OpenAI SDK 的兼容客户端。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.2,
        timeout_seconds: int = 60,
        provider_name: str = "openai_compatible",
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.provider_name = provider_name
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def generate(self, system_prompt: str, user_prompt: str, metadata: dict[str, Any] | None = None) -> str:
        """调用模型生成文本。

        重试策略说明：
        - 触发：限流、临时网络失败、服务瞬态错误。
        - 停止：最多 3 次。
        """

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            logger.warning("%s 返回空内容", self.provider_name)
            return ""
        return content.strip()
