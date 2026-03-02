"""LLM 客户端抽象。

通过统一接口隔离具体供应商，支持配置化替换。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseLLMClient(ABC):
    """LLM 客户端统一抽象接口。"""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, metadata: dict[str, Any] | None = None) -> str:
        """生成文本结果。"""
