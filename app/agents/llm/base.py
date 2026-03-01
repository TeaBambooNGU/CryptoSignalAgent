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


class RuleBasedFallbackLLM(BaseLLMClient):
    """无密钥时的降级 LLM。

    用于本地联调，避免因外部模型不可用导致主流程中断。
    """

    def generate(self, system_prompt: str, user_prompt: str, metadata: dict[str, Any] | None = None) -> str:
        snippets = []
        for line in user_prompt.splitlines():
            line = line.strip()
            if line.startswith("- ") or line.startswith("1."):
                snippets.append(line)
        highlights = "\n".join(snippets[:8])
        if not highlights:
            highlights = user_prompt[:400]

        return (
            "## 一、执行摘要\n"
            "基于当前可用信号与历史证据，市场处于事件驱动与结构性分化并存阶段。\n\n"
            "## 二、关键证据\n"
            f"{highlights}\n\n"
            "## 三、风险与分歧\n"
            "- 数据覆盖度有限时，结论置信度会下降。\n"
            "- 若宏观或监管事件突发，短时波动可能显著放大。\n\n"
            "## 四、后续跟踪\n"
            "- 持续追踪资金费率、链上活跃度与主流媒体事件更新。"
        )
