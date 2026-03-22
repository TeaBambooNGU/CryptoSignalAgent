"""上下文 token 估算工具。"""

from __future__ import annotations

from typing import Any


class TokenCounter:
    """统一封装 token 估算，优先使用 tiktoken。"""

    def __init__(self) -> None:
        self._encoding: Any | None = None
        try:
            import tiktoken  # type: ignore

            self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoding = None

    def count_text(self, text: str) -> int:
        normalized = str(text or "")
        if not normalized:
            return 0
        if self._encoding is not None:
            try:
                return len(self._encoding.encode(normalized))
            except Exception:
                pass
        # 兜底采用保守估算，避免缺失 tiktoken 时完全失效。
        return max(1, (len(normalized) + 2) // 3)

    def count_sections(self, sections: list[str]) -> int:
        return sum(self.count_text(item) for item in sections if item)
