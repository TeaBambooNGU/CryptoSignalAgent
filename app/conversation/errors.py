"""会话一致性相关异常。"""

from __future__ import annotations


class ConversationConflictError(RuntimeError):
    """CAS 版本冲突。"""

    def __init__(self, *, expected_version: int, current_version: int) -> None:
        super().__init__(f"conversation version conflict: expected={expected_version}, current={current_version}")
        self.expected_version = expected_version
        self.current_version = current_version


class DuplicateRequestInFlightError(RuntimeError):
    """同 request_id 请求仍在处理中。"""

