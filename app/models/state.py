"""LangGraph 状态定义。

约束：状态字段命名与设计文档第 10.1 节保持一致，
确保节点之间的输入输出契约稳定。
"""

from __future__ import annotations

from typing import Any, TypedDict

from app.models.schemas import Citation, NormalizedSignal, RetrievedChunk


class ResearchState(TypedDict, total=False):
    """研报工作流状态对象。"""

    user_id: str
    query: str
    task_context: dict[str, Any] | None
    task_id: str
    trace_id: str

    memory_profile: dict[str, Any]
    symbols: list[str]

    raw_signals: list[dict[str, Any]]
    signals: list[NormalizedSignal]
    retrieved_docs: list[RetrievedChunk]

    analysis_summary: str
    report_draft: str
    final_report: str
    citations: list[Citation]

    errors: list[str]
    retry_count: int
