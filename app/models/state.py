"""LangGraph 状态定义。

约束：状态字段命名与设计文档第 10.1 节保持一致，
确保节点之间的输入输出契约稳定。
"""

from __future__ import annotations

from typing import Any, TypedDict

from app.models.schemas import Citation, NormalizedSignal, RetrievedChunk, WorkflowStep


class ResearchState(TypedDict, total=False):
    """研报工作流状态对象。"""

    user_id: str
    query: str
    task_context: dict[str, Any] | None
    conversation_id: str
    turn_id: str
    request_id: str
    conversation_version: int
    task_id: str
    trace_id: str

    memory_profile: dict[str, Any]
    symbols: list[str]
    hard_symbols: list[str]
    soft_symbols: list[str]

    raw_signals: list[dict[str, Any]]
    signals: list[NormalizedSignal]
    retrieved_docs: list[RetrievedChunk]

    analysis_summary: str
    report_draft: str
    final_report: str
    citations: list[Citation]
    workflow_steps: list[WorkflowStep]

    errors: list[str]
    retry_count: int

    mcp_tools_count: int
    mcp_termination_reason: str
