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
    workflow_steps: list[WorkflowStep]

    errors: list[str]
    retry_count: int

    mcp_round: int
    mcp_max_rounds: int
    transient_grace_used: bool

    mcp_raw_plan_calls: list[dict[str, Any]]
    mcp_filtered_plan_calls: list[dict[str, Any]]
    mcp_admissible_calls_count: int
    mcp_filtered_out_calls: list[dict[str, Any]]
    mcp_ban_reasons: list[str]
    mcp_active_constraints_summary: list[str]

    mcp_round_rows: list[dict[str, Any]]
    mcp_round_successes: list[dict[str, Any]]
    mcp_round_failures: list[dict[str, Any]]
    mcp_failure_classes: dict[str, Any]
    mcp_rules: list[dict[str, Any]]
    mcp_new_rules_added: bool

    mcp_new_unique_signal_count: int
    mcp_signal_hash_set: list[str]
    mcp_raw_plan_hash_history: list[str]
    mcp_no_progress_streak: int

    mcp_should_continue: bool
    mcp_termination_reason: str
