"""核心数据模型定义。

本模块集中维护 API 协议、信号标准化结构与研报输出结构，
保证各层之间的数据边界清晰且可校验。
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SignalType(str, Enum):
    """信号类型枚举。"""

    PRICE = "price"
    NEWS = "news"
    SENTIMENT = "sentiment"
    ONCHAIN = "onchain"


class MemoryType(str, Enum):
    """记忆类型枚举。"""

    PREFERENCE = "preference"
    WATCHLIST = "watchlist"
    HABIT = "habit"
    CONTEXT = "context"
    TOOL_CORRECTION = "tool_correction"


class QueryRequest(BaseModel):
    """研报查询请求。"""

    user_id: str = Field(..., description="用户唯一标识")
    query: str = Field(..., description="研究问题")
    task_context: dict[str, Any] | None = Field(default=None, description="可选任务上下文")
    conversation_id: str | None = Field(default=None, description="会话 ID（可选，默认 user 级会话）")
    turn_id: str | None = Field(default=None, description="会话轮次 ID（可选）")
    request_id: str | None = Field(default=None, description="请求幂等键（可选）")
    expected_version: int | None = Field(default=None, ge=0, description="会话 CAS 版本（可选）")


class ResumeConversationRequest(BaseModel):
    """会话恢复并继续请求。"""

    user_id: str = Field(..., description="用户唯一标识")
    query: str = Field(..., description="研究问题")
    task_context: dict[str, Any] | None = Field(default=None, description="可选任务上下文")
    from_turn_id: str | None = Field(default=None, description="分支恢复锚点 turn_id（可选，默认沿当前最新继续）")
    request_id: str | None = Field(default=None, description="请求幂等键（可选）")
    expected_version: int | None = Field(default=None, ge=0, description="会话 CAS 版本（可选）")


class ConversationAction(str, Enum):
    """会话动作类型。"""

    AUTO = "auto"
    CHAT = "chat"
    REWRITE_REPORT = "rewrite_report"
    REGENERATE_REPORT = "regenerate_report"


class Citation(BaseModel):
    """引用信息。"""

    source: str = Field(..., description="引用来源")
    raw_ref: str = Field(..., description="原始链接或引用定位")
    snippet: str = Field(..., description="引用摘要")
    published_at: datetime | None = Field(default=None, description="发布时间（可选）")


class WorkflowStep(BaseModel):
    """工作流节点执行信息。"""

    node_id: str = Field(..., description="节点 ID")
    status: str = Field(..., description="执行状态：success/error")
    duration_ms: int = Field(..., ge=0, description="节点耗时（毫秒）")


class QueryResponse(BaseModel):
    """研报查询响应。"""

    report: str = Field(..., description="最终研报正文")
    citations: list[Citation] = Field(default_factory=list, description="引用列表")
    trace_id: str = Field(..., description="链路追踪 ID")
    conversation_id: str = Field(..., description="会话 ID")
    turn_id: str = Field(..., description="会话轮次 ID")
    request_id: str = Field(..., description="请求幂等键")
    conversation_version: int = Field(..., ge=1, description="会话版本号")
    errors: list[str] = Field(default_factory=list, description="非致命错误信息")
    workflow_steps: list[WorkflowStep] = Field(default_factory=list, description="真实工作流执行轨迹")


class ConversationMetaResponse(BaseModel):
    """会话元信息。"""

    conversation_id: str = Field(..., description="会话 ID")
    latest_version: int = Field(..., ge=0, description="最新会话版本")
    latest_turn_id: str = Field(..., description="最新 turn ID")
    turn_count: int = Field(..., ge=0, description="turn 总数")
    updated_at: int = Field(..., ge=0, description="更新时间（Unix 秒）")


class ConversationTurnSummary(BaseModel):
    """会话 turn 摘要。"""

    conversation_id: str = Field(..., description="会话 ID")
    turn_id: str = Field(..., description="轮次 ID")
    version: int = Field(..., ge=1, description="会话版本")
    request_id: str = Field(..., description="请求幂等键")
    user_id: str = Field(..., description="用户 ID")
    query: str = Field(..., description="用户问题")
    report: str = Field(..., description="研报正文")
    assistant_message: str = Field(default="", description="助手回复")
    trace_id: str = Field(..., description="链路追踪 ID")
    status: str = Field(..., description="turn 状态")
    intent: str = Field(default="", description="意图类型")
    turn_type: str = Field(default="", description="turn 类型")
    parent_turn_id: str | None = Field(default=None, description="父轮次 ID")
    report_id: str | None = Field(default=None, description="关联报告 ID")
    created_at: int = Field(..., ge=0, description="创建时间（Unix 秒）")
    updated_at: int = Field(..., ge=0, description="更新时间（Unix 秒）")


class ConversationTurnDetail(BaseModel):
    """会话 turn 完整详情。"""

    conversation_id: str = Field(..., description="会话 ID")
    turn_id: str = Field(..., description="轮次 ID")
    version: int = Field(..., ge=1, description="会话版本")
    request_id: str = Field(..., description="请求幂等键")
    user_id: str = Field(..., description="用户 ID")
    query: str = Field(..., description="用户问题")
    task_context: dict[str, Any] = Field(default_factory=dict, description="任务上下文")
    report: str = Field(..., description="研报正文")
    assistant_message: str = Field(default="", description="助手回复")
    citations: list[Citation] = Field(default_factory=list, description="引用列表")
    errors: list[str] = Field(default_factory=list, description="错误列表")
    workflow_steps: list[WorkflowStep] = Field(default_factory=list, description="工作流轨迹")
    trace_id: str = Field(..., description="链路追踪 ID")
    status: str = Field(..., description="turn 状态")
    intent: str = Field(default="", description="意图类型")
    turn_type: str = Field(default="", description="turn 类型")
    parent_turn_id: str | None = Field(default=None, description="父轮次 ID")
    report_id: str | None = Field(default=None, description="关联报告 ID")
    created_at: int = Field(..., ge=0, description="创建时间（Unix 秒）")
    updated_at: int = Field(..., ge=0, description="更新时间（Unix 秒）")


class ConversationReport(BaseModel):
    """会话报告版本。"""

    report_id: str = Field(..., description="报告 ID")
    conversation_id: str = Field(..., description="会话 ID")
    report_version: int = Field(..., ge=1, description="报告版本")
    created_by_turn_id: str = Field(..., description="产出该报告的 turn ID")
    based_on_report_id: str | None = Field(default=None, description="重写源报告 ID")
    mode: str = Field(..., description="报告生成模式")
    report: str = Field(..., description="报告正文")
    citations: list[Citation] = Field(default_factory=list, description="引用列表")
    workflow_steps: list[WorkflowStep] = Field(default_factory=list, description="工作流轨迹")
    status: str = Field(..., description="状态")
    created_at: int = Field(..., ge=0, description="创建时间（Unix 秒）")
    updated_at: int = Field(..., ge=0, description="更新时间（Unix 秒）")


class ConversationMessageRequest(BaseModel):
    """会话消息请求。"""

    user_id: str = Field(..., description="用户唯一标识")
    message: str = Field(..., description="用户消息")
    task_context: dict[str, Any] | None = Field(default=None, description="可选任务上下文")
    action: ConversationAction = Field(default=ConversationAction.AUTO, description="会话动作")
    target_report_id: str | None = Field(default=None, description="目标报告 ID（重写时可选）")
    from_turn_id: str | None = Field(default=None, description="分支锚点 turn_id（可选，用于从历史分叉续写）")
    request_id: str | None = Field(default=None, description="请求幂等键（可选）")
    expected_version: int | None = Field(default=None, ge=0, description="会话 CAS 版本（可选）")


class ConversationMessageResponse(BaseModel):
    """会话消息响应。"""

    conversation_id: str = Field(..., description="会话 ID")
    turn_id: str = Field(..., description="轮次 ID")
    request_id: str = Field(..., description="请求幂等键")
    conversation_version: int = Field(..., ge=1, description="会话版本")
    action_taken: ConversationAction = Field(..., description="实际执行动作")
    assistant_message: str = Field(..., description="助手回复")
    trace_id: str = Field(..., description="链路追踪 ID")
    errors: list[str] = Field(default_factory=list, description="错误列表")
    workflow_steps: list[WorkflowStep] = Field(default_factory=list, description="工作流轨迹")
    report: ConversationReport | None = Field(default=None, description="本轮产出的报告（若有）")


class UserPreferencesRequest(BaseModel):
    """用户偏好更新请求。"""

    user_id: str = Field(..., description="用户唯一标识")
    preference: dict[str, Any] = Field(..., description="偏好内容")
    confidence: float = Field(0.8, ge=0.0, le=1.0, description="偏好置信度")


class UserPreferencesResponse(BaseModel):
    """偏好更新结果。"""

    success: bool = Field(..., description="是否更新成功")
    user_id: str = Field(..., description="用户唯一标识")


class UserProfileResponse(BaseModel):
    """用户画像响应。"""

    user_id: str = Field(..., description="用户唯一标识")
    long_term_memory: list[dict[str, Any]] = Field(default_factory=list, description="长期记忆")
    session_memory: list[dict[str, Any]] = Field(default_factory=list, description="短期记忆")


class IngestDocument(BaseModel):
    """手动入库文档。

    该接口用于将“已通过 MCP 获取的内容”回填系统。
    """

    doc_id: str = Field(..., description="文档 ID")
    symbol: str = Field(..., description="关联标的")
    source: str = Field(..., description="来源")
    published_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="发布时间")
    text: str = Field(..., description="文档正文")
    metadata: dict[str, Any] = Field(default_factory=dict, description="附加元信息")


class KnowledgeDocumentCreate(BaseModel):
    """知识库文档创建请求。"""

    title: str = Field(..., description="文档标题")
    source: str = Field(..., description="文档来源")
    doc_type: str = Field(default="research_report", description="文档类型")
    symbols: list[str] = Field(default_factory=list, description="关联标的列表")
    tags: list[str] = Field(default_factory=list, description="标签列表")
    text: str = Field(..., description="文档正文")
    kb_id: str = Field(default="default", description="知识库 ID")
    language: str = Field(default="zh", description="文档语言")
    published_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="发布时间")
    metadata: dict[str, Any] = Field(default_factory=dict, description="附加元信息")


class KnowledgeDocumentRequest(BaseModel):
    """知识库文档入库请求。"""

    user_id: str = Field(..., description="触发入库的用户")
    task_id: str | None = Field(default=None, description="可选任务 ID")
    document: KnowledgeDocumentCreate = Field(..., description="待入库文档")


class KnowledgeDocumentRecord(BaseModel):
    """知识库文档元数据。"""

    doc_id: str = Field(..., description="文档 ID")
    kb_id: str = Field(..., description="知识库 ID")
    title: str = Field(..., description="文档标题")
    source: str = Field(..., description="文档来源")
    doc_type: str = Field(..., description="文档类型")
    symbols: list[str] = Field(default_factory=list, description="关联标的")
    tags: list[str] = Field(default_factory=list, description="标签")
    language: str = Field(default="zh", description="语言")
    file_name: str = Field(default="", description="原始文件名")
    content_type: str = Field(default="text/plain", description="内容类型")
    checksum: str = Field(default="", description="内容摘要")
    status: str = Field(default="ready", description="处理状态")
    chunk_count: int = Field(default=0, ge=0, description="chunk 数量")
    uploaded_by: str = Field(..., description="上传用户")
    published_at: datetime | None = Field(default=None, description="发布时间")
    created_at: int = Field(..., ge=0, description="创建时间")
    updated_at: int = Field(..., ge=0, description="更新时间")


class KnowledgeDocumentResponse(BaseModel):
    """知识库文档入库响应。"""

    success: bool = Field(..., description="是否成功")
    task_id: str = Field(..., description="任务 ID")
    inserted_chunks: int = Field(..., description="写入 chunk 数")
    document: KnowledgeDocumentRecord = Field(..., description="文档元数据")


class KnowledgeDocumentListResponse(BaseModel):
    """知识库文档列表响应。"""

    items: list[KnowledgeDocumentRecord] = Field(default_factory=list, description="文档列表")


class RawSignal(BaseModel):
    """MCP 原始信号结构。"""

    symbol: str
    source: str
    signal_type: SignalType
    value: Any
    raw_ref: str
    published_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class NormalizedSignal(BaseModel):
    """标准化信号结构。

    字段与设计文档约定保持一致，供后续检索与报告节点复用。
    """

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    symbol: str = Field(..., description="标的，如 BTC/ETH")
    source: str = Field(..., description="来源标识")
    signal_type: SignalType = Field(..., description="信号类型")
    value: Any = Field(..., description="原始值或抽取结果")
    confidence: float = Field(0.6, ge=0.0, le=1.0, description="置信度")
    raw_ref: str = Field(..., description="原文引用或来源链接")
    ingest_mode: str = Field(default="mcp", description="采集模式，V1 固定 mcp")
    task_id: str = Field(..., description="任务链路 ID")


class RetrievedChunk(BaseModel):
    """检索结果片段。"""

    chunk_id: str
    doc_id: str
    symbol: str
    source: str
    text: str
    score: float
    published_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReportGenerationInput(BaseModel):
    """报告生成输入。"""

    user_id: str
    query: str
    task_id: str
    memory_profile: dict[str, Any] = Field(default_factory=dict)
    signals: list[NormalizedSignal] = Field(default_factory=list)
    knowledge_docs: list[RetrievedChunk] = Field(default_factory=list)


class ReportGenerationOutput(BaseModel):
    """报告生成输出。"""

    report: str
    citations: list[Citation] = Field(default_factory=list)
    draft: str = ""


def ensure_citations(items: list[Any] | None) -> list[Citation]:
    """将任意 citation 列表归一化为 `Citation` 模型列表。"""

    return [Citation.model_validate(item) for item in (items or [])]


def ensure_workflow_steps(items: list[Any] | None) -> list[WorkflowStep]:
    """将任意 workflow step 列表归一化为 `WorkflowStep` 模型列表。"""

    return [WorkflowStep.model_validate(item) for item in (items or [])]


def ensure_conversation_report(item: Any | None) -> ConversationReport | None:
    """将任意报告对象归一化为 `ConversationReport`。"""

    if item is None:
        return None
    return ConversationReport.model_validate(item)
