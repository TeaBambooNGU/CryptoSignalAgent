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
    errors: list[str] = Field(default_factory=list, description="非致命错误信息")
    workflow_steps: list[WorkflowStep] = Field(default_factory=list, description="真实工作流执行轨迹")


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


class IngestRequest(BaseModel):
    """入库请求。"""

    user_id: str = Field(..., description="触发入库的用户")
    task_id: str | None = Field(default=None, description="可选任务 ID")
    documents: list[IngestDocument] = Field(default_factory=list, description="待入库文档列表")


class IngestResponse(BaseModel):
    """入库响应。"""

    success: bool = Field(..., description="是否成功")
    inserted_chunks: int = Field(..., description="写入 chunk 数")
    task_id: str = Field(..., description="任务 ID")


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
    retrieved_docs: list[RetrievedChunk] = Field(default_factory=list)


class ReportGenerationOutput(BaseModel):
    """报告生成输出。"""

    report: str
    citations: list[Citation] = Field(default_factory=list)
    draft: str = ""
