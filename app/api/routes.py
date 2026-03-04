"""API 路由定义。"""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.api.deps import get_runtime
from app.config.logging import get_current_trace_id, get_logger, log_context
from app.conversation.errors import ConversationConflictError, DuplicateRequestInFlightError
from app.models.schemas import (
    ConversationAction,
    ConversationMessageRequest,
    ConversationMessageResponse,
    ConversationMetaResponse,
    ConversationReport,
    ConversationTurnDetail,
    ConversationTurnSummary,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    ResumeConversationRequest,
    UserPreferencesRequest,
    UserPreferencesResponse,
    UserProfileResponse,
)
from app.runtime import AppRuntime

router = APIRouter(prefix="/v1", tags=["crypto-signal-agent"])
logger = get_logger(__name__)


@router.post("/research/query", response_model=QueryResponse)
async def research_query(
    payload: QueryRequest,
    request: Request,
    runtime: AppRuntime = Depends(get_runtime),
) -> QueryResponse:
    """执行研报工作流。"""

    trace_id = getattr(request.state, "trace_id", "") or get_current_trace_id()
    if not trace_id or trace_id == "-":
        trace_id = str(uuid4())
    try:
        with log_context(trace_id=trace_id, user_id=payload.user_id, component="api.research_query"):
            logger.info("接收研报请求")
            return await runtime.conversation_service.run_research_turn(
                user_id=payload.user_id,
                query=payload.query,
                task_context=payload.task_context,
                trace_id=trace_id,
                conversation_id=payload.conversation_id,
                turn_id=payload.turn_id,
                request_id=payload.request_id,
                expected_version=payload.expected_version,
            )
    except ConversationConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "conversation_version_conflict",
                "expected_version": exc.expected_version,
                "current_version": exc.current_version,
            },
        ) from exc
    except DuplicateRequestInFlightError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "request_in_flight",
                "message": str(exc),
            },
        ) from exc
    except Exception as exc:
        logger.exception("研报请求失败")
        raise HTTPException(status_code=500, detail=f"研报生成失败: {type(exc).__name__}") from exc


@router.post("/conversation/{conversation_id}/message", response_model=ConversationMessageResponse)
async def conversation_message(
    conversation_id: str,
    payload: ConversationMessageRequest,
    request: Request,
    runtime: AppRuntime = Depends(get_runtime),
) -> ConversationMessageResponse:
    """会话统一消息入口。"""

    trace_id = getattr(request.state, "trace_id", "") or get_current_trace_id()
    if not trace_id or trace_id == "-":
        trace_id = str(uuid4())
    try:
        with log_context(trace_id=trace_id, user_id=payload.user_id, component="api.conversation_message"):
            return await runtime.conversation_service.send_message(
                user_id=payload.user_id,
                message=payload.message,
                conversation_id=conversation_id,
                trace_id=trace_id,
                task_context=payload.task_context,
                action=ConversationAction(payload.action),
                target_report_id=payload.target_report_id,
                from_turn_id=payload.from_turn_id,
                request_id=payload.request_id,
                expected_version=payload.expected_version,
            )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConversationConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "conversation_version_conflict",
                "expected_version": exc.expected_version,
                "current_version": exc.current_version,
            },
        ) from exc
    except DuplicateRequestInFlightError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "request_in_flight",
                "message": str(exc),
            },
        ) from exc
    except Exception as exc:
        logger.exception("会话消息处理失败")
        raise HTTPException(status_code=500, detail=f"会话处理失败: {type(exc).__name__}") from exc


@router.get("/conversation/{conversation_id}", response_model=ConversationMetaResponse)
def get_conversation_meta(
    conversation_id: str,
    runtime: AppRuntime = Depends(get_runtime),
) -> ConversationMetaResponse:
    """返回会话元信息。"""

    payload = runtime.conversation_service.get_conversation_meta(conversation_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return ConversationMetaResponse.model_validate(payload)


@router.get("/conversation/{conversation_id}/turns", response_model=list[ConversationTurnSummary])
def list_conversation_turns(
    conversation_id: str,
    limit: int = Query(20, ge=1, le=100),
    before_version: int | None = Query(default=None, ge=1),
    runtime: AppRuntime = Depends(get_runtime),
) -> list[ConversationTurnSummary]:
    """分页返回会话 turn 列表。"""

    rows = runtime.conversation_service.list_conversation_turns(
        conversation_id=conversation_id,
        limit=limit,
        before_version=before_version,
    )
    return [ConversationTurnSummary.model_validate(row) for row in rows]


@router.get("/conversation/{conversation_id}/reports", response_model=list[ConversationReport])
def list_conversation_reports(
    conversation_id: str,
    limit: int = Query(20, ge=1, le=100),
    before_report_version: int | None = Query(default=None, ge=1),
    runtime: AppRuntime = Depends(get_runtime),
) -> list[ConversationReport]:
    """分页返回会话报告版本列表。"""

    rows = runtime.conversation_service.list_conversation_reports(
        conversation_id=conversation_id,
        limit=limit,
        before_report_version=before_report_version,
    )
    return [ConversationReport.model_validate(row) for row in rows]


@router.get("/conversation/{conversation_id}/reports/{report_id}", response_model=ConversationReport)
def get_conversation_report(
    conversation_id: str,
    report_id: str,
    runtime: AppRuntime = Depends(get_runtime),
) -> ConversationReport:
    """返回单个报告版本。"""

    payload = runtime.conversation_service.get_conversation_report(report_id=report_id)
    if payload is None or payload.get("conversation_id") != conversation_id:
        raise HTTPException(status_code=404, detail="report not found")
    return ConversationReport.model_validate(payload)


@router.get("/conversation/{conversation_id}/turns/{turn_id}", response_model=ConversationTurnDetail)
def get_conversation_turn(
    conversation_id: str,
    turn_id: str,
    runtime: AppRuntime = Depends(get_runtime),
) -> ConversationTurnDetail:
    """返回单轮详情。"""

    payload = runtime.conversation_service.get_conversation_turn(
        conversation_id=conversation_id,
        turn_id=turn_id,
    )
    if payload is None:
        raise HTTPException(status_code=404, detail="turn not found")
    return ConversationTurnDetail.model_validate(payload)


@router.post("/conversation/{conversation_id}/resume", response_model=QueryResponse)
async def resume_conversation(
    conversation_id: str,
    payload: ResumeConversationRequest,
    request: Request,
    runtime: AppRuntime = Depends(get_runtime),
) -> QueryResponse:
    """基于历史会话继续执行下一轮（支持 from_turn_id 分支恢复）。"""

    trace_id = getattr(request.state, "trace_id", "") or get_current_trace_id()
    if not trace_id or trace_id == "-":
        trace_id = str(uuid4())

    try:
        with log_context(trace_id=trace_id, user_id=payload.user_id, component="api.resume_conversation"):
            return await runtime.conversation_service.resume_research_turn(
                user_id=payload.user_id,
                query=payload.query,
                conversation_id=conversation_id,
                trace_id=trace_id,
                task_context=payload.task_context,
                from_turn_id=payload.from_turn_id,
                request_id=payload.request_id,
                expected_version=payload.expected_version,
            )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConversationConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "conversation_version_conflict",
                "expected_version": exc.expected_version,
                "current_version": exc.current_version,
            },
        ) from exc
    except DuplicateRequestInFlightError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "request_in_flight",
                "message": str(exc),
            },
        ) from exc
    except Exception as exc:
        logger.exception("恢复会话请求失败")
        raise HTTPException(status_code=500, detail=f"会话恢复失败: {type(exc).__name__}") from exc


@router.post("/user/preferences", response_model=UserPreferencesResponse)
def update_preferences(
    payload: UserPreferencesRequest,
    runtime: AppRuntime = Depends(get_runtime),
) -> UserPreferencesResponse:
    """更新用户长期偏好。"""

    with log_context(user_id=payload.user_id, component="api.user_preferences"):
        runtime.memory_service.save_preference(
            user_id=payload.user_id,
            preference=payload.preference,
            confidence=payload.confidence,
        )
        logger.info("用户偏好已更新")
    return UserPreferencesResponse(success=True, user_id=payload.user_id)


@router.get("/user/profile/{user_id}", response_model=UserProfileResponse)
def get_profile(user_id: str, runtime: AppRuntime = Depends(get_runtime)) -> UserProfileResponse:
    """获取用户聚合画像。"""

    with log_context(user_id=user_id, component="api.user_profile"):
        profile = runtime.memory_service.get_user_profile(user_id)
        logger.info("用户画像已返回")
    return UserProfileResponse(**profile)


@router.post("/research/ingest", response_model=IngestResponse)
def ingest_documents(payload: IngestRequest, runtime: AppRuntime = Depends(get_runtime)) -> IngestResponse:
    """将 MCP 获得的结构化内容写入检索库。"""

    task_id = payload.task_id or str(uuid4())
    with log_context(task_id=task_id, user_id=payload.user_id, component="api.research_ingest"):
        inserted = runtime.research_service.ingest_documents(task_id=task_id, docs=payload.documents)
        logger.info("文档入库请求完成 docs=%s inserted=%s", len(payload.documents), inserted)
    return IngestResponse(success=True, inserted_chunks=inserted, task_id=task_id)
