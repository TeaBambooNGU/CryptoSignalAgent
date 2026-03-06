"""API 路由定义。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from io import BytesIO
from uuid import uuid4
from xml.etree import ElementTree
from zipfile import ZipFile

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile

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
    IngestDocument,
    KnowledgeDocumentListResponse,
    KnowledgeDocumentRecord,
    KnowledgeDocumentRequest,
    KnowledgeDocumentResponse,
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


def _parse_iso_datetime(value: str | None) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    normalized = raw.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def _parse_csv_list(value: str | None) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    result: list[str] = []
    for item in raw.split(","):
        normalized = item.strip()
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def _extract_docx_text(raw_bytes: bytes) -> str:
    with ZipFile(BytesIO(raw_bytes)) as archive:
        xml_payload = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml_payload)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    parts = [node.text.strip() for node in root.findall(".//w:t", namespace) if node.text and node.text.strip()]
    return "\n".join(parts)


def _extract_file_text(file_name: str, raw_bytes: bytes) -> str:
    suffix = file_name.lower().rsplit(".", 1)[-1] if "." in file_name else ""
    if suffix in {"txt", "md"}:
        return raw_bytes.decode("utf-8")
    if suffix == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(raw_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == "docx":
        return _extract_docx_text(raw_bytes)
    raise ValueError(f"unsupported file type: {suffix or 'unknown'}")


def _build_knowledge_record_payload(
    *,
    doc_id: str,
    user_id: str,
    title: str,
    source: str,
    doc_type: str,
    symbols: list[str],
    tags: list[str],
    kb_id: str,
    language: str,
    published_at: datetime | None,
    checksum: str,
    chunk_count: int,
    file_name: str = "",
    content_type: str = "text/plain",
) -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "kb_id": kb_id or "default",
        "title": title,
        "source": source,
        "doc_type": doc_type,
        "symbols": symbols,
        "tags": tags,
        "language": language or "zh",
        "file_name": file_name,
        "content_type": content_type,
        "checksum": checksum,
        "status": "ready",
        "chunk_count": chunk_count,
        "uploaded_by": user_id,
        "published_at": int(published_at.timestamp()) if published_at is not None else None,
    }


def _ingest_knowledge_document(
    *,
    runtime: AppRuntime,
    user_id: str,
    task_id: str,
    title: str,
    source: str,
    doc_type: str,
    symbols: list[str],
    tags: list[str],
    text: str,
    kb_id: str,
    language: str,
    published_at: datetime | None,
    metadata: dict[str, object],
    file_name: str = "",
    content_type: str = "text/plain",
) -> tuple[int, dict[str, object]]:
    normalized_text = text.strip()
    if not normalized_text:
        raise HTTPException(status_code=400, detail="document text is empty")

    doc_id = str(uuid4())
    checksum = hashlib.sha1(normalized_text.encode("utf-8")).hexdigest()
    merged_metadata = {
        **metadata,
        "title": title,
        "doc_type": doc_type,
        "tags": tags,
        "language": language,
        "symbols": symbols,
        "kb_id": kb_id,
        "checksum": checksum,
        "file_name": file_name,
    }
    inserted = runtime.research_service.ingest_documents(
        task_id=task_id,
        docs=[
            IngestDocument(
                doc_id=doc_id,
                symbol=(symbols[0] if symbols else "GENERAL"),
                source=source,
                published_at=published_at or datetime.now(timezone.utc),
                text=normalized_text,
                metadata=merged_metadata,
            )
        ],
    )
    record = runtime.conversation_store.upsert_knowledge_document(
        _build_knowledge_record_payload(
            doc_id=doc_id,
            user_id=user_id,
            title=title,
            source=source,
            doc_type=doc_type,
            symbols=symbols,
            tags=tags,
            kb_id=kb_id,
            language=language,
            published_at=published_at,
            checksum=checksum,
            chunk_count=inserted,
            file_name=file_name,
            content_type=content_type,
        )
    )
    return inserted, record


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
            response = await runtime.conversation_service.run_research_turn(
                user_id=payload.user_id,
                query=payload.query,
                task_context=payload.task_context,
                trace_id=trace_id,
                conversation_id=payload.conversation_id,
                turn_id=payload.turn_id,
                request_id=payload.request_id,
                expected_version=payload.expected_version,
            )
            return response.model_dump(mode="json")
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
            response = await runtime.conversation_service.send_message(
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
            return response.model_dump(mode="json")
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
            response = await runtime.conversation_service.resume_research_turn(
                user_id=payload.user_id,
                query=payload.query,
                conversation_id=conversation_id,
                trace_id=trace_id,
                task_context=payload.task_context,
                from_turn_id=payload.from_turn_id,
                request_id=payload.request_id,
                expected_version=payload.expected_version,
            )
            return response.model_dump(mode="json")
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


@router.post("/knowledge/documents", response_model=KnowledgeDocumentResponse)
def create_knowledge_document(
    payload: KnowledgeDocumentRequest,
    runtime: AppRuntime = Depends(get_runtime),
) -> KnowledgeDocumentResponse:
    """创建知识库文档并写入知识向量库。"""

    task_id = payload.task_id or str(uuid4())
    document = payload.document
    with log_context(task_id=task_id, user_id=payload.user_id, component="api.knowledge_documents.create"):
        inserted, record = _ingest_knowledge_document(
            runtime=runtime,
            user_id=payload.user_id,
            task_id=task_id,
            title=document.title,
            source=document.source,
            doc_type=document.doc_type,
            symbols=document.symbols,
            tags=document.tags,
            text=document.text,
            kb_id=document.kb_id,
            language=document.language,
            published_at=document.published_at,
            metadata=document.metadata,
        )
        logger.info("知识库文档已创建 doc_id=%s chunks=%s", record.get("doc_id", ""), inserted)
    return KnowledgeDocumentResponse(
        success=True,
        task_id=task_id,
        inserted_chunks=inserted,
        document=KnowledgeDocumentRecord.model_validate(record),
    )


@router.post("/knowledge/upload", response_model=KnowledgeDocumentResponse)
async def upload_knowledge_document(
    user_id: str = Form(...),
    title: str = Form(...),
    source: str = Form(...),
    doc_type: str = Form("research_report"),
    symbols: str = Form(""),
    tags: str = Form(""),
    kb_id: str = Form("default"),
    language: str = Form("zh"),
    published_at: str = Form(""),
    metadata_json: str = Form("{}"),
    file: UploadFile = File(...),
    runtime: AppRuntime = Depends(get_runtime),
) -> KnowledgeDocumentResponse:
    """上传知识库文件并入库。"""

    task_id = str(uuid4())
    try:
        metadata = payload = json.loads(metadata_json or "{}")
        if not isinstance(payload, dict):
            raise ValueError("metadata_json must be an object")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid metadata_json: {exc}") from exc

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="uploaded file is empty")

    try:
        text = _extract_file_text(file.filename or "", raw_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"file parse failed: {exc}") from exc

    published_dt = _parse_iso_datetime(published_at)
    with log_context(task_id=task_id, user_id=user_id, component="api.knowledge_documents.upload"):
        inserted, record = _ingest_knowledge_document(
            runtime=runtime,
            user_id=user_id,
            task_id=task_id,
            title=title,
            source=source,
            doc_type=doc_type,
            symbols=_parse_csv_list(symbols),
            tags=_parse_csv_list(tags),
            text=text,
            kb_id=kb_id,
            language=language,
            published_at=published_dt,
            metadata=metadata,
            file_name=file.filename or "",
            content_type=file.content_type or "application/octet-stream",
        )
        logger.info("知识库文件已上传 doc_id=%s chunks=%s", record.get("doc_id", ""), inserted)
    return KnowledgeDocumentResponse(
        success=True,
        task_id=task_id,
        inserted_chunks=inserted,
        document=KnowledgeDocumentRecord.model_validate(record),
    )


@router.get("/knowledge/documents", response_model=KnowledgeDocumentListResponse)
def list_knowledge_documents(
    limit: int = Query(default=50, ge=1, le=200),
    kb_id: str | None = Query(default=None),
    runtime: AppRuntime = Depends(get_runtime),
) -> KnowledgeDocumentListResponse:
    """获取知识库文档列表。"""

    rows = runtime.conversation_store.list_knowledge_documents(limit=limit, kb_id=kb_id)
    return KnowledgeDocumentListResponse(items=[KnowledgeDocumentRecord.model_validate(row) for row in rows])


@router.get("/knowledge/documents/{doc_id}", response_model=KnowledgeDocumentRecord)
def get_knowledge_document(doc_id: str, runtime: AppRuntime = Depends(get_runtime)) -> KnowledgeDocumentRecord:
    """获取单个知识库文档详情。"""

    row = runtime.conversation_store.get_knowledge_document(doc_id=doc_id)
    if row is None or str(row.get("status", "")) == "deleted":
        raise HTTPException(status_code=404, detail="knowledge document not found")
    return KnowledgeDocumentRecord.model_validate(row)


@router.delete("/knowledge/documents/{doc_id}", response_model=KnowledgeDocumentRecord)
def delete_knowledge_document(doc_id: str, runtime: AppRuntime = Depends(get_runtime)) -> KnowledgeDocumentRecord:
    """软删除知识库文档并清理知识向量库。"""

    row = runtime.conversation_store.mark_knowledge_document_deleted(doc_id=doc_id)
    if row is None:
        raise HTTPException(status_code=404, detail="knowledge document not found")
    runtime.milvus_store.delete_knowledge_chunks_by_doc_id(doc_id)
    return KnowledgeDocumentRecord.model_validate(row)
