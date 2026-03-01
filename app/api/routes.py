"""API 路由定义。"""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_runtime
from app.models.schemas import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    UserPreferencesRequest,
    UserPreferencesResponse,
    UserProfileResponse,
)
from app.runtime import AppRuntime

router = APIRouter(prefix="/v1", tags=["crypto-signal-agent"])


@router.post("/research/query", response_model=QueryResponse)
def research_query(payload: QueryRequest, runtime: AppRuntime = Depends(get_runtime)) -> QueryResponse:
    """执行研报工作流。"""

    try:
        return runtime.graph_runner.run(
            user_id=payload.user_id,
            query=payload.query,
            task_context=payload.task_context,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"研报生成失败: {type(exc).__name__}") from exc


@router.post("/user/preferences", response_model=UserPreferencesResponse)
def update_preferences(
    payload: UserPreferencesRequest,
    runtime: AppRuntime = Depends(get_runtime),
) -> UserPreferencesResponse:
    """更新用户长期偏好。"""

    runtime.memory_service.save_preference(
        user_id=payload.user_id,
        preference=payload.preference,
        confidence=payload.confidence,
    )
    return UserPreferencesResponse(success=True, user_id=payload.user_id)


@router.get("/user/profile/{user_id}", response_model=UserProfileResponse)
def get_profile(user_id: str, runtime: AppRuntime = Depends(get_runtime)) -> UserProfileResponse:
    """获取用户聚合画像。"""

    profile = runtime.memory_service.get_user_profile(user_id)
    return UserProfileResponse(**profile)


@router.post("/research/ingest", response_model=IngestResponse)
def ingest_documents(payload: IngestRequest, runtime: AppRuntime = Depends(get_runtime)) -> IngestResponse:
    """将 MCP 获得的结构化内容写入检索库。"""

    task_id = payload.task_id or str(uuid4())
    inserted = runtime.research_service.ingest_documents(task_id=task_id, docs=payload.documents)
    return IngestResponse(success=True, inserted_chunks=inserted, task_id=task_id)
