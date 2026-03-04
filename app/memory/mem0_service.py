"""Mem0 记忆服务。

职责：
- 封装长期记忆（偏好/习惯/观察列表）与短期任务上下文。
- 优先调用 Mem0；若不可用则降级到 Milvus + 本地会话内存。
- 明确长期与短期写回规则，符合设计文档约束。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config.settings import Settings
from app.models.schemas import MemoryType
from app.memory.session_store import SessionMemoryStore
from app.retrieval.embedding import text_to_embedding
from app.retrieval.milvus_store import MilvusStore

logger = logging.getLogger(__name__)


class MemoryService:
    """统一记忆服务。"""

    PREFERENCE_PROFILE_CONTENT_SEED = "__profile_v1__"

    def __init__(
        self,
        settings: Settings,
        milvus_store: MilvusStore,
        session_store: SessionMemoryStore,
        outbox_store: Any | None = None,
        conversation_store: Any | None = None,
    ) -> None:
        self.settings = settings
        self.milvus_store = milvus_store
        self.session_store = session_store
        self.outbox_store = outbox_store
        self.conversation_store = conversation_store
        self._mem0_client: Any | None = self._init_mem0_client()
        self._preference_extractor_client: Any | None = self._init_preference_extractor_client()

    def _init_mem0_client(self) -> Any | None:
        """按配置尝试初始化 Mem0 客户端。"""

        if not self.settings.mem0_enabled:
            return None
        mode = self._mem0_mode()

        try:
            import mem0  # type: ignore

            if mode == "platform":
                if not self.settings.mem0_api_key:
                    logger.warning("MEM0_MODE=platform 但未配置 MEM0_API_KEY，将使用降级模式")
                    return None
                client_cls = getattr(mem0, "MemoryClient", None)
                if client_cls is None:
                    logger.warning("mem0 包未提供 MemoryClient，自动降级")
                    return None
                client = client_cls(
                    api_key=self.settings.mem0_api_key,
                    org_id=self.settings.mem0_org_id or None,
                    project_id=self.settings.mem0_project_id or None,
                )
                logger.info("Mem0 MemoryClient 初始化成功（mode=platform）")
                return client

            memory_cls = getattr(mem0, "Memory", None)
            if memory_cls is None:
                logger.warning("mem0 包未提供 Memory（mode=oss），自动降级")
                return None

            if not self.settings.milvus_enabled:
                logger.warning("MEM0_MODE=oss 需要 Milvus，可当前 MILVUS_ENABLED=false，自动降级")
                return None

            from_config = getattr(memory_cls, "from_config", None)
            if callable(from_config):
                client = from_config(self._build_mem0_oss_config())
            else:
                # 极老版本兼容：若无 from_config，则退化到默认构造（不保证使用工程 Milvus）。
                client = memory_cls()
                logger.warning("mem0.Memory 未提供 from_config，OSS 无法保证复用工程 Milvus 配置")
            logger.info("Mem0 Memory 初始化成功（mode=oss, vector_store=milvus)")
            return client
        except Exception:
            logger.exception("Mem0 客户端初始化失败，自动降级")
            return None

    def _mem0_mode(self) -> str:
        """返回规范化后的 Mem0 运行模式。"""

        raw_mode = self.settings.mem0_mode.strip().lower()
        if raw_mode in {"platform", "oss"}:
            return raw_mode
        logger.warning("MEM0_MODE=%s 非法，回退为 platform", self.settings.mem0_mode)
        return "platform"

    def _build_mem0_oss_config(self) -> dict[str, Any]:
        """构造 Mem0 OSS 配置，复用工程现有 Milvus。"""

        llm_config: dict[str, Any] = {
            "model": self.settings.llm_model,
        }
        embedder_config: dict[str, Any] = {
            "model": self.settings.zhipu_embedding_model,
            "embedding_dims": self.settings.vector_dim,
        }

        # LLM 默认优先复用工程 Minimax 凭据；其次使用 OpenAI 凭据。
        llm_api_key = self.settings.minimax_api_key or self.settings.openai_api_key or os.getenv("OPENAI_API_KEY", "")
        minimax_openai_base = (
            f"{self.settings.minimax_api_host.rstrip('/')}/v1"
            if self.settings.minimax_api_host
            else ""
        )
        llm_base_url = minimax_openai_base or self.settings.openai_base_url or os.getenv("OPENAI_BASE_URL", "")
        if llm_api_key:
            llm_config["api_key"] = llm_api_key
        if llm_base_url:
            llm_config["openai_base_url"] = llm_base_url

        # Embedder 走工程现有智谱配置，默认使用智谱兼容 OpenAI 的 endpoint。
        embedder_api_key = os.getenv("ZHIPUAI_API_KEY", "") or self.settings.openai_api_key or os.getenv("OPENAI_API_KEY", "")
        embedder_base_url = (
            os.getenv("ZHIPU_OPENAI_BASE_URL", "")
            or self.settings.openai_base_url
            or "https://open.bigmodel.cn/api/paas/v4"
        )
        if embedder_api_key:
            embedder_config["api_key"] = embedder_api_key
        if embedder_base_url:
            embedder_config["openai_base_url"] = embedder_base_url

        milvus_config: dict[str, Any] = {
            "url": self.settings.milvus_uri,
            "token": self.settings.milvus_token or None,
            "collection_name": self.settings.mem0_oss_collection,
            "embedding_model_dims": self.settings.vector_dim,
            "metric_type": "COSINE",
            "db_name": self.settings.milvus_db_name,
        }

        return {
            "vector_store": {
                "provider": "milvus",
                "config": milvus_config,
            },
            "llm": {
                "provider": "openai",
                "config": llm_config,
            },
            "embedder": {
                "provider": "openai",
                "config": embedder_config,
            },
            "version": "v1.1",
        }

    def _init_preference_extractor_client(self) -> Any | None:
        """初始化长期偏好抽取小模型（DeepSeek）。"""

        api_key = self.settings.deepseek_api_key.strip()
        if not api_key:
            logger.warning("DEEPSEEK_API_KEY 未配置，自动跳过研报偏好抽取")
            return None

        try:
            base_url = (self.settings.deepseek_base_url or "https://api.deepseek.com/v1").rstrip("/")
            client = ChatOpenAI(
                model_name=self.settings.memory_extractor_model,
                temperature=0,
                request_timeout=self.settings.memory_extractor_timeout_seconds,
                openai_api_key=api_key,
                openai_api_base=base_url,
            )
            logger.info("长期偏好抽取模型初始化成功 model=%s", self.settings.memory_extractor_model)
            return client
        except Exception:
            logger.exception("长期偏好抽取模型初始化失败，将跳过自动抽取")
            return None

    def _build_memory_id(self, user_id: str, memory_type: MemoryType, content: str) -> str:
        """生成幂等记忆 ID。"""

        digest = hashlib.sha1(f"{user_id}:{memory_type.value}:{content}".encode("utf-8")).hexdigest()
        return digest[:40]

    @staticmethod
    def build_preference_profile_id(user_id: str) -> str:
        """生成“单条长期偏好画像”固定 ID。"""

        digest = hashlib.sha1(
            f"{user_id}:{MemoryType.PREFERENCE.value}:{MemoryService.PREFERENCE_PROFILE_CONTENT_SEED}".encode("utf-8")
        ).hexdigest()
        return digest[:40]

    def _event_ids(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
        turn_id: str | None,
        request_id: str | None,
    ) -> tuple[str, str, str]:
        final_conversation_id = (conversation_id or "").strip() or f"default:{user_id}"
        final_request_id = (request_id or "").strip() or str(uuid4())
        final_turn_id = (turn_id or "").strip() or f"turn-{final_request_id[:8]}"
        return final_conversation_id, final_turn_id, final_request_id

    def _append_session_memory(self, conversation_id: str, item: dict[str, Any]) -> None:
        """写入短期记忆，保留最近 50 条。"""

        self.session_store.append(conversation_id=conversation_id, item=item)

    def save_preference(
        self,
        user_id: str,
        preference: dict[str, Any],
        confidence: float = 0.8,
        conversation_id: str | None = None,
        turn_id: str | None = None,
        request_id: str | None = None,
    ) -> None:
        """保存长期偏好。

        长期记忆判定规则：
        - 用户显式提交的偏好视为长期可复用信息。
        - 采用“时间戳 + 置信度”覆盖策略。
        """

        normalized_preference = self._normalize_preference_payload(preference)
        if not normalized_preference:
            return

        updated_at = int(datetime.now(timezone.utc).timestamp())
        final_conversation_id, final_turn_id, final_request_id = self._event_ids(
            user_id=user_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            request_id=request_id,
        )

        if self.outbox_store is not None:
            self.outbox_store.enqueue_outbox_event(
                conversation_id=final_conversation_id,
                turn_id=final_turn_id,
                request_id=final_request_id,
                event_type="memory.save_preference",
                payload={
                    "user_id": user_id,
                    "preference": normalized_preference,
                    "confidence": confidence,
                    "updated_at": updated_at,
                    "request_id": final_request_id,
                },
            )
        else:
            self.apply_preference_write(
                user_id=user_id,
                preference=normalized_preference,
                confidence=confidence,
                updated_at=updated_at,
            )

        self._append_session_memory(
            final_conversation_id,
            {
                "memory_type": MemoryType.PREFERENCE.value,
                "content": normalized_preference,
                "confidence": confidence,
                "updated_at": updated_at,
            },
        )

    def save_task_context(
        self,
        user_id: str,
        conversation_id: str,
        task_context: dict[str, Any] | None,
    ) -> None:
        """保存短期任务上下文。"""

        del user_id
        if not task_context:
            return
        self._append_session_memory(
            conversation_id,
            {
                "memory_type": MemoryType.CONTEXT.value,
                "content": task_context,
                "confidence": 0.6,
                "updated_at": int(datetime.now(timezone.utc).timestamp()),
            },
        )

    def save_tool_correction(
        self,
        user_id: str,
        correction: dict[str, Any],
        confidence: float = 0.8,
        conversation_id: str | None = None,
        turn_id: str | None = None,
        request_id: str | None = None,
    ) -> None:
        """保存 MCP 工具纠错经验到长期记忆。"""

        if not correction:
            return
        final_conversation_id, final_turn_id, final_request_id = self._event_ids(
            user_id=user_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            request_id=request_id,
        )
        payload = {"kind": "mcp_tool_correction", **correction}
        updated_at = int(datetime.now(timezone.utc).timestamp())

        if self.outbox_store is not None:
            self.outbox_store.enqueue_outbox_event(
                conversation_id=final_conversation_id,
                turn_id=final_turn_id,
                request_id=final_request_id,
                event_type="memory.save_tool_correction",
                payload={
                    "user_id": user_id,
                    "correction": correction,
                    "confidence": confidence,
                    "updated_at": updated_at,
                    "request_id": final_request_id,
                },
            )
        else:
            self.apply_tool_correction_write(
                user_id=user_id,
                correction=correction,
                confidence=confidence,
                updated_at=updated_at,
            )

        self._append_session_memory(
            final_conversation_id,
            {
                "memory_type": MemoryType.TOOL_CORRECTION.value,
                "content": payload,
                "confidence": confidence,
                "updated_at": updated_at,
            },
        )

    def load_memory_profile(
        self,
        user_id: str,
        conversation_id: str | None = None,
        context_anchor_turn_id: str | None = None,
    ) -> dict[str, Any]:
        """聚合用户长期记忆与短期记忆。"""

        long_term = self.milvus_store.query_user_memory(user_id=user_id, limit=100)
        session = self.session_store.get(conversation_id=conversation_id or user_id, limit=50)
        recent_turn_context = self.load_recent_turn_context(
            conversation_id=conversation_id,
            limit=8,
            context_anchor_turn_id=context_anchor_turn_id,
        )
        conversation_summary = self.load_conversation_summary(
            conversation_id=conversation_id,
            context_anchor_turn_id=context_anchor_turn_id,
        )

        profile = {
            "user_id": user_id,
            "long_term_memory": self._compact_long_term_memory(user_id=user_id, long_term=long_term),
            "session_memory": session,
            "recent_turn_context": recent_turn_context,
            "conversation_summary": conversation_summary,
            "watchlist": self._extract_watchlist(long_term, session),
            "risk_preference": self._extract_risk_preference(long_term),
            "reading_habit": self._extract_reading_habit(long_term),
            "tool_corrections": self._extract_tool_corrections(long_term),
        }

        # Mem0 可用时补充最近记忆片段，提高上下文完整度。
        mem0_recent = self._mem0_search(user_id=user_id, query="recent user preference", limit=5)
        if mem0_recent:
            profile["mem0_recent"] = mem0_recent
        return profile

    def load_recent_turn_context(
        self,
        conversation_id: str | None,
        limit: int = 8,
        context_anchor_turn_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """从会话真相库读取最近轮次上下文。"""

        if not conversation_id or self.conversation_store is None:
            return []
        try:
            final_limit = max(1, min(limit, 20))
            rows = (
                self.conversation_store.list_turn_lineage(
                    conversation_id=conversation_id,
                    leaf_turn_id=context_anchor_turn_id,
                    limit=final_limit,
                )
                if context_anchor_turn_id
                else self.conversation_store.list_turns(
                    conversation_id=conversation_id,
                    limit=final_limit,
                )
            )
        except Exception:
            logger.exception("读取会话 turn 上下文失败，已降级为空")
            return []

        result: list[dict[str, Any]] = []
        for row in rows:
            result.append(
                {
                    "turn_id": str(row.get("turn_id", "")),
                    "version": int(row.get("version", 0) or 0),
                    "query": str(row.get("query", "")),
                    "report": str(row.get("report", "")),
                    "status": str(row.get("status", "")),
                    "trace_id": str(row.get("trace_id", "")),
                    "updated_at": int(row.get("updated_at", 0) or 0),
                }
            )
        return result

    def load_conversation_summary(
        self,
        conversation_id: str | None,
        context_anchor_turn_id: str | None = None,
    ) -> dict[str, Any] | None:
        """读取会话摘要压缩结果。"""

        if not conversation_id or self.conversation_store is None:
            return None
        try:
            if context_anchor_turn_id:
                lineage_turns = self.conversation_store.list_turn_lineage(
                    conversation_id=conversation_id,
                    leaf_turn_id=context_anchor_turn_id,
                    limit=24,
                )
                if len(lineage_turns) <= 8:
                    return None
                older_turns = lineage_turns[8:]
                if not older_turns:
                    return None
                lines: list[str] = []
                for turn in reversed(older_turns[:16]):
                    query = str(turn.get("query", "")).strip().replace("\n", " ")
                    answer = str(turn.get("assistant_message", "")).strip().replace("\n", " ")
                    if len(answer) > 120:
                        answer = answer[:120] + "..."
                    lines.append(f"- v{turn.get('version')}[{turn.get('intent', '')}] Q:{query} A:{answer}")
                return {
                    "through_version": int(older_turns[0].get("version", 0) or 0),
                    "summary_text": "\n".join(lines),
                    "updated_at": int(lineage_turns[0].get("updated_at", 0) or 0),
                }
            row = self.conversation_store.get_context_summary(conversation_id=conversation_id)
            if not row:
                return None
            return {
                "through_version": int(row.get("through_version", 0) or 0),
                "summary_text": str(row.get("summary_text", "")),
                "updated_at": int(row.get("updated_at", 0) or 0),
            }
        except Exception:
            logger.exception("读取会话摘要失败，已降级为空")
            return None

    def persist_report_memory(
        self,
        user_id: str,
        query: str,
        report: str,
        inferred_preferences: dict[str, Any] | None = None,
        conversation_id: str | None = None,
        turn_id: str | None = None,
        request_id: str | None = None,
    ) -> None:
        """从研报结果提取可复用偏好并写回长期记忆。

        规则说明：
        - 仅写回“跨任务可复用”的偏好，例如关注标的、风险偏好、格式偏好。
        - 一次性任务上下文不写入长期记忆，转入短期会话缓存。
        """

        extracted = (
            inferred_preferences
            if inferred_preferences is not None
            else self._extract_preferences_with_llm(query=query, report=report)
        )
        extracted = self._normalize_preference_payload(extracted)
        if not extracted:
            return

        self.save_preference(
            user_id=user_id,
            preference=extracted,
            confidence=0.7,
            conversation_id=conversation_id,
            turn_id=turn_id,
            request_id=request_id,
        )

    def apply_preference_write(
        self,
        *,
        user_id: str,
        preference: dict[str, Any],
        confidence: float,
        updated_at: int,
    ) -> None:
        """将偏好写入长期存储（Milvus + Mem0）。"""

        normalized = self._normalize_preference_payload(preference)
        if not normalized:
            return

        historical = self.milvus_store.query_user_memory(user_id=user_id, limit=200)
        existing_profile = self._extract_preference_profile(historical)
        merged_profile = self._merge_preference_payload(existing_profile, normalized)
        if not merged_profile:
            return

        content = json.dumps(merged_profile, ensure_ascii=False, sort_keys=True)
        embedding = text_to_embedding(content, self.settings)
        record = {
            "id": self.build_preference_profile_id(user_id),
            "user_id": user_id,
            "memory_type": MemoryType.PREFERENCE.value,
            "content": content,
            "confidence": confidence,
            "updated_at": int(updated_at or datetime.now(timezone.utc).timestamp()),
            "embedding": embedding,
        }
        self.milvus_store.upsert_user_memory([record])
        stale_ids = [
            str(row.get("id", "")).strip()
            for row in historical
            if str(row.get("memory_type", "")) == MemoryType.PREFERENCE.value
            and str(row.get("id", "")).strip()
            and str(row.get("id", "")).strip() != record["id"]
        ]
        if stale_ids:
            delete_fn = getattr(self.milvus_store, "delete_user_memory_by_ids", None)
            if callable(delete_fn):
                try:
                    delete_fn(stale_ids)
                except Exception:
                    logger.exception("清理历史 preference 冗余记录失败")
        self._mem0_add(user_id=user_id, content=content, metadata={"memory_type": MemoryType.PREFERENCE.value})

    def apply_tool_correction_write(
        self,
        *,
        user_id: str,
        correction: dict[str, Any],
        confidence: float,
        updated_at: int,
    ) -> None:
        """将工具纠错经验写入长期存储（Milvus + Mem0）。"""

        if not correction:
            return
        payload = {"kind": "mcp_tool_correction", **correction}
        content = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        embedding = text_to_embedding(content, self.settings)
        record = {
            "id": self._build_memory_id(user_id, MemoryType.TOOL_CORRECTION, content),
            "user_id": user_id,
            "memory_type": MemoryType.TOOL_CORRECTION.value,
            "content": content,
            "confidence": confidence,
            "updated_at": int(updated_at or datetime.now(timezone.utc).timestamp()),
            "embedding": embedding,
        }
        self.milvus_store.upsert_user_memory([record])
        self._mem0_add(
            user_id=user_id,
            content=content,
            metadata={
                "memory_type": MemoryType.TOOL_CORRECTION.value,
                "server": str(payload.get("server", "")),
            },
        )

    def get_user_profile(self, user_id: str) -> dict[str, Any]:
        """返回 API 需要的用户画像结构。"""

        profile = self.load_memory_profile(user_id=user_id, conversation_id=user_id)
        return {
            "user_id": user_id,
            "long_term_memory": profile.get("long_term_memory", []),
            "session_memory": profile.get("session_memory", []),
        }

    def _decode_memory_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """将存储层记录转换为可读结构。"""

        content_raw = row.get("content", "")
        content: Any = content_raw
        try:
            content = json.loads(content_raw)
        except Exception:
            pass
        return {
            "id": row.get("id", ""),
            "memory_type": row.get("memory_type", ""),
            "content": content,
            "confidence": float(row.get("confidence", 0.0)),
            "updated_at": int(row.get("updated_at", 0)),
        }

    def _extract_watchlist(self, long_term: list[dict[str, Any]], session: list[dict[str, Any]]) -> list[str]:
        """从记忆中提取关注标的。"""

        profile = self._extract_preference_profile(long_term)
        watchlist = profile.get("watchlist")
        if isinstance(watchlist, list):
            return [str(item).upper() for item in watchlist]

        # 兼容极端降级场景：当长期画像为空时，回退到 session 中的临时偏好。
        candidates: set[str] = set()
        for row in session:
            parsed = self._coerce_dict(row.get("content"))
            if not parsed:
                continue
            raw_watchlist = parsed.get("watchlist")
            if isinstance(raw_watchlist, list):
                candidates.update(str(item).upper() for item in raw_watchlist)
        return sorted(candidates)

    def _extract_risk_preference(self, long_term: list[dict[str, Any]]) -> str | None:
        """提取用户风险偏好。"""

        profile = self._extract_preference_profile(long_term)
        risk = profile.get("risk_preference")
        return str(risk) if isinstance(risk, str) and risk else None

    def _extract_reading_habit(self, long_term: list[dict[str, Any]]) -> str | None:
        """提取用户阅读偏好。"""

        profile = self._extract_preference_profile(long_term)
        reading_habit = profile.get("reading_habit")
        return str(reading_habit) if isinstance(reading_habit, str) and reading_habit else None

    def _extract_tool_corrections(self, long_term: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """提取长期纠错经验，供 MCP planner 作为 few-shot 上下文。"""

        corrections: list[tuple[int, dict[str, Any]]] = []
        for row in long_term:
            if str(row.get("memory_type", "")) != MemoryType.TOOL_CORRECTION.value:
                continue
            content = row.get("content")
            parsed = content
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                except Exception:
                    parsed = {}
            if isinstance(parsed, dict):
                corrections.append((int(row.get("updated_at", 0) or 0), parsed))
        corrections.sort(key=lambda item: item[0], reverse=True)
        return [item for _, item in corrections[:12]]

    def _extract_preferences_with_llm(self, query: str, report: str) -> dict[str, Any]:
        """调用小模型抽取长期可复用偏好。"""

        if self._preference_extractor_client is None:
            return {}

        system_prompt = (
            "你是用户长期偏好抽取器。"
            "请只抽取可跨任务复用的偏好，并输出严格 JSON 对象。"
            "允许字段：watchlist(字符串数组)、risk_preference(字符串)、reading_habit(字符串)。"
            "若没有可复用偏好，返回空对象 {}。"
            "禁止输出 Markdown、解释文字、注释。"
        )
        user_prompt = (
            f"用户问题:\n{query}\n\n"
            f"研报内容:\n{report}\n\n"
            "请返回 JSON。"
        )

        try:
            response = self._preference_extractor_client.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            text = self._extract_text(getattr(response, "content", response))
            parsed = self._parse_json_object(text)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            logger.exception("LLM 偏好抽取失败，已跳过本次写回")
            return {}

    @staticmethod
    def _normalize_preference_payload(payload: Any) -> dict[str, Any]:
        """清洗 LLM 偏好抽取结果，避免脏数据入库。"""

        if not isinstance(payload, dict):
            return {}

        normalized: dict[str, Any] = {}

        watchlist_raw = payload.get("watchlist")
        if isinstance(watchlist_raw, list):
            watchlist: list[str] = []
            for item in watchlist_raw:
                token = str(item).strip().upper().lstrip("$")
                if not token or not re.fullmatch(r"[A-Z0-9._-]{2,20}", token):
                    continue
                if token not in watchlist:
                    watchlist.append(token)
            if watchlist:
                normalized["watchlist"] = watchlist

        risk_raw = str(payload.get("risk_preference", "")).strip().lower()
        if risk_raw in {"conservative", "balanced", "aggressive"}:
            normalized["risk_preference"] = risk_raw

        reading_habit_raw = str(payload.get("reading_habit", "")).strip().lower()
        if reading_habit_raw and re.fullmatch(r"[a-z_]{2,32}", reading_habit_raw):
            normalized["reading_habit"] = reading_habit_raw

        return normalized

    @staticmethod
    def _merge_preference_payload(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
        """按规则合并偏好：watchlist 并集，risk/reading 覆盖。"""

        merged: dict[str, Any] = {}
        base_watchlist = base.get("watchlist")
        incoming_watchlist = incoming.get("watchlist")
        merged_watchlist: list[str] = []

        if isinstance(base_watchlist, list):
            for item in base_watchlist:
                token = str(item).strip().upper()
                if token and token not in merged_watchlist:
                    merged_watchlist.append(token)
        if isinstance(incoming_watchlist, list):
            for item in incoming_watchlist:
                token = str(item).strip().upper()
                if token and token not in merged_watchlist:
                    merged_watchlist.append(token)
        if merged_watchlist:
            merged["watchlist"] = merged_watchlist

        risk = incoming.get("risk_preference") or base.get("risk_preference")
        if isinstance(risk, str) and risk:
            merged["risk_preference"] = risk

        reading_habit = incoming.get("reading_habit") or base.get("reading_habit")
        if isinstance(reading_habit, str) and reading_habit:
            merged["reading_habit"] = reading_habit

        return merged

    def _compact_long_term_memory(self, *, user_id: str, long_term: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """将长期记忆压缩为“偏好画像单条记录”。"""

        profile = self._extract_preference_profile(long_term)
        compact: list[dict[str, Any]] = []
        if profile:
            latest_updated_at = max(
                int(row.get("updated_at", 0) or 0)
                for row in long_term
                if str(row.get("memory_type", "")) == MemoryType.PREFERENCE.value
            )
            latest_confidence = max(
                float(row.get("confidence", 0.0) or 0.0)
                for row in long_term
                if str(row.get("memory_type", "")) == MemoryType.PREFERENCE.value
            )
            compact.append(
                {
                    "id": self.build_preference_profile_id(user_id),
                    "memory_type": MemoryType.PREFERENCE.value,
                    "content": profile,
                    "confidence": latest_confidence,
                    "updated_at": latest_updated_at,
                }
            )
        return compact

    def _extract_preference_profile(self, long_term: list[dict[str, Any]]) -> dict[str, Any]:
        """从长期记忆中提取并合并偏好画像。"""

        preference_rows = [
            row for row in long_term if str(row.get("memory_type", "")) == MemoryType.PREFERENCE.value
        ]
        if not preference_rows:
            return {}

        preference_rows.sort(key=lambda item: int(item.get("updated_at", 0) or 0))
        merged: dict[str, Any] = {}
        for row in preference_rows:
            parsed = self._coerce_dict(row.get("content"))
            if not parsed:
                continue
            normalized = self._normalize_preference_payload(parsed)
            if not normalized:
                continue
            merged = self._merge_preference_payload(merged, normalized)
        return merged

    @staticmethod
    def _coerce_dict(content: Any) -> dict[str, Any]:
        """将记忆内容安全转为 dict。"""

        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except Exception:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @staticmethod
    def _extract_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
                        continue
                chunks.append(str(item))
            return "".join(chunks).strip()
        return str(content).strip()

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if not cleaned:
            return {}

        parsed = MemoryService._try_parse_json(cleaned)
        if isinstance(parsed, dict):
            return parsed

        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
        if fence_match:
            parsed = MemoryService._try_parse_json(fence_match.group(1))
            if isinstance(parsed, dict):
                return parsed

        object_match = re.search(r"(\{.*\})", cleaned, flags=re.DOTALL)
        if object_match:
            parsed = MemoryService._try_parse_json(object_match.group(1))
            if isinstance(parsed, dict):
                return parsed

        return {}

    @staticmethod
    def _try_parse_json(text: str) -> Any:
        try:
            return json.loads(text)
        except Exception:
            return None

    def _mem0_add(self, user_id: str, content: str, metadata: dict[str, Any]) -> None:
        """安全调用 Mem0 add，失败仅记录日志。"""

        if self._mem0_client is None:
            return
        try:
            add_fn = getattr(self._mem0_client, "add", None)
            if callable(add_fn):
                add_fn(content, user_id=user_id, metadata=metadata)
        except Exception:
            logger.exception("Mem0 add 调用失败，已忽略")

    def _mem0_search(self, user_id: str, query: str, limit: int) -> list[dict[str, Any]]:
        """安全调用 Mem0 search。"""

        if self._mem0_client is None:
            return []
        try:
            search_fn = getattr(self._mem0_client, "search", None)
            if not callable(search_fn):
                return []
            result = search_fn(**self._build_mem0_search_kwargs(user_id=user_id, query=query, limit=limit))
            return self._normalize_mem0_search_result(result)
        except Exception:
            logger.exception("Mem0 search 调用失败，已忽略")
        return []

    def _build_mem0_search_kwargs(self, user_id: str, query: str, limit: int) -> dict[str, Any]:
        """构造兼容 Mem0 Platform / OSS 的 search 参数。

        - Platform (`MemoryClient`) 要求通过 `filters` 指定实体范围。
        - OSS (`Memory`) 兼容 `user_id + limit` 形式。
        """

        if self._mem0_mode() == "platform":
            return {
                "query": query,
                "top_k": limit,
                "filters": {"user_id": user_id},
            }
        return {
            "query": query,
            "user_id": user_id,
            "limit": limit,
        }

    def _normalize_mem0_search_result(self, result: Any) -> list[dict[str, Any]]:
        """兼容 Mem0 不同客户端的 search 返回结构。"""

        if isinstance(result, list):
            return [item for item in result if isinstance(item, dict)]
        if isinstance(result, dict):
            for key in ("results", "data"):
                items = result.get(key)
                if isinstance(items, list):
                    return [item for item in items if isinstance(item, dict)]
        return []
