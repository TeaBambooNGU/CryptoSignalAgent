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
from datetime import datetime, timezone
from typing import Any

from app.config.settings import Settings
from app.models.schemas import MemoryType
from app.retrieval.embedding import text_to_embedding
from app.retrieval.milvus_store import MilvusStore

logger = logging.getLogger(__name__)


class MemoryService:
    """统一记忆服务。"""

    def __init__(self, settings: Settings, milvus_store: MilvusStore) -> None:
        self.settings = settings
        self.milvus_store = milvus_store
        self._session_memory: dict[str, list[dict[str, Any]]] = {}
        self._mem0_client: Any | None = self._init_mem0_client()

    def _init_mem0_client(self) -> Any | None:
        """按配置尝试初始化 Mem0 客户端。"""

        if not self.settings.mem0_enabled:
            return None
        if not self.settings.mem0_api_key:
            logger.warning("MEM0_ENABLED=true 但未配置 MEM0_API_KEY，将使用降级模式")
            return None

        try:
            import mem0  # type: ignore

            memory_cls = getattr(mem0, "Memory", None)
            if memory_cls is None:
                logger.warning("mem0 包未提供 Memory 类，自动降级")
                return None

            client = memory_cls(
                api_key=self.settings.mem0_api_key,
                org_id=self.settings.mem0_org_id or None,
                project_id=self.settings.mem0_project_id or None,
            )
            logger.info("Mem0 客户端初始化成功")
            return client
        except Exception:
            logger.exception("Mem0 客户端初始化失败，自动降级")
            return None

    def _build_memory_id(self, user_id: str, memory_type: MemoryType, content: str) -> str:
        """生成幂等记忆 ID。"""

        digest = hashlib.sha1(f"{user_id}:{memory_type.value}:{content}".encode("utf-8")).hexdigest()
        return digest[:40]

    def _append_session_memory(self, user_id: str, item: dict[str, Any]) -> None:
        """写入短期记忆，保留最近 50 条。"""

        bucket = self._session_memory.setdefault(user_id, [])
        bucket.append(item)
        if len(bucket) > 50:
            del bucket[:-50]

    def save_preference(self, user_id: str, preference: dict[str, Any], confidence: float = 0.8) -> None:
        """保存长期偏好。

        长期记忆判定规则：
        - 用户显式提交的偏好视为长期可复用信息。
        - 采用“时间戳 + 置信度”覆盖策略。
        """

        content = json.dumps(preference, ensure_ascii=False, sort_keys=True)
        embedding = text_to_embedding(content, self.settings)
        record = {
            "id": self._build_memory_id(user_id, MemoryType.PREFERENCE, content),
            "user_id": user_id,
            "memory_type": MemoryType.PREFERENCE.value,
            "content": content,
            "confidence": confidence,
            "updated_at": int(datetime.now(timezone.utc).timestamp()),
            "embedding": embedding,
        }
        self.milvus_store.upsert_user_memory([record])
        self._append_session_memory(
            user_id,
            {
                "memory_type": MemoryType.PREFERENCE.value,
                "content": preference,
                "confidence": confidence,
                "updated_at": record["updated_at"],
            },
        )

        # Mem0 写回为增强能力，失败不阻断主流程。
        self._mem0_add(user_id=user_id, content=content, metadata={"memory_type": MemoryType.PREFERENCE.value})

    def save_task_context(self, user_id: str, task_context: dict[str, Any] | None) -> None:
        """保存短期任务上下文。"""

        if not task_context:
            return
        self._append_session_memory(
            user_id,
            {
                "memory_type": MemoryType.CONTEXT.value,
                "content": task_context,
                "confidence": 0.6,
                "updated_at": int(datetime.now(timezone.utc).timestamp()),
            },
        )

    def load_memory_profile(self, user_id: str) -> dict[str, Any]:
        """聚合用户长期记忆与短期记忆。"""

        long_term = self.milvus_store.query_user_memory(user_id=user_id, limit=100)
        session = self._session_memory.get(user_id, [])

        profile = {
            "user_id": user_id,
            "long_term_memory": [self._decode_memory_row(row) for row in long_term],
            "session_memory": session,
            "watchlist": self._extract_watchlist(long_term, session),
            "risk_preference": self._extract_risk_preference(long_term),
        }

        # Mem0 可用时补充最近记忆片段，提高上下文完整度。
        mem0_recent = self._mem0_search(user_id=user_id, query="recent user preference", limit=5)
        if mem0_recent:
            profile["mem0_recent"] = mem0_recent
        return profile

    def persist_report_memory(
        self,
        user_id: str,
        query: str,
        report: str,
        inferred_preferences: dict[str, Any] | None = None,
    ) -> None:
        """从研报结果提取可复用偏好并写回长期记忆。

        规则说明：
        - 仅写回“跨任务可复用”的偏好，例如关注标的、风险偏好、格式偏好。
        - 一次性任务上下文不写入长期记忆，转入短期会话缓存。
        """

        extracted = inferred_preferences or self._infer_preferences_from_text(query=query, report=report)
        if not extracted:
            return

        self.save_preference(user_id=user_id, preference=extracted, confidence=0.7)

    def get_user_profile(self, user_id: str) -> dict[str, Any]:
        """返回 API 需要的用户画像结构。"""

        profile = self.load_memory_profile(user_id=user_id)
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

        candidates: set[str] = set()
        sources = [*long_term, *session]
        for row in sources:
            content = row.get("content")
            parsed = content
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                except Exception:
                    parsed = {}
            if isinstance(parsed, dict):
                watchlist = parsed.get("watchlist")
                if isinstance(watchlist, list):
                    candidates.update(str(item).upper() for item in watchlist)
        return sorted(candidates)

    def _extract_risk_preference(self, long_term: list[dict[str, Any]]) -> str | None:
        """提取用户风险偏好。"""

        for row in long_term:
            content = row.get("content")
            parsed = content
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                except Exception:
                    parsed = {}
            if isinstance(parsed, dict) and parsed.get("risk_preference"):
                return str(parsed["risk_preference"])
        return None

    def _infer_preferences_from_text(self, query: str, report: str) -> dict[str, Any]:
        """基于查询和报告粗粒度抽取可复用偏好。"""

        merged_text = f"{query}\n{report}".upper()
        watchlist = []
        for symbol in ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA"]:
            if symbol in merged_text:
                watchlist.append(symbol)

        result: dict[str, Any] = {}
        if watchlist:
            result["watchlist"] = sorted(set(watchlist))

        if "风险" in report and "保守" in report:
            result["risk_preference"] = "conservative"
        elif "风险" in report and ("激进" in report or "高波动" in report):
            result["risk_preference"] = "aggressive"

        if "总结" in report or "摘要" in report:
            result["reading_habit"] = "summary_first"

        return result

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
            result = search_fn(query=query, user_id=user_id, limit=limit)
            if isinstance(result, list):
                return [item for item in result if isinstance(item, dict)]
        except Exception:
            logger.exception("Mem0 search 调用失败，已忽略")
        return []
