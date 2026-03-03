"""Outbox 异步投影器。

职责：
- 后台轮询 outbox_event
- 将 memory 写入事件投影到 Milvus/Mem0
- 失败按 attempts 重试，超过阈值标记 failed
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

from app.config.logging import get_logger, log_context
from app.conversation.store import SQLiteConversationTruthStore
from app.memory.mem0_service import MemoryService

logger = get_logger(__name__)


class OutboxProjector:
    """后台 outbox 消费器。"""

    def __init__(
        self,
        *,
        truth_store: SQLiteConversationTruthStore,
        memory_service: MemoryService,
        poll_interval_seconds: float = 0.5,
        batch_size: int = 50,
        max_attempts: int = 8,
    ) -> None:
        self.truth_store = truth_store
        self.memory_service = memory_service
        self.poll_interval_seconds = max(poll_interval_seconds, 0.1)
        self.batch_size = max(batch_size, 1)
        self.max_attempts = max(max_attempts, 1)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="outbox-projector", daemon=True)
        self._thread.start()
        logger.info("OutboxProjector 已启动")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        logger.info("OutboxProjector 已停止")

    def run_once(self) -> int:
        """同步处理一批 outbox 事件，返回处理条数。"""

        rows = self.truth_store.fetch_pending_outbox(limit=self.batch_size)
        for row in rows:
            outbox_id = int(row["id"])
            try:
                self._apply_row(row)
                self.truth_store.mark_outbox_done(outbox_id)
            except Exception as exc:
                self.truth_store.mark_outbox_retry(
                    outbox_id,
                    error_text=type(exc).__name__,
                    max_attempts=self.max_attempts,
                )
                logger.exception("outbox 处理失败 id=%s event=%s", outbox_id, row.get("event_type"))
        return len(rows)

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            handled = self.run_once()
            if handled == 0:
                time.sleep(self.poll_interval_seconds)

    def _apply_row(self, row: dict[str, Any]) -> None:
        event_type = str(row.get("event_type", "")).strip()
        payload_json = str(row.get("payload_json", "")).strip()
        payload = json.loads(payload_json) if payload_json else {}
        if not isinstance(payload, dict):
            payload = {}

        with log_context(component="outbox.projector", task_id=str(payload.get("request_id", "-"))):
            if event_type == "memory.save_preference":
                self.memory_service.apply_preference_write(
                    user_id=str(payload["user_id"]),
                    preference=dict(payload["preference"]),
                    confidence=float(payload.get("confidence", 0.8)),
                    updated_at=int(payload.get("updated_at", 0) or 0),
                )
                return

            if event_type == "memory.save_tool_correction":
                self.memory_service.apply_tool_correction_write(
                    user_id=str(payload["user_id"]),
                    correction=dict(payload["correction"]),
                    confidence=float(payload.get("confidence", 0.8)),
                    updated_at=int(payload.get("updated_at", 0) or 0),
                )
                return

            # 非 memory 事件默认视为已消费（例如未来扩展）。
            logger.info("outbox 事件已忽略 event_type=%s", event_type)

