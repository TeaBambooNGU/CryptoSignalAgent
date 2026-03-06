"""会话真相库与幂等存储。

当前实现使用 SQLite 落地以下数据：
- conversation_snapshot: 会话最新版本快照
- conversation_turn: 每轮完整记录（恢复主表）
- conversation_state_checkpoint: 节点级状态快照（预留）
- idempotency_request: request_id 幂等记录
- conversation_event: 事件日志
- outbox_event: 异步投影外箱
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.conversation.errors import ConversationConflictError, DuplicateRequestInFlightError
from app.models.schemas import Citation, WorkflowStep


def _now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _jsonable(payload: Any) -> Any:
    if isinstance(payload, Citation | WorkflowStep):
        return payload.model_dump(mode="json")
    if isinstance(payload, list):
        return [_jsonable(item) for item in payload]
    if isinstance(payload, dict):
        return {str(key): _jsonable(value) for key, value in payload.items()}
    return payload


def _json_dumps(payload: Any) -> str:
    return json.dumps(_jsonable(payload), ensure_ascii=False)


def _json_loads(payload: str | None, fallback: Any) -> Any:
    raw = (payload or "").strip()
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except Exception:
        return fallback


def _decode_citations(payload: str | None) -> list[Citation]:
    rows = _json_loads(payload, [])
    if not isinstance(rows, list):
        return []
    return [Citation.model_validate(item) for item in rows]


def _decode_workflow_steps(payload: str | None) -> list[WorkflowStep]:
    rows = _json_loads(payload, [])
    if not isinstance(rows, list):
        return []
    return [WorkflowStep.model_validate(item) for item in rows]


@dataclass(slots=True)
class TurnPreparation:
    """会话轮次预处理结果。"""

    conversation_id: str
    turn_id: str
    request_id: str
    conversation_version: int
    cached_response: dict[str, Any] | None = None


class ConversationLockManager:
    """同 conversation 串行执行锁。"""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._guard = asyncio.Lock()

    async def _get_lock(self, conversation_id: str) -> asyncio.Lock:
        async with self._guard:
            lock = self._locks.get(conversation_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[conversation_id] = lock
            return lock

    @asynccontextmanager
    async def acquire(self, conversation_id: str):
        lock = await self._get_lock(conversation_id)
        await lock.acquire()
        try:
            yield
        finally:
            lock.release()


class SQLiteConversationTruthStore:
    """SQLite 实现的会话真相库。"""

    def __init__(self, db_path: str) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_guard = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversation_snapshot (
                    conversation_id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    last_turn_id TEXT NOT NULL,
                    updated_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS idempotency_request (
                    request_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    response_json TEXT,
                    error_text TEXT,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversation_turn (
                    conversation_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    request_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    assistant_message_text TEXT NOT NULL DEFAULT '',
                    task_context_json TEXT NOT NULL,
                    response_report TEXT,
                    response_citations_json TEXT NOT NULL,
                    response_errors_json TEXT NOT NULL,
                    workflow_steps_json TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    intent TEXT NOT NULL DEFAULT 'regenerate_report',
                    turn_type TEXT NOT NULL DEFAULT 'assistant_report',
                    parent_turn_id TEXT,
                    report_id TEXT,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY (conversation_id, turn_id),
                    UNIQUE (conversation_id, version),
                    UNIQUE (request_id)
                );

                CREATE INDEX IF NOT EXISTS idx_turn_conversation_version
                    ON conversation_turn(conversation_id, version DESC);

                CREATE TABLE IF NOT EXISTS conversation_state_checkpoint (
                    conversation_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    PRIMARY KEY (conversation_id, turn_id, node_id)
                );

                CREATE TABLE IF NOT EXISTS conversation_report (
                    report_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    report_version INTEGER NOT NULL,
                    created_by_turn_id TEXT NOT NULL,
                    based_on_report_id TEXT,
                    mode TEXT NOT NULL,
                    report_text TEXT NOT NULL,
                    citations_json TEXT NOT NULL,
                    workflow_steps_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    UNIQUE (conversation_id, report_version)
                );

                CREATE INDEX IF NOT EXISTS idx_report_conversation_version
                    ON conversation_report(conversation_id, report_version DESC);

                CREATE TABLE IF NOT EXISTS conversation_context_summary (
                    conversation_id TEXT PRIMARY KEY,
                    summary_text TEXT NOT NULL,
                    through_version INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS knowledge_document (
                    doc_id TEXT PRIMARY KEY,
                    kb_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    source TEXT NOT NULL,
                    doc_type TEXT NOT NULL,
                    symbols_json TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    status TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    uploaded_by TEXT NOT NULL,
                    published_at INTEGER,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_knowledge_document_updated_at
                    ON knowledge_document(updated_at DESC);

                CREATE TABLE IF NOT EXISTS conversation_event (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS outbox_event (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL,
                    last_error TEXT,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_outbox_status ON outbox_event(status, created_at);
                """
            )
            existing = {str(row["name"]) for row in conn.execute("PRAGMA table_info(outbox_event)")}
            if "last_error" not in existing:
                conn.execute("ALTER TABLE outbox_event ADD COLUMN last_error TEXT")
            turn_columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(conversation_turn)")}
            migrations = [
                ("assistant_message_text", "ALTER TABLE conversation_turn ADD COLUMN assistant_message_text TEXT NOT NULL DEFAULT ''"),
                ("intent", "ALTER TABLE conversation_turn ADD COLUMN intent TEXT NOT NULL DEFAULT 'regenerate_report'"),
                ("turn_type", "ALTER TABLE conversation_turn ADD COLUMN turn_type TEXT NOT NULL DEFAULT 'assistant_report'"),
                ("parent_turn_id", "ALTER TABLE conversation_turn ADD COLUMN parent_turn_id TEXT"),
                ("report_id", "ALTER TABLE conversation_turn ADD COLUMN report_id TEXT"),
            ]
            for column_name, ddl in migrations:
                if column_name not in turn_columns:
                    conn.execute(ddl)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_turn_report_id ON conversation_turn(conversation_id, report_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_turn_parent_turn ON conversation_turn(conversation_id, parent_turn_id)"
            )

    def upsert_knowledge_document(self, payload: dict[str, Any]) -> dict[str, Any]:
        """写入或更新知识库文档元数据。"""

        now = _now_ts()
        row = {
            "doc_id": str(payload.get("doc_id") or uuid4()),
            "kb_id": str(payload.get("kb_id") or "default"),
            "title": str(payload.get("title") or ""),
            "source": str(payload.get("source") or "manual"),
            "doc_type": str(payload.get("doc_type") or "research_report"),
            "symbols_json": _json_dumps(payload.get("symbols") or []),
            "tags_json": _json_dumps(payload.get("tags") or []),
            "file_name": str(payload.get("file_name") or ""),
            "content_type": str(payload.get("content_type") or "text/plain"),
            "checksum": str(payload.get("checksum") or ""),
            "status": str(payload.get("status") or "ready"),
            "chunk_count": int(payload.get("chunk_count", 0) or 0),
            "uploaded_by": str(payload.get("uploaded_by") or "unknown"),
            "published_at": int(payload.get("published_at", 0) or 0) or None,
            "created_at": int(payload.get("created_at", now) or now),
            "updated_at": int(payload.get("updated_at", now) or now),
        }
        with self._write_guard:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO knowledge_document(
                      doc_id, kb_id, title, source, doc_type, symbols_json, tags_json,
                      file_name, content_type, checksum, status, chunk_count,
                      uploaded_by, published_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(doc_id) DO UPDATE SET
                      kb_id=excluded.kb_id,
                      title=excluded.title,
                      source=excluded.source,
                      doc_type=excluded.doc_type,
                      symbols_json=excluded.symbols_json,
                      tags_json=excluded.tags_json,
                      file_name=excluded.file_name,
                      content_type=excluded.content_type,
                      checksum=excluded.checksum,
                      status=excluded.status,
                      chunk_count=excluded.chunk_count,
                      uploaded_by=excluded.uploaded_by,
                      published_at=excluded.published_at,
                      updated_at=excluded.updated_at
                    """,
                    (
                        row["doc_id"],
                        row["kb_id"],
                        row["title"],
                        row["source"],
                        row["doc_type"],
                        row["symbols_json"],
                        row["tags_json"],
                        row["file_name"],
                        row["content_type"],
                        row["checksum"],
                        row["status"],
                        row["chunk_count"],
                        row["uploaded_by"],
                        row["published_at"],
                        row["created_at"],
                        row["updated_at"],
                    ),
                )
                stored = conn.execute(
                    "SELECT * FROM knowledge_document WHERE doc_id=?",
                    (row["doc_id"],),
                ).fetchone()
        return self._decode_knowledge_document_row(stored) if stored is not None else {}

    def list_knowledge_documents(self, *, limit: int = 100, kb_id: str | None = None) -> list[dict[str, Any]]:
        """获取知识库文档列表。"""

        final_limit = max(1, min(int(limit), 500))
        sql = "SELECT * FROM knowledge_document WHERE status != 'deleted'"
        params: list[Any] = []
        if kb_id:
            sql += " AND kb_id=?"
            params.append(str(kb_id))
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(final_limit)
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._decode_knowledge_document_row(row) for row in rows]

    def get_knowledge_document(self, *, doc_id: str) -> dict[str, Any] | None:
        """获取单个知识库文档元数据。"""

        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM knowledge_document WHERE doc_id=?",
                (str(doc_id),),
            ).fetchone()
        if row is None:
            return None
        return self._decode_knowledge_document_row(row)

    def mark_knowledge_document_deleted(self, *, doc_id: str) -> dict[str, Any] | None:
        """软删除知识库文档。"""

        now = _now_ts()
        with self._write_guard:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE knowledge_document SET status='deleted', updated_at=? WHERE doc_id=?",
                    (now, str(doc_id)),
                )
                row = conn.execute(
                    "SELECT * FROM knowledge_document WHERE doc_id=?",
                    (str(doc_id),),
                ).fetchone()
        if row is None:
            return None
        return self._decode_knowledge_document_row(row)

    def prepare_turn(
        self,
        *,
        conversation_id: str,
        turn_id: str | None,
        request_id: str,
        expected_version: int | None,
    ) -> TurnPreparation:
        """预处理会话轮次：幂等检查 + CAS + 快照推进。"""

        now = _now_ts()
        with self._write_guard:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")

                idem = conn.execute(
                    "SELECT request_id, status, response_json FROM idempotency_request WHERE request_id = ?",
                    (request_id,),
                ).fetchone()
                if idem is not None:
                    status = str(idem["status"])
                    if status == "completed":
                        payload = str(idem["response_json"] or "").strip()
                        cached = json.loads(payload) if payload else None
                        conn.execute("COMMIT")
                        return TurnPreparation(
                            conversation_id=conversation_id,
                            turn_id=turn_id or "",
                            request_id=request_id,
                            conversation_version=-1,
                            cached_response=cached if isinstance(cached, dict) else None,
                        )
                    conn.execute("ROLLBACK")
                    raise DuplicateRequestInFlightError(f"request_id={request_id} is still in-flight")

                row = conn.execute(
                    "SELECT version FROM conversation_snapshot WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
                current_version = int(row["version"]) if row is not None else 0
                if expected_version is not None and expected_version != current_version:
                    conn.execute("ROLLBACK")
                    raise ConversationConflictError(
                        expected_version=expected_version,
                        current_version=current_version,
                    )

                next_version = current_version + 1
                resolved_turn_id = turn_id or f"turn-{next_version}"

                conn.execute(
                    """
                    INSERT INTO conversation_snapshot(conversation_id, version, last_turn_id, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(conversation_id) DO UPDATE SET
                      version=excluded.version,
                      last_turn_id=excluded.last_turn_id,
                      updated_at=excluded.updated_at
                    """,
                    (conversation_id, next_version, resolved_turn_id, now),
                )
                conn.execute(
                    """
                    INSERT INTO idempotency_request(
                      request_id, conversation_id, turn_id, status, response_json, error_text, created_at, updated_at
                    )
                    VALUES (?, ?, ?, 'pending', NULL, NULL, ?, ?)
                    """,
                    (request_id, conversation_id, resolved_turn_id, now, now),
                )

                self._insert_event(
                    conn=conn,
                    conversation_id=conversation_id,
                    turn_id=resolved_turn_id,
                    request_id=request_id,
                    event_type="turn.accepted",
                    payload={
                        "expected_version": expected_version,
                        "applied_version": next_version,
                    },
                    to_outbox=False,
                )
                conn.execute("COMMIT")
                return TurnPreparation(
                    conversation_id=conversation_id,
                    turn_id=resolved_turn_id,
                    request_id=request_id,
                    conversation_version=next_version,
                )

    def save_turn_result(
        self,
        *,
        request_id: str,
        user_id: str,
        query_text: str,
        task_context: dict[str, Any] | None,
        response: dict[str, Any],
        assistant_message: str | None = None,
        intent: str = "regenerate_report",
        turn_type: str = "assistant_report",
        parent_turn_id: str | None = None,
        report_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """原子写入 turn 结果并完成幂等状态。"""

        now = _now_ts()
        with self._write_guard:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    """
                    SELECT conversation_id, turn_id, status
                    FROM idempotency_request
                    WHERE request_id = ?
                    """,
                    (request_id,),
                ).fetchone()
                if row is None:
                    conn.execute("ROLLBACK")
                    return

                conversation_id = str(row["conversation_id"])
                turn_id = str(row["turn_id"])
                status = str(row["status"])
                if status == "completed":
                    conn.execute("COMMIT")
                    existing_report_id = conn.execute(
                        "SELECT report_id FROM conversation_turn WHERE request_id = ?",
                        (request_id,),
                    ).fetchone()
                    if existing_report_id is None:
                        return None
                    report_id = str(existing_report_id["report_id"] or "")
                    if not report_id:
                        return None
                    report_row = conn.execute(
                        "SELECT * FROM conversation_report WHERE report_id = ?",
                        (report_id,),
                    ).fetchone()
                    return dict(report_row) if report_row is not None else None

                conversation_version = int(response.get("conversation_version", 0) or 0)
                if conversation_version <= 0:
                    snapshot = conn.execute(
                        "SELECT version FROM conversation_snapshot WHERE conversation_id = ?",
                        (conversation_id,),
                    ).fetchone()
                    conversation_version = int(snapshot["version"]) if snapshot is not None else 1

                persisted_report: dict[str, Any] | None = None
                report_id: str | None = None
                if report_payload is not None:
                    next_version_row = conn.execute(
                        """
                        SELECT COALESCE(MAX(report_version), 0) + 1 AS next_version
                        FROM conversation_report
                        WHERE conversation_id = ?
                        """,
                        (conversation_id,),
                    ).fetchone()
                    next_report_version = int(next_version_row["next_version"]) if next_version_row else 1
                    report_id = str(report_payload.get("report_id") or f"rpt-{uuid4().hex[:12]}")
                    report_mode = str(report_payload.get("mode", "regenerate"))
                    based_on_report_id = report_payload.get("based_on_report_id")
                    citations = report_payload.get("citations", [])
                    workflow_steps = report_payload.get("workflow_steps", [])
                    report_text = str(report_payload.get("report", ""))

                    conn.execute(
                        """
                        INSERT INTO conversation_report(
                          report_id, conversation_id, report_version, created_by_turn_id, based_on_report_id, mode,
                          report_text, citations_json, workflow_steps_json, status, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'completed', ?, ?)
                        ON CONFLICT(report_id) DO UPDATE SET
                          report_text=excluded.report_text,
                          citations_json=excluded.citations_json,
                          workflow_steps_json=excluded.workflow_steps_json,
                          status='completed',
                          updated_at=excluded.updated_at
                        """,
                        (
                            report_id,
                            conversation_id,
                            next_report_version,
                            turn_id,
                            str(based_on_report_id) if based_on_report_id else None,
                            report_mode,
                            report_text,
                            _json_dumps(citations if isinstance(citations, list) else []),
                            _json_dumps(workflow_steps if isinstance(workflow_steps, list) else []),
                            now,
                            now,
                        ),
                    )
                    persisted_report = {
                        "report_id": report_id,
                        "conversation_id": conversation_id,
                        "report_version": next_report_version,
                        "created_by_turn_id": turn_id,
                        "based_on_report_id": str(based_on_report_id) if based_on_report_id else None,
                        "mode": report_mode,
                        "report": report_text,
                        "citations": citations if isinstance(citations, list) else [],
                        "workflow_steps": workflow_steps if isinstance(workflow_steps, list) else [],
                        "status": "completed",
                        "created_at": now,
                        "updated_at": now,
                    }

                assistant_text = (
                    str(assistant_message)
                    if assistant_message is not None
                    else str(response.get("assistant_message") or response.get("report") or "")
                )
                report_value = response.get("report")
                response_report_text = str(report_value) if isinstance(report_value, str) else assistant_text
                conn.execute(
                    """
                    INSERT INTO conversation_turn(
                      conversation_id, turn_id, version, request_id, user_id, query_text,
                      assistant_message_text, task_context_json, response_report,
                      response_citations_json, response_errors_json, workflow_steps_json,
                      trace_id, status, intent, turn_type, parent_turn_id, report_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'completed', ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(request_id) DO UPDATE SET
                      assistant_message_text=excluded.assistant_message_text,
                      response_report=excluded.response_report,
                      response_citations_json=excluded.response_citations_json,
                      response_errors_json=excluded.response_errors_json,
                      workflow_steps_json=excluded.workflow_steps_json,
                      trace_id=excluded.trace_id,
                      intent=excluded.intent,
                      turn_type=excluded.turn_type,
                      parent_turn_id=excluded.parent_turn_id,
                      report_id=excluded.report_id,
                      status='completed',
                      updated_at=excluded.updated_at
                    """,
                    (
                        conversation_id,
                        turn_id,
                        conversation_version,
                        request_id,
                        user_id,
                        query_text,
                        assistant_text,
                        _json_dumps(task_context or {}),
                        response_report_text,
                        _json_dumps(response.get("citations", [])),
                        _json_dumps(response.get("errors", [])),
                        _json_dumps(response.get("workflow_steps", [])),
                        str(response.get("trace_id", "")),
                        intent,
                        turn_type,
                        parent_turn_id,
                        report_id,
                        now,
                        now,
                    ),
                )
                conn.execute(
                    """
                    UPDATE idempotency_request
                    SET status='completed', response_json=?, error_text=NULL, updated_at=?
                    WHERE request_id=?
                    """,
                    (_json_dumps(response), now, request_id),
                )
                self._insert_event(
                    conn=conn,
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    request_id=request_id,
                    event_type="turn.completed",
                    payload={
                        "conversation_version": conversation_version,
                        "trace_id": response.get("trace_id"),
                        "intent": intent,
                        "turn_type": turn_type,
                        "report_id": report_id,
                    },
                    to_outbox=False,
                )
                conn.execute("COMMIT")
                return persisted_report

    def update_idempotency_response(self, *, request_id: str, response: dict[str, Any]) -> None:
        """更新幂等响应体，用于补全后置字段（如 report 版本信息）。"""

        now = _now_ts()
        with self._write_guard:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE idempotency_request
                    SET response_json=?, updated_at=?
                    WHERE request_id=? AND status='completed'
                    """,
                    (_json_dumps(response), now, request_id),
                )

    def save_turn_failure(
        self,
        *,
        request_id: str,
        user_id: str,
        query_text: str,
        task_context: dict[str, Any] | None,
        trace_id: str,
        error_text: str,
        intent: str = "regenerate_report",
        turn_type: str = "assistant_report",
        parent_turn_id: str | None = None,
    ) -> None:
        """原子写入失败 turn 并更新幂等状态。"""

        now = _now_ts()
        trimmed_error = error_text[:1000]
        with self._write_guard:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    """
                    SELECT conversation_id, turn_id, status
                    FROM idempotency_request
                    WHERE request_id = ?
                    """,
                    (request_id,),
                ).fetchone()
                if row is None:
                    conn.execute("ROLLBACK")
                    return

                conversation_id = str(row["conversation_id"])
                turn_id = str(row["turn_id"])
                status = str(row["status"])
                if status == "completed":
                    conn.execute("COMMIT")
                    return

                snapshot = conn.execute(
                    "SELECT version FROM conversation_snapshot WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
                conversation_version = int(snapshot["version"]) if snapshot is not None else 1
                conn.execute(
                    """
                    INSERT INTO conversation_turn(
                      conversation_id, turn_id, version, request_id, user_id, query_text,
                      assistant_message_text, task_context_json, response_report,
                      response_citations_json, response_errors_json, workflow_steps_json,
                      trace_id, status, intent, turn_type, parent_turn_id, report_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, '', '[]', ?, '[]', ?, 'failed', ?, ?, ?, NULL, ?, ?)
                    ON CONFLICT(request_id) DO UPDATE SET
                      assistant_message_text=excluded.assistant_message_text,
                      response_errors_json=excluded.response_errors_json,
                      trace_id=excluded.trace_id,
                      intent=excluded.intent,
                      turn_type=excluded.turn_type,
                      parent_turn_id=excluded.parent_turn_id,
                      status='failed',
                      updated_at=excluded.updated_at
                    """,
                    (
                        conversation_id,
                        turn_id,
                        conversation_version,
                        request_id,
                        user_id,
                        query_text,
                        trimmed_error,
                        _json_dumps(task_context or {}),
                        _json_dumps([trimmed_error]),
                        trace_id,
                        intent,
                        turn_type,
                        parent_turn_id,
                        now,
                        now,
                    ),
                )
                conn.execute(
                    """
                    UPDATE idempotency_request
                    SET status='failed', error_text=?, updated_at=?
                    WHERE request_id=?
                    """,
                    (trimmed_error, now, request_id),
                )
                self._insert_event(
                    conn=conn,
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    request_id=request_id,
                    event_type="turn.failed",
                    payload={"error": trimmed_error, "intent": intent, "turn_type": turn_type},
                    to_outbox=False,
                )
                conn.execute("COMMIT")

    def complete_turn(self, *, request_id: str, response: dict[str, Any]) -> None:
        """兼容旧调用：仅更新幂等完成状态。"""

        self.save_turn_result(
            request_id=request_id,
            user_id=str(response.get("user_id", "")),
            query_text=str(response.get("query", "")),
            task_context=response.get("task_context") if isinstance(response.get("task_context"), dict) else None,
            response=response,
        )

    def fail_turn(self, *, request_id: str, error_text: str) -> None:
        """兼容旧调用：仅更新幂等失败状态。"""

        self.save_turn_failure(
            request_id=request_id,
            user_id="",
            query_text="",
            task_context=None,
            trace_id="",
            error_text=error_text,
        )

    def get_conversation_meta(self, conversation_id: str) -> dict[str, Any] | None:
        """返回会话元信息。"""

        with self._connect() as conn:
            snapshot = conn.execute(
                """
                SELECT conversation_id, version, last_turn_id, updated_at
                FROM conversation_snapshot
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
            if snapshot is None:
                return None
            turns_count_row = conn.execute(
                "SELECT COUNT(1) AS turn_count FROM conversation_turn WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        turn_count = int(turns_count_row["turn_count"]) if turns_count_row is not None else 0
        return {
            "conversation_id": str(snapshot["conversation_id"]),
            "latest_version": int(snapshot["version"]),
            "latest_turn_id": str(snapshot["last_turn_id"]),
            "turn_count": turn_count,
            "updated_at": int(snapshot["updated_at"]),
        }

    def list_turns(
        self,
        *,
        conversation_id: str,
        limit: int = 20,
        before_version: int | None = None,
    ) -> list[dict[str, Any]]:
        """分页获取会话 turn 摘要。"""

        final_limit = min(max(int(limit), 1), 100)
        with self._connect() as conn:
            if before_version is None:
                rows = conn.execute(
                    """
                    SELECT conversation_id, turn_id, version, request_id, user_id, query_text,
                           assistant_message_text, response_report, trace_id, status,
                           intent, turn_type, parent_turn_id, report_id, created_at, updated_at
                    FROM conversation_turn
                    WHERE conversation_id = ?
                    ORDER BY version DESC
                    LIMIT ?
                    """,
                    (conversation_id, final_limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT conversation_id, turn_id, version, request_id, user_id, query_text,
                           assistant_message_text, response_report, trace_id, status,
                           intent, turn_type, parent_turn_id, report_id, created_at, updated_at
                    FROM conversation_turn
                    WHERE conversation_id = ? AND version < ?
                    ORDER BY version DESC
                    LIMIT ?
                    """,
                    (conversation_id, int(before_version), final_limit),
                ).fetchall()

        return [
            {
                "conversation_id": str(row["conversation_id"]),
                "turn_id": str(row["turn_id"]),
                "version": int(row["version"]),
                "request_id": str(row["request_id"]),
                "user_id": str(row["user_id"]),
                "query": str(row["query_text"]),
                "report": str(row["response_report"] or ""),
                "assistant_message": str(row["assistant_message_text"] or ""),
                "trace_id": str(row["trace_id"]),
                "status": str(row["status"]),
                "intent": str(row["intent"] or ""),
                "turn_type": str(row["turn_type"] or ""),
                "parent_turn_id": str(row["parent_turn_id"]) if row["parent_turn_id"] else None,
                "report_id": str(row["report_id"]) if row["report_id"] else None,
                "created_at": int(row["created_at"]),
                "updated_at": int(row["updated_at"]),
            }
            for row in rows
        ]

    def get_turn(self, *, conversation_id: str, turn_id: str) -> dict[str, Any] | None:
        """获取单轮完整详情。"""

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT conversation_id, turn_id, version, request_id, user_id, query_text,
                       assistant_message_text, task_context_json, response_report,
                       response_citations_json, response_errors_json, workflow_steps_json,
                       trace_id, status, intent, turn_type, parent_turn_id, report_id, created_at, updated_at
                FROM conversation_turn
                WHERE conversation_id = ? AND turn_id = ?
                LIMIT 1
                """,
                (conversation_id, turn_id),
            ).fetchone()
        if row is None:
            return None
        return self._decode_turn_row(row)

    def get_latest_turn(self, *, conversation_id: str) -> dict[str, Any] | None:
        """获取最新一轮详情。"""

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT conversation_id, turn_id, version, request_id, user_id, query_text,
                       assistant_message_text, task_context_json, response_report,
                       response_citations_json, response_errors_json, workflow_steps_json,
                       trace_id, status, intent, turn_type, parent_turn_id, report_id, created_at, updated_at
                FROM conversation_turn
                WHERE conversation_id = ?
                ORDER BY version DESC
                LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
        if row is None:
            return None
        return self._decode_turn_row(row)

    def list_turn_lineage(
        self,
        *,
        conversation_id: str,
        leaf_turn_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """获取从叶子 turn 回溯到祖先链路的明细（按叶子->祖先顺序）。"""

        final_limit = min(max(int(limit), 1), 500)
        with self._connect() as conn:
            rows = conn.execute(
                """
                WITH RECURSIVE lineage AS (
                    SELECT conversation_id, turn_id, version, request_id, user_id, query_text,
                           assistant_message_text, task_context_json, response_report,
                           response_citations_json, response_errors_json, workflow_steps_json,
                           trace_id, status, intent, turn_type, parent_turn_id, report_id,
                           created_at, updated_at, 0 AS depth
                    FROM conversation_turn
                    WHERE conversation_id = ? AND turn_id = ?

                    UNION ALL

                    SELECT parent.conversation_id, parent.turn_id, parent.version, parent.request_id, parent.user_id,
                           parent.query_text, parent.assistant_message_text, parent.task_context_json, parent.response_report,
                           parent.response_citations_json, parent.response_errors_json, parent.workflow_steps_json,
                           parent.trace_id, parent.status, parent.intent, parent.turn_type, parent.parent_turn_id,
                           parent.report_id, parent.created_at, parent.updated_at, lineage.depth + 1 AS depth
                    FROM conversation_turn AS parent
                    JOIN lineage
                      ON parent.conversation_id = lineage.conversation_id
                     AND parent.turn_id = lineage.parent_turn_id
                    WHERE lineage.parent_turn_id IS NOT NULL
                )
                SELECT conversation_id, turn_id, version, request_id, user_id, query_text,
                       assistant_message_text, task_context_json, response_report,
                       response_citations_json, response_errors_json, workflow_steps_json,
                       trace_id, status, intent, turn_type, parent_turn_id, report_id,
                       created_at, updated_at
                FROM lineage
                ORDER BY depth ASC
                LIMIT ?
                """,
                (conversation_id, leaf_turn_id, final_limit),
            ).fetchall()
        return [self._decode_turn_row(row) for row in rows]

    def list_reports(
        self,
        *,
        conversation_id: str,
        limit: int = 20,
        before_report_version: int | None = None,
    ) -> list[dict[str, Any]]:
        """分页获取报告版本列表。"""

        final_limit = min(max(int(limit), 1), 100)
        with self._connect() as conn:
            if before_report_version is None:
                rows = conn.execute(
                    """
                    SELECT report_id, conversation_id, report_version, created_by_turn_id, based_on_report_id,
                           mode, report_text, citations_json, workflow_steps_json, status, created_at, updated_at
                    FROM conversation_report
                    WHERE conversation_id = ?
                    ORDER BY report_version DESC
                    LIMIT ?
                    """,
                    (conversation_id, final_limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT report_id, conversation_id, report_version, created_by_turn_id, based_on_report_id,
                           mode, report_text, citations_json, workflow_steps_json, status, created_at, updated_at
                    FROM conversation_report
                    WHERE conversation_id = ? AND report_version < ?
                    ORDER BY report_version DESC
                    LIMIT ?
                    """,
                    (conversation_id, int(before_report_version), final_limit),
                ).fetchall()
        return [self._decode_report_row(row) for row in rows]

    def get_report(self, *, report_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT report_id, conversation_id, report_version, created_by_turn_id, based_on_report_id,
                       mode, report_text, citations_json, workflow_steps_json, status, created_at, updated_at
                FROM conversation_report
                WHERE report_id = ?
                LIMIT 1
                """,
                (report_id,),
            ).fetchone()
        if row is None:
            return None
        return self._decode_report_row(row)

    def get_latest_report(self, *, conversation_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT report_id, conversation_id, report_version, created_by_turn_id, based_on_report_id,
                       mode, report_text, citations_json, workflow_steps_json, status, created_at, updated_at
                FROM conversation_report
                WHERE conversation_id = ?
                ORDER BY report_version DESC
                LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
        if row is None:
            return None
        return self._decode_report_row(row)

    def get_latest_report_on_lineage(
        self,
        *,
        conversation_id: str,
        leaf_turn_id: str,
    ) -> dict[str, Any] | None:
        """获取分支链路可见的最新报告。"""

        with self._connect() as conn:
            row = conn.execute(
                """
                WITH RECURSIVE lineage AS (
                    SELECT conversation_id, turn_id, parent_turn_id
                    FROM conversation_turn
                    WHERE conversation_id = ? AND turn_id = ?

                    UNION ALL

                    SELECT parent.conversation_id, parent.turn_id, parent.parent_turn_id
                    FROM conversation_turn AS parent
                    JOIN lineage
                      ON parent.conversation_id = lineage.conversation_id
                     AND parent.turn_id = lineage.parent_turn_id
                    WHERE lineage.parent_turn_id IS NOT NULL
                )
                SELECT report.report_id, report.conversation_id, report.report_version,
                       report.created_by_turn_id, report.based_on_report_id, report.mode,
                       report.report_text, report.citations_json, report.workflow_steps_json,
                       report.status, report.created_at, report.updated_at
                FROM lineage
                JOIN conversation_turn AS turn_row
                  ON turn_row.conversation_id = lineage.conversation_id
                 AND turn_row.turn_id = lineage.turn_id
                JOIN conversation_report AS report
                  ON report.report_id = turn_row.report_id
                ORDER BY report.report_version DESC
                LIMIT 1
                """,
                (conversation_id, leaf_turn_id),
            ).fetchone()
        if row is None:
            return None
        return self._decode_report_row(row)

    def get_context_summary(self, *, conversation_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT conversation_id, summary_text, through_version, updated_at
                FROM conversation_context_summary
                WHERE conversation_id = ?
                LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "conversation_id": str(row["conversation_id"]),
            "summary_text": str(row["summary_text"]),
            "through_version": int(row["through_version"]),
            "updated_at": int(row["updated_at"]),
        }

    def upsert_context_summary(
        self,
        *,
        conversation_id: str,
        summary_text: str,
        through_version: int,
    ) -> None:
        now = _now_ts()
        with self._write_guard:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO conversation_context_summary(conversation_id, summary_text, through_version, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(conversation_id) DO UPDATE SET
                      summary_text=excluded.summary_text,
                      through_version=excluded.through_version,
                      updated_at=excluded.updated_at
                    """,
                    (conversation_id, summary_text, int(through_version), now),
                )

    def list_turns_up_to_version(
        self,
        *,
        conversation_id: str,
        through_version: int,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        final_limit = min(max(int(limit), 1), 500)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT conversation_id, turn_id, version, request_id, user_id, query_text,
                       assistant_message_text, task_context_json, response_report, response_citations_json,
                       response_errors_json, workflow_steps_json, trace_id, status, intent, turn_type,
                       parent_turn_id, report_id, created_at, updated_at
                FROM conversation_turn
                WHERE conversation_id = ? AND version <= ?
                ORDER BY version ASC
                LIMIT ?
                """,
                (conversation_id, int(through_version), final_limit),
            ).fetchall()
        return [self._decode_turn_row(row) for row in rows]

    def enqueue_outbox_event(
        self,
        *,
        conversation_id: str,
        turn_id: str,
        request_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        """写入业务事件并加入 outbox。"""

        with self._write_guard:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                self._insert_event(
                    conn=conn,
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    request_id=request_id,
                    event_type=event_type,
                    payload=payload,
                    to_outbox=True,
                )
                conn.execute("COMMIT")

    def fetch_pending_outbox(self, limit: int = 100) -> list[dict[str, Any]]:
        """获取待处理 outbox 事件。"""

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, conversation_id, turn_id, request_id, event_type, payload_json, attempts
                FROM outbox_event
                WHERE status = 'pending'
                ORDER BY id ASC
                LIMIT ?
                """,
                (max(limit, 1),),
            ).fetchall()
        return [dict(row) for row in rows]

    def mark_outbox_done(self, outbox_id: int) -> None:
        """标记 outbox 事件处理完成。"""

        now = _now_ts()
        with self._write_guard:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE outbox_event SET status='done', updated_at=?, last_error=NULL WHERE id=?",
                    (now, int(outbox_id)),
                )

    def mark_outbox_retry(self, outbox_id: int, *, error_text: str, max_attempts: int) -> None:
        """标记 outbox 事件失败并决定是否继续重试。"""

        now = _now_ts()
        with self._write_guard:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT attempts FROM outbox_event WHERE id=?",
                    (int(outbox_id),),
                ).fetchone()
                if row is None:
                    return
                attempts = int(row["attempts"]) + 1
                next_status = "pending" if attempts < max_attempts else "failed"
                conn.execute(
                    """
                    UPDATE outbox_event
                    SET status=?, attempts=?, last_error=?, updated_at=?
                    WHERE id=?
                    """,
                    (next_status, attempts, error_text[:1000], now, int(outbox_id)),
                )

    def _decode_turn_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "conversation_id": str(row["conversation_id"]),
            "turn_id": str(row["turn_id"]),
            "version": int(row["version"]),
            "request_id": str(row["request_id"]),
            "user_id": str(row["user_id"]),
            "query": str(row["query_text"]),
            "assistant_message": str(row["assistant_message_text"] or ""),
            "task_context": _json_loads(str(row["task_context_json"]), {}),
            "report": str(row["response_report"] or ""),
            "citations": _decode_citations(str(row["response_citations_json"])),
            "errors": _json_loads(str(row["response_errors_json"]), []),
            "workflow_steps": _decode_workflow_steps(str(row["workflow_steps_json"])),
            "trace_id": str(row["trace_id"]),
            "status": str(row["status"]),
            "intent": str(row["intent"] or ""),
            "turn_type": str(row["turn_type"] or ""),
            "parent_turn_id": str(row["parent_turn_id"]) if row["parent_turn_id"] else None,
            "report_id": str(row["report_id"]) if row["report_id"] else None,
            "created_at": int(row["created_at"]),
            "updated_at": int(row["updated_at"]),
        }

    def _decode_report_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "report_id": str(row["report_id"]),
            "conversation_id": str(row["conversation_id"]),
            "report_version": int(row["report_version"]),
            "created_by_turn_id": str(row["created_by_turn_id"]),
            "based_on_report_id": str(row["based_on_report_id"]) if row["based_on_report_id"] else None,
            "mode": str(row["mode"]),
            "report": str(row["report_text"]),
            "citations": _decode_citations(str(row["citations_json"])),
            "workflow_steps": _decode_workflow_steps(str(row["workflow_steps_json"])),
            "status": str(row["status"]),
            "created_at": int(row["created_at"]),
            "updated_at": int(row["updated_at"]),
        }

    def _decode_knowledge_document_row(self, row: sqlite3.Row) -> dict[str, Any]:
        published_at = row["published_at"]
        return {
            "doc_id": str(row["doc_id"]),
            "kb_id": str(row["kb_id"]),
            "title": str(row["title"]),
            "source": str(row["source"]),
            "doc_type": str(row["doc_type"]),
            "symbols": _json_loads(str(row["symbols_json"]), []),
            "tags": _json_loads(str(row["tags_json"]), []),
            "file_name": str(row["file_name"] or ""),
            "content_type": str(row["content_type"] or "text/plain"),
            "checksum": str(row["checksum"] or ""),
            "status": str(row["status"]),
            "chunk_count": int(row["chunk_count"] or 0),
            "uploaded_by": str(row["uploaded_by"]),
            "published_at": datetime.fromtimestamp(int(published_at), tz=timezone.utc) if published_at else None,
            "created_at": int(row["created_at"]),
            "updated_at": int(row["updated_at"]),
        }

    def _insert_event(
        self,
        *,
        conn: sqlite3.Connection,
        conversation_id: str,
        turn_id: str,
        request_id: str,
        event_type: str,
        payload: dict[str, Any],
        to_outbox: bool,
    ) -> None:
        now = _now_ts()
        payload_json = json.dumps(payload, ensure_ascii=False)
        conn.execute(
            """
            INSERT INTO conversation_event(
              conversation_id, turn_id, request_id, event_type, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (conversation_id, turn_id, request_id, event_type, payload_json, now),
        )
        if to_outbox:
            conn.execute(
                """
                INSERT INTO outbox_event(
                  conversation_id, turn_id, request_id, event_type, payload_json, status, attempts, last_error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'pending', 0, NULL, ?, ?)
                """,
                (conversation_id, turn_id, request_id, event_type, payload_json, now, now),
            )
