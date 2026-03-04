"""会话恢复能力测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from app.config.settings import Settings
from app.conversation.errors import ConversationConflictError
from app.conversation.store import SQLiteConversationTruthStore
from app.memory.mem0_service import MemoryService
from app.memory.session_store import InMemorySessionMemoryStore


class _DummyMilvusStore:
    def query_user_memory(self, user_id: str, limit: int) -> list[dict[str, Any]]:
        del user_id, limit
        return []

    def upsert_user_memory(self, records: list[dict[str, Any]]) -> None:
        del records


class ConversationRecoveryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "conversation_recovery.db"
        self.store = SQLiteConversationTruthStore(str(self.db_path))

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _append_turn(self, *, conversation_id: str, version: int) -> None:
        request_id = f"req-{version}"
        prepared = self.store.prepare_turn(
            conversation_id=conversation_id,
            turn_id=None,
            request_id=request_id,
            expected_version=version - 1,
        )
        self.store.save_turn_result(
            request_id=request_id,
            user_id="u-recovery",
            query_text=f"query-{version}",
            task_context={"step": version},
            response={
                "report": f"report-{version}",
                "citations": [],
                "trace_id": f"trace-{version}",
                "conversation_id": conversation_id,
                "turn_id": prepared.turn_id,
                "request_id": request_id,
                "conversation_version": prepared.conversation_version,
                "errors": [],
                "workflow_steps": [{"node_id": "finalize_response", "status": "success", "duration_ms": 1}],
            },
        )

    def test_restart_can_restore_turn_history(self) -> None:
        conversation_id = "conv-recovery-1"
        for version in range(1, 6):
            self._append_turn(conversation_id=conversation_id, version=version)

        restarted_store = SQLiteConversationTruthStore(str(self.db_path))
        meta = restarted_store.get_conversation_meta(conversation_id)
        self.assertIsNotNone(meta)
        assert meta is not None
        self.assertEqual(meta["latest_version"], 5)
        self.assertEqual(meta["latest_turn_id"], "turn-5")
        self.assertEqual(meta["turn_count"], 5)

        turns = restarted_store.list_turns(conversation_id=conversation_id, limit=10)
        self.assertEqual(len(turns), 5)
        self.assertEqual(turns[0]["version"], 5)
        detail = restarted_store.get_turn(conversation_id=conversation_id, turn_id="turn-3")
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertEqual(detail["query"], "query-3")
        self.assertEqual(detail["report"], "report-3")
        self.assertEqual(detail["workflow_steps"][0]["node_id"], "finalize_response")

    def test_request_id_retry_returns_cached_result(self) -> None:
        conversation_id = "conv-idem"
        self._append_turn(conversation_id=conversation_id, version=1)

        prepared = self.store.prepare_turn(
            conversation_id=conversation_id,
            turn_id=None,
            request_id="req-1",
            expected_version=1,
        )
        self.assertIsNotNone(prepared.cached_response)
        assert prepared.cached_response is not None
        self.assertEqual(prepared.cached_response["turn_id"], "turn-1")
        self.assertEqual(prepared.cached_response["conversation_version"], 1)
        self.assertEqual(prepared.cached_response["report"], "report-1")

    def test_report_version_store_and_read(self) -> None:
        conversation_id = "conv-report"
        prepared = self.store.prepare_turn(
            conversation_id=conversation_id,
            turn_id=None,
            request_id="req-report-1",
            expected_version=0,
        )
        saved = self.store.save_turn_result(
            request_id="req-report-1",
            user_id="u-recovery",
            query_text="生成报告",
            task_context={},
            assistant_message="report-body-1",
            response={
                "conversation_id": conversation_id,
                "turn_id": prepared.turn_id,
                "request_id": "req-report-1",
                "conversation_version": 1,
                "trace_id": "trace-report-1",
                "report": "report-body-1",
                "citations": [],
                "errors": [],
                "workflow_steps": [],
            },
            report_payload={
                "mode": "regenerate",
                "report": "report-body-1",
                "citations": [],
                "workflow_steps": [],
            },
        )
        self.assertIsNotNone(saved)
        assert saved is not None
        self.assertEqual(saved["report_version"], 1)
        reports = self.store.list_reports(conversation_id=conversation_id, limit=5)
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0]["report"], "report-body-1")

    def test_cas_conflict_raises_error(self) -> None:
        conversation_id = "conv-cas"
        self._append_turn(conversation_id=conversation_id, version=1)

        with self.assertRaises(ConversationConflictError):
            self.store.prepare_turn(
                conversation_id=conversation_id,
                turn_id=None,
                request_id="req-cas-2",
                expected_version=0,
            )

    def test_load_recent_turn_context_survives_session_cache_loss(self) -> None:
        conversation_id = "conv-cache-loss"
        self._append_turn(conversation_id=conversation_id, version=1)
        service = MemoryService(
            settings=Settings(mem0_enabled=False, milvus_enabled=False),
            milvus_store=_DummyMilvusStore(),
            session_store=InMemorySessionMemoryStore(),
            conversation_store=self.store,
        )

        profile = service.load_memory_profile(user_id="u-recovery", conversation_id=conversation_id)

        self.assertEqual(profile["session_memory"], [])
        self.assertEqual(len(profile["recent_turn_context"]), 1)
        self.assertEqual(profile["recent_turn_context"][0]["query"], "query-1")

    def test_lineage_and_latest_report_on_lineage(self) -> None:
        conversation_id = "conv-lineage"

        first = self.store.prepare_turn(
            conversation_id=conversation_id,
            turn_id=None,
            request_id="req-lineage-1",
            expected_version=0,
        )
        first_report = self.store.save_turn_result(
            request_id="req-lineage-1",
            user_id="u-lineage",
            query_text="根节点报告",
            task_context={},
            assistant_message="report-root",
            response={
                "conversation_id": conversation_id,
                "turn_id": first.turn_id,
                "request_id": "req-lineage-1",
                "conversation_version": first.conversation_version,
                "trace_id": "trace-lineage-1",
                "report": "report-root",
                "citations": [],
                "errors": [],
                "workflow_steps": [],
            },
            parent_turn_id=None,
            report_payload={
                "mode": "regenerate",
                "report": "report-root",
                "citations": [],
                "workflow_steps": [],
            },
        )
        assert first_report is not None

        branch = self.store.prepare_turn(
            conversation_id=conversation_id,
            turn_id=None,
            request_id="req-lineage-2",
            expected_version=1,
        )
        branch_report = self.store.save_turn_result(
            request_id="req-lineage-2",
            user_id="u-lineage",
            query_text="从根节点分支报告",
            task_context={},
            assistant_message="report-branch",
            response={
                "conversation_id": conversation_id,
                "turn_id": branch.turn_id,
                "request_id": "req-lineage-2",
                "conversation_version": branch.conversation_version,
                "trace_id": "trace-lineage-2",
                "report": "report-branch",
                "citations": [],
                "errors": [],
                "workflow_steps": [],
            },
            parent_turn_id=first.turn_id,
            report_payload={
                "mode": "regenerate",
                "report": "report-branch",
                "citations": [],
                "workflow_steps": [],
            },
        )
        assert branch_report is not None

        main = self.store.prepare_turn(
            conversation_id=conversation_id,
            turn_id=None,
            request_id="req-lineage-3",
            expected_version=2,
        )
        main_report = self.store.save_turn_result(
            request_id="req-lineage-3",
            user_id="u-lineage",
            query_text="主线后续报告",
            task_context={},
            assistant_message="report-main",
            response={
                "conversation_id": conversation_id,
                "turn_id": main.turn_id,
                "request_id": "req-lineage-3",
                "conversation_version": main.conversation_version,
                "trace_id": "trace-lineage-3",
                "report": "report-main",
                "citations": [],
                "errors": [],
                "workflow_steps": [],
            },
            parent_turn_id=first.turn_id,
            report_payload={
                "mode": "regenerate",
                "report": "report-main",
                "citations": [],
                "workflow_steps": [],
            },
        )
        assert main_report is not None

        branch_chat = self.store.prepare_turn(
            conversation_id=conversation_id,
            turn_id=None,
            request_id="req-lineage-4",
            expected_version=3,
        )
        self.store.save_turn_result(
            request_id="req-lineage-4",
            user_id="u-lineage",
            query_text="分支继续追问",
            task_context={},
            assistant_message="chat-branch",
            response={
                "conversation_id": conversation_id,
                "turn_id": branch_chat.turn_id,
                "request_id": "req-lineage-4",
                "conversation_version": branch_chat.conversation_version,
                "trace_id": "trace-lineage-4",
                "report": "chat-branch",
                "citations": [],
                "errors": [],
                "workflow_steps": [],
            },
            parent_turn_id=branch.turn_id,
            report_payload=None,
        )

        lineage_turns = self.store.list_turn_lineage(
            conversation_id=conversation_id,
            leaf_turn_id=branch_chat.turn_id,
            limit=10,
        )
        lineage_turn_ids = [row["turn_id"] for row in lineage_turns]
        self.assertEqual(lineage_turn_ids[:3], [branch_chat.turn_id, branch.turn_id, first.turn_id])
        self.assertNotIn(main.turn_id, lineage_turn_ids)

        global_latest_report = self.store.get_latest_report(conversation_id=conversation_id)
        assert global_latest_report is not None
        self.assertEqual(global_latest_report["report_id"], main_report["report_id"])

        branch_latest_report = self.store.get_latest_report_on_lineage(
            conversation_id=conversation_id,
            leaf_turn_id=branch_chat.turn_id,
        )
        assert branch_latest_report is not None
        self.assertEqual(branch_latest_report["report_id"], branch_report["report_id"])


if __name__ == "__main__":
    unittest.main()
