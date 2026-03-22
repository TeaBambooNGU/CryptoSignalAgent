"""上下文压缩与全文摘要管理测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.config.settings import Settings
from app.conversation.context_manager import ConversationContextManager
from app.conversation.store import SQLiteConversationTruthStore


class _CompressionLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def invoke(self, messages):
        content = str(getattr(messages[-1], "content", ""))
        self.calls.append(content)
        return "COMPRESSED_CONTEXT"


class _FullSummaryLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def invoke(self, messages):
        content = str(getattr(messages[-1], "content", ""))
        self.calls.append(content)
        return "FULL_CONVERSATION_SUMMARY"


class ConversationContextManagerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store = SQLiteConversationTruthStore(str(Path(self._tmpdir.name) / "context_manager.db"))
        self.settings = Settings(
            mem0_enabled=False,
            milvus_enabled=False,
            context_archive_dir=str(Path(self._tmpdir.name) / "archives"),
            context_compression_trigger_tokens=120,
            context_compression_hard_limit_tokens=180,
            context_recent_turn_window=1,
            context_full_summary_every_n_compressions=5,
        )
        self.compression_llm = _CompressionLLM()
        self.full_summary_llm = _FullSummaryLLM()
        self.manager = ConversationContextManager(
            settings=self.settings,
            truth_store=self.store,
            compression_llm=self.compression_llm,
            full_summary_llm=self.full_summary_llm,
        )

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _append_turn(self, *, conversation_id: str, version: int, body_seed: str) -> str:
        request_id = f"req-{version}"
        prepared = self.store.prepare_turn(
            conversation_id=conversation_id,
            turn_id=None,
            request_id=request_id,
            expected_version=version - 1,
        )
        body = f"{body_seed}-" + ("内容很长" * 30)
        self.store.save_turn_result(
            request_id=request_id,
            user_id="u-context",
            query_text=f"query-{version}-{body}",
            task_context={"step": version},
            assistant_message=f"assistant-{version}-{body}",
            response={
                "report": f"report-{version}-{body}",
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
        return prepared.turn_id

    def test_single_request_triggers_one_compression_and_archives_raw_file(self) -> None:
        conversation_id = "conv-compress-once"
        for version in range(1, 7):
            self._append_turn(
                conversation_id=conversation_id,
                version=version,
                body_seed=f"seed-{version}",
            )

        with self.assertLogs("app.conversation.context_manager", level="INFO") as captured:
            result = self.manager.ensure_context_budget(
                conversation_id=conversation_id,
                anchor_turn_id=None,
                include_latest_report=False,
                pending_user_message="继续分析",
            )

        compressions = self.store.list_context_compressions(
            conversation_id=conversation_id,
            anchor_turn_id=None,
        )
        self.assertEqual(len(compressions), 1)
        self.assertTrue(result["hard_limit_exceeded"] or result["token_estimate"] > 0)
        state = self.store.get_context_state_for_anchor(
            conversation_id=conversation_id,
            anchor_turn_id=None,
        )
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(state["compression_round_count"], 1)
        self.assertEqual(state["latest_materialized_version"], 5)
        raw_file = Path(compressions[0]["raw_file_path"])
        self.assertTrue(raw_file.exists())
        archive_text = raw_file.read_text(encoding="utf-8")
        self.assertIn("## Turn v1", archive_text)
        self.assertIn("assistant-1-seed-1", archive_text)
        joined_logs = "\n".join(captured.output)
        self.assertIn("context.compress.triggered", joined_logs)
        self.assertIn("context.compress.completed", joined_logs)

    def test_five_compressions_trigger_full_summary_from_raw_files(self) -> None:
        conversation_id = "conv-full-summary"
        self._append_turn(conversation_id=conversation_id, version=1, body_seed="seed-1")

        for version in range(2, 7):
            self._append_turn(
                conversation_id=conversation_id,
                version=version,
                body_seed=f"seed-{version}",
            )
            if version < 6:
                self.manager.ensure_context_budget(
                    conversation_id=conversation_id,
                    anchor_turn_id=None,
                    include_latest_report=False,
                    pending_user_message=f"继续-{version}",
                )
                continue
            with self.assertLogs("app.conversation.context_manager", level="INFO") as captured:
                self.manager.ensure_context_budget(
                    conversation_id=conversation_id,
                    anchor_turn_id=None,
                    include_latest_report=False,
                    pending_user_message=f"继续-{version}",
                )

        summary = self.store.get_full_context_summary(
            conversation_id=conversation_id,
            anchor_turn_id=None,
        )
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["summary_text"], "FULL_CONVERSATION_SUMMARY")
        self.assertEqual(summary["source_round_upto"], 5)
        self.assertEqual(len(summary["raw_manifest"]), 5)
        self.assertTrue(self.full_summary_llm.calls)
        full_summary_prompt = self.full_summary_llm.calls[-1]
        self.assertIn("## Turn v1", full_summary_prompt)
        self.assertIn("assistant-1-seed-1", full_summary_prompt)
        self.assertNotIn("COMPRESSED_CONTEXT", full_summary_prompt)
        joined_logs = "\n".join(captured.output)
        self.assertIn("context.full_summary.triggered", joined_logs)
        self.assertIn("context.full_summary.completed", joined_logs)

    def test_prompt_uses_full_summary_and_incremental_compressions(self) -> None:
        conversation_id = "conv-prompt-assembly"
        self._append_turn(conversation_id=conversation_id, version=1, body_seed="seed-1")

        for version in range(2, 7):
            self._append_turn(
                conversation_id=conversation_id,
                version=version,
                body_seed=f"seed-{version}",
            )
            self.manager.ensure_context_budget(
                conversation_id=conversation_id,
                anchor_turn_id=None,
                include_latest_report=False,
                pending_user_message=f"继续-{version}",
            )

        self._append_turn(conversation_id=conversation_id, version=7, body_seed="seed-7")
        self.manager.ensure_context_budget(
            conversation_id=conversation_id,
            anchor_turn_id=None,
            include_latest_report=False,
            pending_user_message="继续-7",
        )

        prompt = self.manager.build_context_prompt(
            conversation_id=conversation_id,
            anchor_turn_id=None,
            include_latest_report=False,
        )
        self.assertIn("[全文摘要]", prompt)
        self.assertIn("[增量压缩片段]", prompt)
        self.assertIn("[未压缩原始轮次]", prompt)

    def test_branch_anchor_assets_are_isolated_from_mainline(self) -> None:
        conversation_id = "conv-anchor-isolation"
        root_turn_id = ""
        for version in range(1, 6):
            turn_id = self._append_turn(
                conversation_id=conversation_id,
                version=version,
                body_seed=f"main-{version}",
            )
            if version == 3:
                root_turn_id = turn_id

        self.manager.ensure_context_budget(
            conversation_id=conversation_id,
            anchor_turn_id=None,
            include_latest_report=False,
            pending_user_message="主线继续",
        )

        parent_turn_id = root_turn_id
        for version in (6, 7):
            request_id = f"req-branch-{version}"
            prepared = self.store.prepare_turn(
                conversation_id=conversation_id,
                turn_id=f"branch-turn-{version}",
                request_id=request_id,
                expected_version=version - 1,
            )
            body = f"branch-{version}-" + ("内容很长" * 30)
            self.store.save_turn_result(
                request_id=request_id,
                user_id="u-context",
                query_text=f"branch-query-{version}-{body}",
                task_context={"step": version},
                assistant_message=f"branch-assistant-{version}-{body}",
                response={
                    "report": f"branch-report-{version}-{body}",
                    "citations": [],
                    "trace_id": f"trace-branch-{version}",
                    "conversation_id": conversation_id,
                    "turn_id": prepared.turn_id,
                    "request_id": request_id,
                    "conversation_version": prepared.conversation_version,
                    "errors": [],
                    "workflow_steps": [{"node_id": "finalize_response", "status": "success", "duration_ms": 1}],
                },
                parent_turn_id=parent_turn_id,
            )
            parent_turn_id = prepared.turn_id

        self.manager.ensure_context_budget(
            conversation_id=conversation_id,
            anchor_turn_id=parent_turn_id,
            include_latest_report=False,
            pending_user_message="分支继续",
        )

        main_state = self.store.get_context_state_for_anchor(
            conversation_id=conversation_id,
            anchor_turn_id=None,
        )
        branch_state = self.store.get_context_state_for_anchor(
            conversation_id=conversation_id,
            anchor_turn_id=parent_turn_id,
        )
        assert main_state is not None
        assert branch_state is not None
        self.assertIsNone(main_state["anchor_turn_id"])
        self.assertEqual(branch_state["anchor_turn_id"], parent_turn_id)

        main_compressions = self.store.list_context_compressions(
            conversation_id=conversation_id,
            anchor_turn_id=None,
        )
        branch_compressions = self.store.list_context_compressions(
            conversation_id=conversation_id,
            anchor_turn_id=parent_turn_id,
        )
        self.assertEqual(len(main_compressions), 1)
        self.assertEqual(len(branch_compressions), 1)
        self.assertNotEqual(
            Path(main_compressions[0]["raw_file_path"]).parent,
            Path(branch_compressions[0]["raw_file_path"]).parent,
        )
        self.assertIn("__mainline__", main_compressions[0]["raw_file_path"])
        self.assertIn(parent_turn_id, branch_compressions[0]["raw_file_path"])
