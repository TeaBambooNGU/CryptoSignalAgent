"""会话历史 API 测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
from importlib import reload
from pathlib import Path

from fastapi.testclient import TestClient


class ConversationAPITestCase(unittest.TestCase):
    class _StubLLM:
        def invoke(self, messages):
            del messages
            return "会话恢复测试报告"

    class _ActionClassifierLLM:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def invoke(self, messages):
            content = str(getattr(messages[-1], "content", ""))
            self.calls.append(content)
            if "改成更保守" in content or "改写" in content or "重写" in content or "润色" in content:
                return '{"action":"rewrite_report"}'
            if (
                "生成" in content
                or "来一版" in content
                or "给我一版" in content
                or "先给我一版" in content
                or "出个报告" in content
                or "重新分析" in content
            ):
                return '{"action":"regenerate_report"}'
            return '{"action":"chat"}'

    class _FailingActionClassifierLLM:
        def invoke(self, messages):
            del messages
            raise RuntimeError("classifier unavailable")

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        os.environ["MILVUS_ENABLED"] = "false"
        os.environ["MEM0_ENABLED"] = "false"
        os.environ["MCP_CONFIG_PATH"] = "/tmp/crypto_signal_agent_test_no_mcp.json"
        os.environ["LLM_PROVIDER"] = "minimax"
        os.environ["MINIMAX_API_KEY"] = "test-minimax-key"
        os.environ["ZHIPUAI_API_KEY"] = ""
        os.environ["CONVERSATION_STORE_PATH"] = str(Path(cls._tmpdir.name) / "conversation_api.db")

        import app.config.settings as settings_module
        import app.main as main_module

        reload(settings_module)
        reload(main_module)

        cls._app = main_module.create_app()
        cls._client_cm = TestClient(cls._app)
        cls.client = cls._client_cm.__enter__()

        runtime = cls._app.state.runtime
        cls._original_llm = runtime.report_agent.llm
        cls._original_action_classifier_llm = runtime.conversation_service.action_classifier_llm
        cls._action_classifier_llm = cls._ActionClassifierLLM()
        runtime.report_agent.llm = cls._StubLLM()
        runtime.conversation_service.action_classifier_llm = cls._action_classifier_llm

    @classmethod
    def tearDownClass(cls) -> None:
        runtime = cls._app.state.runtime
        runtime.report_agent.llm = cls._original_llm
        runtime.conversation_service.action_classifier_llm = cls._original_action_classifier_llm
        cls._client_cm.__exit__(None, None, None)
        cls._tmpdir.cleanup()

    def test_conversation_meta_and_turns_api(self) -> None:
        conversation_id = "conv-api-1"
        first = self.client.post(
            "/v1/research/query",
            json={
                "user_id": "u-conv-api",
                "query": "第一轮问题",
                "conversation_id": conversation_id,
                "request_id": "req-conv-api-1",
            },
        )
        self.assertEqual(first.status_code, 200)

        second = self.client.post(
            "/v1/research/query",
            json={
                "user_id": "u-conv-api",
                "query": "第二轮问题",
                "conversation_id": conversation_id,
                "request_id": "req-conv-api-2",
                "expected_version": 1,
            },
        )
        self.assertEqual(second.status_code, 200)

        meta_resp = self.client.get(f"/v1/conversation/{conversation_id}")
        self.assertEqual(meta_resp.status_code, 200)
        meta_payload = meta_resp.json()
        self.assertEqual(meta_payload["latest_version"], 2)
        self.assertEqual(meta_payload["turn_count"], 2)

        turns_resp = self.client.get(f"/v1/conversation/{conversation_id}/turns", params={"limit": 1})
        self.assertEqual(turns_resp.status_code, 200)
        turns_payload = turns_resp.json()
        self.assertEqual(len(turns_payload), 1)
        self.assertEqual(turns_payload[0]["version"], 2)

        turn_id = turns_payload[0]["turn_id"]
        detail_resp = self.client.get(f"/v1/conversation/{conversation_id}/turns/{turn_id}")
        self.assertEqual(detail_resp.status_code, 200)
        detail_payload = detail_resp.json()
        self.assertEqual(detail_payload["status"], "completed")
        self.assertEqual(detail_payload["version"], 2)
        self.assertEqual(detail_payload["intent"], "regenerate_report")
        self.assertTrue(isinstance(detail_payload["workflow_steps"], list))

        reports_resp = self.client.get(f"/v1/conversation/{conversation_id}/reports")
        self.assertEqual(reports_resp.status_code, 200)
        reports_payload = reports_resp.json()
        self.assertEqual(len(reports_payload), 2)
        self.assertEqual(reports_payload[0]["report_version"], 2)
        report_id = reports_payload[0]["report_id"]
        report_detail = self.client.get(f"/v1/conversation/{conversation_id}/reports/{report_id}")
        self.assertEqual(report_detail.status_code, 200)
        self.assertEqual(report_detail.json()["report_id"], report_id)

    def test_conversation_message_chat_rewrite_regenerate(self) -> None:
        conversation_id = "conv-message-1"
        first = self.client.post(
            f"/v1/conversation/{conversation_id}/message",
            json={
                "user_id": "u-msg-1",
                "message": "先给我一版 BTC 报告",
                "action": "auto",
                "request_id": "req-msg-1",
            },
        )
        self.assertEqual(first.status_code, 200)
        first_payload = first.json()
        self.assertEqual(first_payload["action_taken"], "regenerate_report")
        self.assertTrue(first_payload["report"] is not None)
        report_id = first_payload["report"]["report_id"]

        chat = self.client.post(
            f"/v1/conversation/{conversation_id}/message",
            json={
                "user_id": "u-msg-1",
                "message": "解释下你上一版报告的主要风险",
                "action": "auto",
                "request_id": "req-msg-2",
                "expected_version": first_payload["conversation_version"],
            },
        )
        self.assertEqual(chat.status_code, 200)
        chat_payload = chat.json()
        self.assertEqual(chat_payload["action_taken"], "chat")
        self.assertTrue(chat_payload["report"] is None)
        self.assertTrue(chat_payload["assistant_message"])
        chat_detail = self.client.get(
            f"/v1/conversation/{conversation_id}/turns/{chat_payload['turn_id']}"
        ).json()
        self.assertEqual(chat_detail["parent_turn_id"], first_payload["turn_id"])

        rewrite = self.client.post(
            f"/v1/conversation/{conversation_id}/message",
            json={
                "user_id": "u-msg-1",
                "message": "把报告改成更保守的版本",
                "action": "rewrite_report",
                "target_report_id": report_id,
                "request_id": "req-msg-3",
                "expected_version": chat_payload["conversation_version"],
            },
        )
        self.assertEqual(rewrite.status_code, 200)
        rewrite_payload = rewrite.json()
        self.assertEqual(rewrite_payload["action_taken"], "rewrite_report")
        self.assertTrue(rewrite_payload["report"] is not None)
        self.assertEqual(rewrite_payload["report"]["mode"], "rewrite")
        self.assertEqual(rewrite_payload["report"]["based_on_report_id"], report_id)

        reports_resp = self.client.get(f"/v1/conversation/{conversation_id}/reports")
        self.assertEqual(reports_resp.status_code, 200)
        reports_payload = reports_resp.json()
        self.assertEqual(len(reports_payload), 2)
        self.assertEqual(reports_payload[0]["mode"], "rewrite")
        self.assertEqual(reports_payload[1]["mode"], "regenerate")

        idem_retry = self.client.post(
            f"/v1/conversation/{conversation_id}/message",
            json={
                "user_id": "u-msg-1",
                "message": "把报告改成更保守的版本",
                "action": "rewrite_report",
                "target_report_id": report_id,
                "request_id": "req-msg-3",
            },
        )
        self.assertEqual(idem_retry.status_code, 200)
        self.assertEqual(idem_retry.json()["turn_id"], rewrite_payload["turn_id"])
        self.assertEqual(idem_retry.json()["conversation_version"], rewrite_payload["conversation_version"])

    def test_auto_action_allows_chat_before_first_report(self) -> None:
        conversation_id = "conv-auto-chat-first"
        chat_first = self.client.post(
            f"/v1/conversation/{conversation_id}/message",
            json={
                "user_id": "u-auto-chat-first",
                "message": "先聊聊今天市场风险",
                "action": "auto",
                "request_id": "req-auto-chat-first-1",
            },
        )
        self.assertEqual(chat_first.status_code, 200)
        chat_payload = chat_first.json()
        self.assertEqual(chat_payload["action_taken"], "chat")
        self.assertTrue(chat_payload["report"] is None)

        generate = self.client.post(
            f"/v1/conversation/{conversation_id}/message",
            json={
                "user_id": "u-auto-chat-first",
                "message": "请生成一版 BTC 报告",
                "action": "auto",
                "request_id": "req-auto-chat-first-2",
                "expected_version": chat_payload["conversation_version"],
            },
        )
        self.assertEqual(generate.status_code, 200)
        generate_payload = generate.json()
        self.assertEqual(generate_payload["action_taken"], "regenerate_report")
        self.assertTrue(generate_payload["report"] is not None)

    def test_auto_action_uses_llm_to_route_rewrite(self) -> None:
        conversation_id = "conv-auto-llm-rewrite"
        service = self._app.state.runtime.conversation_service

        first = self.client.post(
            f"/v1/conversation/{conversation_id}/message",
            json={
                "user_id": "u-auto-llm-rewrite",
                "message": "先给我一版 BTC 报告",
                "action": "auto",
                "request_id": "req-auto-llm-rewrite-1",
            },
        )
        self.assertEqual(first.status_code, 200)
        first_payload = first.json()
        self.assertEqual(first_payload["action_taken"], "regenerate_report")

        call_count_before = len(self._action_classifier_llm.calls)
        rewrite = self.client.post(
            f"/v1/conversation/{conversation_id}/message",
            json={
                "user_id": "u-auto-llm-rewrite",
                "message": "把这版报告改成更保守的版本",
                "action": "auto",
                "request_id": "req-auto-llm-rewrite-2",
                "expected_version": first_payload["conversation_version"],
            },
        )
        self.assertEqual(rewrite.status_code, 200)
        rewrite_payload = rewrite.json()
        self.assertEqual(rewrite_payload["action_taken"], "rewrite_report")
        self.assertEqual(len(service.action_classifier_llm.calls), call_count_before + 1)

    def test_auto_action_falls_back_to_rules_when_classifier_errors(self) -> None:
        conversation_id = "conv-auto-classifier-fallback"
        service = self._app.state.runtime.conversation_service
        original_classifier = service.action_classifier_llm
        service.action_classifier_llm = self._FailingActionClassifierLLM()
        try:
            response = self.client.post(
                f"/v1/conversation/{conversation_id}/message",
                json={
                    "user_id": "u-auto-classifier-fallback",
                    "message": "请生成一版 BTC 报告",
                    "action": "auto",
                    "request_id": "req-auto-classifier-fallback-1",
                },
            )
        finally:
            service.action_classifier_llm = original_classifier

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["action_taken"], "regenerate_report")

    def test_context_summary_is_generated_for_long_conversation(self) -> None:
        conversation_id = "conv-summary-1"
        for idx in range(1, 11):
            resp = self.client.post(
                f"/v1/conversation/{conversation_id}/message",
                json={
                    "user_id": "u-summary",
                    "message": f"第{idx}轮，继续解释报告",
                    "action": "chat" if idx > 1 else "regenerate_report",
                    "request_id": f"req-summary-{idx}",
                    "expected_version": idx - 1 if idx > 1 else None,
                },
            )
            self.assertEqual(resp.status_code, 200)
        summary = self._app.state.runtime.conversation_store.get_context_summary(conversation_id=conversation_id)
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertGreaterEqual(summary["through_version"], 1)
        self.assertTrue(summary["summary_text"])

    def test_resume_conversation_api(self) -> None:
        conversation_id = "conv-resume-api"
        first = self.client.post(
            "/v1/research/query",
            json={
                "user_id": "u-conv-resume",
                "query": "先给我第一轮",
                "conversation_id": conversation_id,
                "request_id": "req-resume-1",
            },
        )
        self.assertEqual(first.status_code, 200)
        first_payload = first.json()

        resume = self.client.post(
            f"/v1/conversation/{conversation_id}/resume",
            json={
                "user_id": "u-conv-resume",
                "query": "继续第二轮",
                "from_turn_id": first_payload["turn_id"],
                "expected_version": first_payload["conversation_version"],
                "request_id": "req-resume-2",
            },
        )
        self.assertEqual(resume.status_code, 200)
        resume_payload = resume.json()
        self.assertEqual(resume_payload["conversation_version"], 2)
        self.assertNotEqual(resume_payload["turn_id"], first_payload["turn_id"])
        resume_detail = self.client.get(
            f"/v1/conversation/{conversation_id}/turns/{resume_payload['turn_id']}"
        ).json()
        self.assertEqual(resume_detail["parent_turn_id"], first_payload["turn_id"])

        missing = self.client.post(
            f"/v1/conversation/{conversation_id}/resume",
            json={
                "user_id": "u-conv-resume",
                "query": "无效恢复",
                "from_turn_id": "turn-not-exists",
                "request_id": "req-resume-3",
            },
        )
        self.assertEqual(missing.status_code, 404)

    def test_resume_branch_lineage_is_isolated(self) -> None:
        conversation_id = "conv-resume-branch"
        first = self.client.post(
            "/v1/research/query",
            json={
                "user_id": "u-branch",
                "query": "主线第一轮",
                "conversation_id": conversation_id,
                "request_id": "req-branch-1",
            },
        )
        self.assertEqual(first.status_code, 200)
        first_payload = first.json()

        second = self.client.post(
            "/v1/research/query",
            json={
                "user_id": "u-branch",
                "query": "主线第二轮",
                "conversation_id": conversation_id,
                "request_id": "req-branch-2",
                "expected_version": first_payload["conversation_version"],
            },
        )
        self.assertEqual(second.status_code, 200)
        second_payload = second.json()

        branch = self.client.post(
            f"/v1/conversation/{conversation_id}/resume",
            json={
                "user_id": "u-branch",
                "query": "从第一轮分叉重跑",
                "from_turn_id": first_payload["turn_id"],
                "expected_version": second_payload["conversation_version"],
                "request_id": "req-branch-3",
            },
        )
        self.assertEqual(branch.status_code, 200)
        branch_payload = branch.json()

        branch_chat = self.client.post(
            f"/v1/conversation/{conversation_id}/message",
            json={
                "user_id": "u-branch",
                "message": "继续分支追问",
                "action": "chat",
                "from_turn_id": branch_payload["turn_id"],
                "expected_version": branch_payload["conversation_version"],
                "request_id": "req-branch-4",
            },
        )
        self.assertEqual(branch_chat.status_code, 200)
        branch_chat_payload = branch_chat.json()

        runtime_store = self._app.state.runtime.conversation_store
        lineage = runtime_store.list_turn_lineage(
            conversation_id=conversation_id,
            leaf_turn_id=branch_chat_payload["turn_id"],
            limit=10,
        )
        lineage_turn_ids = [row["turn_id"] for row in lineage]
        self.assertEqual(lineage_turn_ids[:3], [branch_chat_payload["turn_id"], branch_payload["turn_id"], first_payload["turn_id"]])
        self.assertNotIn(second_payload["turn_id"], lineage_turn_ids)

    def test_rewrite_without_target_report_uses_branch_report(self) -> None:
        conversation_id = "conv-rewrite-branch"
        first = self.client.post(
            f"/v1/conversation/{conversation_id}/message",
            json={
                "user_id": "u-rewrite-branch",
                "message": "生成第一版报告",
                "action": "regenerate_report",
                "request_id": "req-rewrite-branch-1",
            },
        )
        self.assertEqual(first.status_code, 200)
        first_payload = first.json()
        first_report_id = first_payload.get("report", {}).get("report_id")
        self.assertTrue(first_report_id)

        second = self.client.post(
            f"/v1/conversation/{conversation_id}/message",
            json={
                "user_id": "u-rewrite-branch",
                "message": "主线再生成一版",
                "action": "regenerate_report",
                "request_id": "req-rewrite-branch-2",
                "expected_version": first_payload["conversation_version"],
            },
        )
        self.assertEqual(second.status_code, 200)
        second_payload = second.json()

        rewrite = self.client.post(
            f"/v1/conversation/{conversation_id}/message",
            json={
                "user_id": "u-rewrite-branch",
                "message": "从第一版分支改写",
                "action": "rewrite_report",
                "from_turn_id": first_payload["turn_id"],
                "request_id": "req-rewrite-branch-3",
                "expected_version": second_payload["conversation_version"],
            },
        )
        self.assertEqual(rewrite.status_code, 200)
        rewrite_payload = rewrite.json()
        self.assertEqual(rewrite_payload["action_taken"], "rewrite_report")
        self.assertEqual(rewrite_payload["report"]["based_on_report_id"], first_report_id)


if __name__ == "__main__":
    unittest.main()
