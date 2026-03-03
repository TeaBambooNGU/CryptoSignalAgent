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
        runtime.report_agent.llm = cls._StubLLM()

    @classmethod
    def tearDownClass(cls) -> None:
        runtime = cls._app.state.runtime
        runtime.report_agent.llm = cls._original_llm
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


if __name__ == "__main__":
    unittest.main()
