"""API 基础联调测试。"""

from __future__ import annotations

import os
import unittest
from importlib import reload
from pathlib import Path

from fastapi.testclient import TestClient


class APITestCase(unittest.TestCase):
    """覆盖核心接口的烟雾测试。"""

    class _StubLLM:
        def invoke(self, messages):
            del messages
            return "测试报告正文"

    class _FailingLLM:
        def invoke(self, messages):
            del messages
            raise RuntimeError("llm unavailable")

    @classmethod
    def setUpClass(cls) -> None:
        # 测试环境关闭外部依赖，保证本地可重复执行。
        os.environ["MILVUS_ENABLED"] = "false"
        os.environ["MEM0_ENABLED"] = "false"
        os.environ["MCP_CONFIG_PATH"] = "/tmp/crypto_signal_agent_test_no_mcp.json"
        os.environ["LLM_PROVIDER"] = "minimax"
        os.environ["MINIMAX_API_KEY"] = "test-minimax-key"
        os.environ["ZHIPUAI_API_KEY"] = ""
        os.environ["CONVERSATION_STORE_PATH"] = "/tmp/crypto_signal_agent_test_conversation.db"
        db_path = Path(os.environ["CONVERSATION_STORE_PATH"])
        if db_path.exists():
            db_path.unlink()

        import app.config.settings as settings_module

        reload(settings_module)

        from app.main import create_app

        cls._app = create_app()
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

    def test_update_preferences_and_get_profile(self) -> None:
        update_resp = self.client.post(
            "/v1/user/preferences",
            json={
                "user_id": "u-test-1",
                "preference": {
                    "watchlist": ["BTC", "ETH"],
                    "risk_preference": "balanced",
                },
                "confidence": 0.9,
            },
        )
        self.assertEqual(update_resp.status_code, 200)

        profile_resp = self.client.get("/v1/user/profile/u-test-1")
        self.assertEqual(profile_resp.status_code, 200)
        payload = profile_resp.json()
        self.assertEqual(payload["user_id"], "u-test-1")
        self.assertTrue(isinstance(payload["long_term_memory"], list))

    def test_research_query(self) -> None:
        resp = self.client.post(
            "/v1/research/query",
            json={
                "user_id": "u-test-2",
                "query": "请分析 BTC 和 ETH 的短线风险",
                "task_context": {"symbols": ["BTC", "ETH"]},
            },
        )
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertIn("report", payload)
        self.assertIn("trace_id", payload)
        self.assertIn("conversation_id", payload)
        self.assertIn("turn_id", payload)
        self.assertIn("request_id", payload)
        self.assertIn("conversation_version", payload)
        self.assertTrue(isinstance(payload["citations"], list))
        self.assertEqual(len(payload["workflow_steps"]), 9)
        self.assertTrue(all("node_id" in step and "duration_ms" in step for step in payload["workflow_steps"]))
        self.assertEqual(resp.headers.get("X-Trace-Id"), payload["trace_id"])

    def test_research_query_uses_request_trace_id(self) -> None:
        request_trace_id = "trace-test-fixed-id"
        resp = self.client.post(
            "/v1/research/query",
            headers={"X-Trace-Id": request_trace_id},
            json={
                "user_id": "u-test-2",
                "query": "请分析 BTC 和 ETH 的短线风险",
                "task_context": {"symbols": ["BTC", "ETH"]},
            },
        )
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload["trace_id"], request_trace_id)
        self.assertEqual(resp.headers.get("X-Trace-Id"), request_trace_id)

    def test_research_query_idempotency_by_request_id(self) -> None:
        base_payload = {
            "user_id": "u-test-idem",
            "query": "请分析 BTC 和 ETH 的短线风险",
            "task_context": {"symbols": ["BTC", "ETH"]},
            "conversation_id": "conv-idem-1",
            "turn_id": "turn-1",
            "request_id": "req-idem-1",
        }
        first = self.client.post("/v1/research/query", json=base_payload)
        self.assertEqual(first.status_code, 200)
        first_payload = first.json()

        second = self.client.post(
            "/v1/research/query",
            json={**base_payload, "query": "这个字段应被幂等返回忽略"},
        )
        self.assertEqual(second.status_code, 200)
        second_payload = second.json()

        self.assertEqual(first_payload["conversation_id"], second_payload["conversation_id"])
        self.assertEqual(first_payload["turn_id"], second_payload["turn_id"])
        self.assertEqual(first_payload["request_id"], second_payload["request_id"])
        self.assertEqual(first_payload["conversation_version"], second_payload["conversation_version"])

    def test_research_query_returns_409_on_cas_conflict(self) -> None:
        first = self.client.post(
            "/v1/research/query",
            json={
                "user_id": "u-test-cas",
                "query": "首次请求",
                "conversation_id": "conv-cas-1",
                "request_id": "req-cas-1",
            },
        )
        self.assertEqual(first.status_code, 200)

        conflict = self.client.post(
            "/v1/research/query",
            json={
                "user_id": "u-test-cas",
                "query": "冲突请求",
                "conversation_id": "conv-cas-1",
                "request_id": "req-cas-2",
                "expected_version": 0,
            },
        )
        self.assertEqual(conflict.status_code, 409)
        detail = conflict.json()["detail"]
        self.assertEqual(detail["error"], "conversation_version_conflict")

    def test_knowledge_document_create_and_list(self) -> None:
        create_resp = self.client.post(
            "/v1/knowledge/documents",
            json={
                "user_id": "u-kb-1",
                "document": {
                    "title": "BTC Liquidity Framework",
                    "source": "research-desk",
                    "doc_type": "research_report",
                    "symbols": ["BTC", "ETH"],
                    "tags": ["macro", "liquidity"],
                    "text": "BTC 与 ETH 流动性变化通常会先在 ETF 资金与稳定币供给中体现。",
                    "kb_id": "macro-research",
                    "language": "zh",
                    "published_at": "2026-03-01T00:00:00Z",
                    "metadata": {"author": "Desk"},
                },
            },
        )
        self.assertEqual(create_resp.status_code, 200)
        create_payload = create_resp.json()
        self.assertTrue(create_payload["success"])
        self.assertGreaterEqual(create_payload["inserted_chunks"], 1)
        self.assertEqual(create_payload["document"]["title"], "BTC Liquidity Framework")

        list_resp = self.client.get("/v1/knowledge/documents")
        self.assertEqual(list_resp.status_code, 200)
        list_payload = list_resp.json()
        self.assertTrue(list_payload["items"])
        self.assertEqual(list_payload["items"][0]["kb_id"], "macro-research")

    def test_knowledge_document_delete(self) -> None:
        create_resp = self.client.post(
            "/v1/knowledge/documents",
            json={
                "user_id": "u-kb-2",
                "document": {
                    "title": "ETH Upgrade Note",
                    "source": "internal",
                    "doc_type": "event_note",
                    "symbols": ["ETH"],
                    "tags": ["upgrade"],
                    "text": "ETH 升级窗口会影响短期风险偏好与链上活跃度。",
                },
            },
        )
        self.assertEqual(create_resp.status_code, 200)
        doc_id = create_resp.json()["document"]["doc_id"]

        delete_resp = self.client.delete(f"/v1/knowledge/documents/{doc_id}")
        self.assertEqual(delete_resp.status_code, 200)
        self.assertEqual(delete_resp.json()["status"], "deleted")

        get_resp = self.client.get(f"/v1/knowledge/documents/{doc_id}")
        self.assertEqual(get_resp.status_code, 404)

    def test_research_query_returns_500_when_llm_unavailable(self) -> None:
        runtime = self._app.state.runtime
        original_llm = runtime.report_agent.llm
        runtime.report_agent.llm = self._FailingLLM()
        try:
            resp = self.client.post(
                "/v1/research/query",
                json={
                    "user_id": "u-test-500",
                    "query": "请分析 BTC 和 ETH 的短线风险",
                    "task_context": {"symbols": ["BTC", "ETH"]},
                },
            )
            self.assertEqual(resp.status_code, 500)
        finally:
            runtime.report_agent.llm = original_llm


if __name__ == "__main__":
    unittest.main()
