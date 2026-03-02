"""API 基础联调测试。"""

from __future__ import annotations

import os
import unittest
from importlib import reload
from types import MethodType

from fastapi.testclient import TestClient


class APITestCase(unittest.TestCase):
    """覆盖核心接口的烟雾测试。"""

    @classmethod
    def setUpClass(cls) -> None:
        # 测试环境关闭外部依赖，保证本地可重复执行。
        os.environ["MILVUS_ENABLED"] = "false"
        os.environ["MEM0_ENABLED"] = "false"
        os.environ["MCP_SERVERS"] = ""
        os.environ["LLM_PROVIDER"] = "minimax"
        os.environ["MINIMAX_API_KEY"] = "test-minimax-key"
        os.environ["ZHIPUAI_API_KEY"] = ""

        import app.config.settings as settings_module

        reload(settings_module)

        from app.main import create_app

        cls._app = create_app()
        cls._client_cm = TestClient(cls._app)
        cls.client = cls._client_cm.__enter__()

        runtime = cls._app.state.runtime
        cls._original_llm_generate = runtime.report_agent.llm_client.generate
        runtime.report_agent.llm_client.generate = MethodType(
            lambda _self, system_prompt, user_prompt, metadata=None: "测试报告正文",
            runtime.report_agent.llm_client,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        runtime = cls._app.state.runtime
        runtime.report_agent.llm_client.generate = cls._original_llm_generate
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

    def test_research_ingest(self) -> None:
        resp = self.client.post(
            "/v1/research/ingest",
            json={
                "user_id": "u-test-3",
                "documents": [
                    {
                        "doc_id": "doc-1",
                        "symbol": "BTC",
                        "source": "manual-mcp",
                        "published_at": "2026-03-01T00:00:00Z",
                        "text": "BTC funding rate raised and open interest climbed.",
                        "metadata": {"raw_ref": "https://example.com/doc-1"},
                    }
                ],
            },
        )
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertTrue(payload["success"])
        self.assertGreaterEqual(payload["inserted_chunks"], 1)

    def test_research_query_returns_500_when_llm_unavailable(self) -> None:
        runtime = self._app.state.runtime
        original_generate = runtime.report_agent.llm_client.generate
        runtime.report_agent.llm_client.generate = MethodType(
            lambda _self, system_prompt, user_prompt, metadata=None: (_ for _ in ()).throw(
                RuntimeError("llm unavailable")
            ),
            runtime.report_agent.llm_client,
        )
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
            runtime.report_agent.llm_client.generate = original_generate


if __name__ == "__main__":
    unittest.main()
