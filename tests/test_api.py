"""API 基础联调测试。"""

from __future__ import annotations

import os
import unittest
from importlib import reload

from fastapi.testclient import TestClient


class APITestCase(unittest.TestCase):
    """覆盖核心接口的烟雾测试。"""

    @classmethod
    def setUpClass(cls) -> None:
        # 测试环境关闭外部依赖，保证本地可重复执行。
        os.environ["MILVUS_ENABLED"] = "false"
        os.environ["MEM0_ENABLED"] = "false"
        os.environ["MCP_SERVERS"] = ""
        os.environ["MINIMAX_API_KEY"] = ""
        os.environ["ZHIPUAI_API_KEY"] = ""

        import app.config.settings as settings_module

        reload(settings_module)

        from app.main import create_app

        cls._app = create_app()
        cls._client_cm = TestClient(cls._app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
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


if __name__ == "__main__":
    unittest.main()
