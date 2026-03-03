"""Mem0 参数兼容性测试。"""

from __future__ import annotations

import os
import unittest
from typing import Any
from unittest.mock import patch

from app.config.settings import Settings
from app.memory.mem0_service import MemoryService


class _DummyMilvusStore:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def upsert_user_memory(self, records: list[dict[str, Any]]) -> None:
        self.records.extend(records)
        return None

    def query_user_memory(self, user_id: str, limit: int) -> list[dict[str, Any]]:
        rows = [item for item in self.records if item.get("user_id") == user_id]
        rows.sort(key=lambda item: int(item.get("updated_at", 0)), reverse=True)
        return rows[:limit]


class _SearchClient:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] = {}

    def search(self, **kwargs: Any) -> dict[str, Any]:
        self.last_kwargs = kwargs
        return {"results": [{"id": "m1"}]}


class Mem0ServiceCompatibilityTestCase(unittest.TestCase):
    def test_build_oss_config_reuses_project_milvus(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ZHIPUAI_API_KEY": "zhipu-key",
            },
            clear=False,
        ):
            settings = Settings(
                mem0_enabled=True,
                mem0_mode="oss",
                mem0_oss_collection="mem0_memory",
                milvus_enabled=True,
                milvus_uri="http://127.0.0.1:19530",
                milvus_token="milvus-token",
                milvus_db_name="default",
                vector_dim=384,
                llm_model="MiniMax-M2.5",
                minimax_api_key="minimax-key",
                minimax_api_host="https://api.minimax.chat",
                zhipu_embedding_model="embedding-3",
            )
            service = MemoryService(settings=settings, milvus_store=_DummyMilvusStore())

            config = service._build_mem0_oss_config()

        self.assertEqual(config["vector_store"]["provider"], "milvus")
        self.assertEqual(config["vector_store"]["config"]["url"], "http://127.0.0.1:19530")
        self.assertEqual(config["vector_store"]["config"]["collection_name"], "mem0_memory")
        self.assertEqual(config["vector_store"]["config"]["embedding_model_dims"], 384)
        self.assertEqual(config["vector_store"]["config"]["metric_type"], "COSINE")
        self.assertEqual(config["vector_store"]["config"]["db_name"], "default")
        self.assertEqual(config["llm"]["provider"], "openai")
        self.assertEqual(config["llm"]["config"]["model"], "MiniMax-M2.5")
        self.assertEqual(config["llm"]["config"]["api_key"], "minimax-key")
        self.assertEqual(config["llm"]["config"]["openai_base_url"], "https://api.minimax.chat/v1")
        self.assertEqual(config["embedder"]["provider"], "openai")
        self.assertEqual(config["embedder"]["config"]["model"], "embedding-3")
        self.assertEqual(config["embedder"]["config"]["embedding_dims"], 384)
        self.assertEqual(config["embedder"]["config"]["api_key"], "zhipu-key")
        self.assertEqual(config["embedder"]["config"]["openai_base_url"], "https://open.bigmodel.cn/api/paas/v4")

    def test_search_uses_platform_filters_for_memory_client(self) -> None:
        settings = Settings(mem0_enabled=False, mem0_mode="platform")
        service = MemoryService(settings=settings, milvus_store=_DummyMilvusStore())
        client = _SearchClient()
        service._mem0_client = client

        result = service._mem0_search(user_id="u-1", query="latest preference", limit=5)

        self.assertEqual(result, [{"id": "m1"}])
        self.assertEqual(client.last_kwargs["query"], "latest preference")
        self.assertEqual(client.last_kwargs["top_k"], 5)
        self.assertEqual(client.last_kwargs["filters"], {"user_id": "u-1"})
        self.assertNotIn("user_id", client.last_kwargs)
        self.assertNotIn("limit", client.last_kwargs)

    def test_search_uses_legacy_kwargs_for_oss_memory(self) -> None:
        settings = Settings(mem0_enabled=False, mem0_mode="oss")
        service = MemoryService(settings=settings, milvus_store=_DummyMilvusStore())
        client = _SearchClient()
        service._mem0_client = client

        result = service._mem0_search(user_id="u-2", query="risk profile", limit=3)

        self.assertEqual(result, [{"id": "m1"}])
        self.assertEqual(client.last_kwargs["query"], "risk profile")
        self.assertEqual(client.last_kwargs["user_id"], "u-2")
        self.assertEqual(client.last_kwargs["limit"], 3)
        self.assertNotIn("filters", client.last_kwargs)
        self.assertNotIn("top_k", client.last_kwargs)

    def test_save_tool_correction_written_to_long_term_profile(self) -> None:
        with patch.dict(os.environ, {"ZHIPUAI_API_KEY": ""}, clear=False):
            settings = Settings(mem0_enabled=False)
            milvus = _DummyMilvusStore()
            service = MemoryService(settings=settings, milvus_store=milvus)

            service.save_tool_correction(
                user_id="u-3",
                correction={
                    "server": "defillama",
                    "failed_tool": "protocol_information",
                    "failed_arguments": {"protocol": "BTC"},
                    "fixed_tool": "protocol_information",
                    "fixed_arguments": {"protocol": "bitcoin"},
                },
            )

            profile = service.load_memory_profile("u-3")
            corrections = profile.get("tool_corrections", [])
            self.assertEqual(len(corrections), 1)
            self.assertEqual(corrections[0]["server"], "defillama")
            self.assertEqual(corrections[0]["fixed_arguments"]["protocol"], "bitcoin")


if __name__ == "__main__":
    unittest.main()
