"""Mem0 参数兼容性测试。"""

from __future__ import annotations

import os
import unittest
from typing import Any
from unittest.mock import patch

from app.config.settings import Settings
from app.memory.mem0_service import MemoryService
from app.memory.session_store import InMemorySessionMemoryStore


class _DummyMilvusStore:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def upsert_user_memory(self, records: list[dict[str, Any]]) -> None:
        for incoming in records:
            incoming_id = str(incoming.get("id", ""))
            self.records = [row for row in self.records if str(row.get("id", "")) != incoming_id]
            self.records.append(dict(incoming))
        return None

    def query_user_memory(self, user_id: str, limit: int) -> list[dict[str, Any]]:
        rows = [item for item in self.records if item.get("user_id") == user_id]
        rows.sort(key=lambda item: int(item.get("updated_at", 0)), reverse=True)
        return rows[:limit]

    def delete_user_memory_by_ids(self, ids: list[str]) -> int:
        id_set = {str(item).strip() for item in ids if str(item).strip()}
        before = len(self.records)
        self.records = [row for row in self.records if str(row.get("id", "")) not in id_set]
        return before - len(self.records)


class _SearchClient:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] = {}

    def search(self, **kwargs: Any) -> dict[str, Any]:
        self.last_kwargs = kwargs
        return {"results": [{"id": "m1"}]}


class _ExtractorResponse:
    def __init__(self, content: Any) -> None:
        self.content = content


class _PreferenceExtractorClient:
    def __init__(self, content: Any) -> None:
        self.content = content
        self.messages: list[Any] = []

    def invoke(self, messages: list[Any]) -> _ExtractorResponse:
        self.messages = messages
        return _ExtractorResponse(self.content)


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
            service = MemoryService(
                settings=settings,
                milvus_store=_DummyMilvusStore(),
                session_store=InMemorySessionMemoryStore(),
            )

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
        service = MemoryService(
            settings=settings,
            milvus_store=_DummyMilvusStore(),
            session_store=InMemorySessionMemoryStore(),
        )
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
        service = MemoryService(
            settings=settings,
            milvus_store=_DummyMilvusStore(),
            session_store=InMemorySessionMemoryStore(),
        )
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
            service = MemoryService(
                settings=settings,
                milvus_store=milvus,
                session_store=InMemorySessionMemoryStore(),
            )

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

    def test_session_memory_isolation_by_conversation_id(self) -> None:
        settings = Settings(mem0_enabled=False)
        service = MemoryService(
            settings=settings,
            milvus_store=_DummyMilvusStore(),
            session_store=InMemorySessionMemoryStore(),
        )

        service.save_task_context(user_id="u-4", conversation_id="conv-a", task_context={"symbols": ["BTC"]})
        service.save_task_context(user_id="u-4", conversation_id="conv-b", task_context={"symbols": ["ETH"]})

        profile_a = service.load_memory_profile(user_id="u-4", conversation_id="conv-a")
        profile_b = service.load_memory_profile(user_id="u-4", conversation_id="conv-b")

        self.assertEqual(len(profile_a["session_memory"]), 1)
        self.assertEqual(len(profile_b["session_memory"]), 1)
        self.assertEqual(profile_a["session_memory"][0]["content"]["symbols"], ["BTC"])
        self.assertEqual(profile_b["session_memory"][0]["content"]["symbols"], ["ETH"])

    def test_persist_report_memory_uses_llm_extractor(self) -> None:
        with patch.dict(os.environ, {"ZHIPUAI_API_KEY": ""}, clear=False):
            settings = Settings(mem0_enabled=False)
            milvus = _DummyMilvusStore()
            service = MemoryService(
                settings=settings,
                milvus_store=milvus,
                session_store=InMemorySessionMemoryStore(),
            )
            service._preference_extractor_client = _PreferenceExtractorClient(
                """```json
{"watchlist":["btc","ETH","$sol"],"risk_preference":"aggressive","reading_habit":"summary_first","noise":1}
```"""
            )

            service.persist_report_memory(
                user_id="u-5",
                query="请给我 BTC 与 ETH 的日内观点",
                report="本次重点关注 BTC、ETH、SOL。",
            )

            profile = service.load_memory_profile("u-5")
            self.assertEqual(len(profile["long_term_memory"]), 1)
            content = profile["long_term_memory"][0]["content"]
            self.assertEqual(content["watchlist"], ["BTC", "ETH", "SOL"])
            self.assertEqual(content["risk_preference"], "aggressive")
            self.assertEqual(content["reading_habit"], "summary_first")

    def test_persist_report_memory_skips_when_llm_output_invalid(self) -> None:
        with patch.dict(os.environ, {"ZHIPUAI_API_KEY": ""}, clear=False):
            settings = Settings(mem0_enabled=False)
            milvus = _DummyMilvusStore()
            service = MemoryService(
                settings=settings,
                milvus_store=milvus,
                session_store=InMemorySessionMemoryStore(),
            )
            service._preference_extractor_client = _PreferenceExtractorClient("not-json")

            service.persist_report_memory(
                user_id="u-6",
                query="给我一份市场观点",
                report="这是一段没有明确偏好的研报。",
            )

            self.assertEqual(milvus.records, [])

    def test_preference_profile_keeps_single_record_with_merge_rules(self) -> None:
        with patch.dict(os.environ, {"ZHIPUAI_API_KEY": ""}, clear=False):
            settings = Settings(mem0_enabled=False)
            milvus = _DummyMilvusStore()
            service = MemoryService(
                settings=settings,
                milvus_store=milvus,
                session_store=InMemorySessionMemoryStore(),
            )

            service.save_preference(
                user_id="u-7",
                preference={
                    "watchlist": ["BTC", "ETH"],
                    "risk_preference": "balanced",
                    "reading_habit": "summary_first",
                },
            )
            service.save_preference(
                user_id="u-7",
                preference={
                    "watchlist": ["SOL", "BTC"],
                    "risk_preference": "aggressive",
                },
            )

            profile = service.load_memory_profile("u-7", conversation_id="conv-u7")
            self.assertEqual(len(profile["long_term_memory"]), 1)
            self.assertEqual(profile["watchlist"], ["BTC", "ETH", "SOL"])
            self.assertEqual(profile["risk_preference"], "aggressive")
            self.assertEqual(profile["reading_habit"], "summary_first")
            content = profile["long_term_memory"][0]["content"]
            self.assertEqual(content["watchlist"], ["BTC", "ETH", "SOL"])
            self.assertEqual(content["risk_preference"], "aggressive")
            self.assertEqual(content["reading_habit"], "summary_first")


if __name__ == "__main__":
    unittest.main()
