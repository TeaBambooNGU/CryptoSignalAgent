"""LLM 客户端工厂测试。"""

from __future__ import annotations

import unittest

from app.agents.llm import create_llm_client
from app.agents.llm.langchain_client import LangChainLLMClient
from app.config.settings import Settings


class LLMFactoryTestCase(unittest.TestCase):
    def test_create_minimax_client_success(self) -> None:
        settings = Settings(
            llm_provider="minimax",
            minimax_api_key="test-key",
        )
        client = create_llm_client(settings)
        self.assertIsInstance(client, LangChainLLMClient)
        self.assertEqual(client.provider_name, "minimax")
        self.assertEqual(client.llm.__class__.__name__, "ChatOpenAI")

    def test_create_minimax_client_requires_credentials(self) -> None:
        settings = Settings(
            llm_provider="minimax",
            minimax_api_key="",
        )
        with self.assertRaises(ValueError):
            create_llm_client(settings)

    def test_create_openai_client_success(self) -> None:
        settings = Settings(
            llm_provider="openai",
            openai_api_key="test-openai-key",
            openai_base_url="https://api.openai.com/v1",
        )
        client = create_llm_client(settings)
        self.assertIsInstance(client, LangChainLLMClient)

    def test_create_openai_client_requires_api_key(self) -> None:
        settings = Settings(
            llm_provider="openai",
            openai_api_key="",
        )
        with self.assertRaises(ValueError):
            create_llm_client(settings)

    def test_unknown_provider_raises(self) -> None:
        settings = Settings(llm_provider="unknown-provider")
        with self.assertRaises(ValueError):
            create_llm_client(settings)


if __name__ == "__main__":
    unittest.main()
