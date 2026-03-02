"""LLM 客户端工厂测试。"""

from __future__ import annotations

import unittest

from app.agents.llm import create_llm_client
from app.agents.llm.openai_compatible import OpenAICompatibleLLMClient
from app.config.settings import Settings


class LLMFactoryTestCase(unittest.TestCase):
    def test_create_minimax_client_success(self) -> None:
        settings = Settings(
            llm_provider="minimax",
            minimax_api_key="test-key",
        )
        client = create_llm_client(settings)
        self.assertIsInstance(client, OpenAICompatibleLLMClient)

    def test_create_minimax_client_requires_api_key(self) -> None:
        settings = Settings(
            llm_provider="minimax",
            minimax_api_key="",
        )
        with self.assertRaises(ValueError):
            create_llm_client(settings)

    def test_create_openai_compatible_requires_full_config(self) -> None:
        settings = Settings(
            llm_provider="openai_compatible",
            openai_compatible_api_key="",
            openai_compatible_base_url="",
        )
        with self.assertRaises(ValueError):
            create_llm_client(settings)

    def test_unknown_provider_raises(self) -> None:
        settings = Settings(llm_provider="unknown-provider")
        with self.assertRaises(ValueError):
            create_llm_client(settings)


if __name__ == "__main__":
    unittest.main()
