"""LLM 客户端工厂测试。"""

from __future__ import annotations

import unittest

from langchain_openai import ChatOpenAI

from app.agents.llm import create_deepseek_client, create_llm_client
from app.config.settings import Settings


class LLMFactoryTestCase(unittest.TestCase):
    def test_create_minimax_client_success(self) -> None:
        settings = Settings(
            llm_provider="minimax",
            minimax_api_key="test-key",
        )
        client = create_llm_client(settings)
        self.assertIsInstance(client, ChatOpenAI)

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
        self.assertIsInstance(client, ChatOpenAI)

    def test_create_openai_client_requires_api_key(self) -> None:
        settings = Settings(
            llm_provider="openai",
            openai_api_key="",
        )
        with self.assertRaises(ValueError):
            create_llm_client(settings)

    def test_create_deepseek_client_success(self) -> None:
        settings = Settings(
            deepseek_api_key="test-deepseek-key",
            deepseek_base_url="https://api.deepseek.com/v1",
        )
        client = create_deepseek_client(
            settings,
            model_name="deepseek-chat",
            timeout_seconds=15,
        )
        self.assertIsInstance(client, ChatOpenAI)

    def test_create_deepseek_client_requires_api_key(self) -> None:
        settings = Settings(deepseek_api_key="")
        with self.assertRaises(ValueError):
            create_deepseek_client(
                settings,
                model_name="deepseek-chat",
                timeout_seconds=15,
            )

    def test_unknown_provider_raises(self) -> None:
        settings = Settings(llm_provider="unknown-provider")
        with self.assertRaises(ValueError):
            create_llm_client(settings)


if __name__ == "__main__":
    unittest.main()
