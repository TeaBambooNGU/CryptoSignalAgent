"""ReportAgent 行为测试。"""

from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage

from app.agents.report_agent import ReportAgent
from app.config.settings import Settings
from app.models.schemas import ReportGenerationInput, RetrievedChunk


class _CaptureLLM:
    def __init__(self) -> None:
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return AIMessage(content="报告草稿")


class ReportAgentTestCase(unittest.TestCase):
    def test_generate_uses_full_docs_in_prompt(self) -> None:
        llm = _CaptureLLM()
        agent = ReportAgent(settings=Settings(), llm=llm)

        long_text = "A" * 220
        docs = [
            RetrievedChunk(
                chunk_id=f"c-{idx}",
                doc_id=f"d-{idx}",
                symbol="BTC",
                source="mcp-news",
                text=f"doc-{idx} {long_text}",
                score=0.9 - idx * 0.01,
            )
            for idx in range(1, 10)
        ]

        payload = ReportGenerationInput(
            user_id="u1",
            query="分析 BTC",
            task_id="task-1",
            memory_profile={},
            signals=[],
            retrieved_docs=docs,
        )

        agent.generate(payload)

        self.assertIsNotNone(llm.messages)
        user_prompt = llm.messages[1].content
        self.assertIn("证据摘要", user_prompt)
        self.assertIn("9. [mcp-news/BTC]", user_prompt)
        self.assertIn(f"doc-1 {long_text}", user_prompt)
        self.assertNotIn("A" * 180 + "...", user_prompt)
        reinforcement_signal = "【强化信号】请严格遵守：先给结论和置信度；每个结论绑定证据；明确风险与反例；保持中文专业研报风格。"
        query_reinforcement = "【用户问题强化】分析 BTC"
        self.assertEqual(user_prompt.count(reinforcement_signal), 2)
        self.assertTrue(user_prompt.startswith(reinforcement_signal))
        self.assertTrue(user_prompt.strip().endswith(query_reinforcement))


if __name__ == "__main__":
    unittest.main()
