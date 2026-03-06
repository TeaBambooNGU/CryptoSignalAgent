"""ReportAgent 行为测试。"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone

from langchain_core.messages import AIMessage

from app.agents.report_agent import ReportAgent
from app.config.settings import Settings
from app.models.schemas import NormalizedSignal, ReportGenerationInput, RetrievedChunk, SignalType


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
            knowledge_docs=docs,
        )

        agent.generate(payload)

        self.assertIsNotNone(llm.messages)
        user_prompt = llm.messages[1].content
        self.assertIn("知识证据摘要", user_prompt)
        self.assertIn("9. [mcp-news/BTC]", user_prompt)
        self.assertIn(f"doc-1 {long_text}", user_prompt)
        self.assertNotIn("A" * 180 + "...", user_prompt)
        reinforcement_signal = "【强化信号】请严格遵守：先给结论和置信度；每个结论绑定证据；明确风险与反例；保持中文专业研报风格。"
        query_reinforcement = "【用户问题强化】分析 BTC"
        self.assertEqual(user_prompt.count(reinforcement_signal), 2)
        self.assertTrue(user_prompt.startswith(reinforcement_signal))
        self.assertTrue(user_prompt.strip().endswith(query_reinforcement))

    def test_generate_includes_signal_details_value_and_raw_ref(self) -> None:
        llm = _CaptureLLM()
        agent = ReportAgent(settings=Settings(), llm=llm)

        payload = ReportGenerationInput(
            user_id="u1",
            query="给我 BTC 的研报",
            task_id="task-1",
            memory_profile={},
            signals=[
                NormalizedSignal(
                    timestamp=datetime(2026, 3, 7, 10, 0, tzinfo=timezone.utc),
                    symbol="BTC",
                    source="mcp:coingecko",
                    signal_type=SignalType.ONCHAIN,
                    value={"metric": "active_addresses", "value": 1024000},
                    confidence=0.91,
                    raw_ref="https://example.com/btc/onchain",
                    task_id="task-1",
                )
            ],
            knowledge_docs=[],
        )

        agent.generate(payload)

        self.assertIsNotNone(llm.messages)
        user_prompt = llm.messages[1].content
        self.assertIn("实时信号明细", user_prompt)
        self.assertIn("symbol=BTC", user_prompt)
        self.assertIn("raw_ref=https://example.com/btc/onchain", user_prompt)
        self.assertIn('"metric": "active_addresses"', user_prompt)
        self.assertIn('"value": 1024000', user_prompt)

    def test_signal_value_truncation_limit_from_settings(self) -> None:
        llm = _CaptureLLM()
        agent = ReportAgent(
            settings=Settings(report_signal_value_max_chars=16),
            llm=llm,
        )

        payload = ReportGenerationInput(
            user_id="u1",
            query="分析 BTC",
            task_id="task-1",
            memory_profile={},
            signals=[
                NormalizedSignal(
                    timestamp=datetime(2026, 3, 7, 10, 0, tzinfo=timezone.utc),
                    symbol="BTC",
                    source="mcp:news",
                    signal_type=SignalType.NEWS,
                    value="x" * 40,
                    confidence=0.88,
                    raw_ref="ref-news",
                    task_id="task-1",
                )
            ],
            knowledge_docs=[],
        )

        agent.generate(payload)

        user_prompt = llm.messages[1].content
        self.assertIn("value=xxxxxxxxxxxxxxxx...(truncated)", user_prompt)

    def test_signal_details_keep_all_when_within_limit(self) -> None:
        llm = _CaptureLLM()
        agent = ReportAgent(settings=Settings(report_signal_detail_limit=5), llm=llm)

        payload = ReportGenerationInput(
            user_id="u1",
            query="分析市场",
            task_id="task-1",
            memory_profile={},
            signals=[
                self._make_signal(symbol="BTC", confidence=0.90, raw_ref="ref-btc"),
                self._make_signal(symbol="ETH", confidence=0.80, raw_ref="ref-eth"),
                self._make_signal(symbol="SOL", confidence=0.70, raw_ref="ref-sol"),
            ],
            knowledge_docs=[],
        )

        agent.generate(payload)

        user_prompt = llm.messages[1].content
        self.assertIn("- 共 3 条，全部已展示", user_prompt)
        self.assertIn("raw_ref=ref-btc", user_prompt)
        self.assertIn("raw_ref=ref-eth", user_prompt)
        self.assertIn("raw_ref=ref-sol", user_prompt)

    def test_signal_details_trim_only_when_exceed_limit(self) -> None:
        llm = _CaptureLLM()
        agent = ReportAgent(settings=Settings(report_signal_detail_limit=2), llm=llm)

        payload = ReportGenerationInput(
            user_id="u1",
            query="给我 BTC 的研报",
            task_id="task-1",
            memory_profile={},
            signals=[
                self._make_signal(symbol="BTC", confidence=0.20, raw_ref="ref-btc"),
                self._make_signal(symbol="ETH", confidence=0.95, raw_ref="ref-eth"),
                self._make_signal(symbol="ADA", confidence=0.85, raw_ref="ref-ada"),
                self._make_signal(symbol="XRP", confidence=0.75, raw_ref="ref-xrp"),
            ],
            knowledge_docs=[],
        )

        agent.generate(payload)

        user_prompt = llm.messages[1].content
        self.assertIn("超过上限 2", user_prompt)
        self.assertIn("焦点标的: BTC", user_prompt)
        self.assertIn("raw_ref=ref-btc", user_prompt)
        self.assertIn("raw_ref=ref-eth", user_prompt)
        self.assertNotIn("raw_ref=ref-ada", user_prompt)
        self.assertNotIn("raw_ref=ref-xrp", user_prompt)

    def test_settings_from_env_reads_report_signal_detail_limit(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as handle:
            handle.write(
                "REPORT_SIGNAL_DETAIL_LIMIT=7\n"
                "REPORT_SIGNAL_VALUE_MAX_CHARS=333\n"
            )
            env_path = handle.name

        settings = Settings.from_env(env_path)

        self.assertEqual(settings.report_signal_detail_limit, 7)
        self.assertEqual(settings.report_signal_value_max_chars, 333)

    @staticmethod
    def _make_signal(symbol: str, confidence: float, raw_ref: str) -> NormalizedSignal:
        return NormalizedSignal(
            timestamp=datetime(2026, 3, 7, 10, 0, tzinfo=timezone.utc),
            symbol=symbol,
            source="mcp:test",
            signal_type=SignalType.NEWS,
            value={"headline": f"{symbol} signal", "score": confidence},
            confidence=confidence,
            raw_ref=raw_ref,
            task_id="task-1",
        )


if __name__ == "__main__":
    unittest.main()
