"""Workflow symbol 路由策略测试。"""

from __future__ import annotations

import unittest

from app.graph.workflow import ResearchGraphRunner


class _DummyMemoryService:
    def save_task_context(self, user_id: str, task_context):  # pragma: no cover - not used in this suite
        del user_id, task_context

    def load_memory_profile(self, user_id: str):  # pragma: no cover - not used in this suite
        del user_id
        return {}

    def persist_report_memory(self, user_id: str, query: str, report: str):  # pragma: no cover - not used in this suite
        del user_id, query, report


class _DummyMCPSubgraph:
    async def arun(self, **kwargs):  # pragma: no cover - not used in this suite
        del kwargs
        return {"raw_signals": [], "errors": []}


class _DummyResearchService:
    def normalize_signals(self, task_id: str, raw_signals):  # pragma: no cover - not used in this suite
        del task_id, raw_signals
        return []

    def ingest_signals(self, signals):  # pragma: no cover - not used in this suite
        del signals
        return 0

    def retrieve(self, query: str, symbols, top_k: int = 8):  # pragma: no cover - not used in this suite
        del query, symbols, top_k
        return []


class _DummyReportAgent:
    def generate(self, payload):  # pragma: no cover - not used in this suite
        del payload
        raise RuntimeError("not used")


class WorkflowSymbolResolutionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = ResearchGraphRunner(
            memory_service=_DummyMemoryService(),
            mcp_subgraph=_DummyMCPSubgraph(),
            research_service=_DummyResearchService(),
            report_agent=_DummyReportAgent(),
        )

    def test_task_context_symbols_take_precedence(self) -> None:
        state = {
            "query": "请分析 BTC 和 ETH",
            "task_context": {"symbols": ["XRP", "SOL"]},
            "memory_profile": {"watchlist": ["DOGE"]},
            "workflow_steps": [],
        }
        output = self.runner.resolve_symbols(state)

        self.assertEqual(output["hard_symbols"], ["XRP", "SOL"])
        self.assertEqual(output["soft_symbols"], ["XRP", "SOL", "DOGE"])
        self.assertEqual(output["symbols"], ["XRP", "SOL"])

    def test_query_symbols_are_case_insensitive_and_filter_stopwords(self) -> None:
        state = {
            "query": "请分析 btc、eth 和 ETF 影响",
            "task_context": {},
            "memory_profile": {"watchlist": ["DOGE"]},
            "workflow_steps": [],
        }
        output = self.runner.resolve_symbols(state)

        self.assertEqual(output["hard_symbols"], ["BTC", "ETH"])
        self.assertEqual(output["soft_symbols"], ["BTC", "ETH", "DOGE"])

    def test_watchlist_does_not_expand_hard_symbols(self) -> None:
        state = {
            "query": "给我一份宏观风险总结",
            "task_context": {},
            "memory_profile": {"watchlist": ["BTC", "ETH"]},
            "workflow_steps": [],
        }
        output = self.runner.resolve_symbols(state)

        self.assertEqual(output["hard_symbols"], [])
        self.assertEqual(output["soft_symbols"], ["BTC", "ETH"])
        self.assertEqual(output["symbols"], [])


if __name__ == "__main__":
    unittest.main()
