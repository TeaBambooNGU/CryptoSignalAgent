"""Workflow symbol 路由策略测试。"""

from __future__ import annotations

import asyncio
import unittest

from app.graph.workflow import ResearchGraphRunner


class _DummyMemoryService:
    def save_task_context(self, user_id: str, conversation_id: str, task_context):  # pragma: no cover - not used
        del user_id, conversation_id, task_context

    def load_memory_profile(
        self,
        user_id: str,
        conversation_id: str | None = None,
        context_anchor_turn_id: str | None = None,
    ):  # pragma: no cover
        del user_id, conversation_id, context_anchor_turn_id
        return {}

    def persist_report_memory(  # pragma: no cover - not used in this suite
        self,
        user_id: str,
        query: str,
        report: str,
        conversation_id: str | None = None,
        turn_id: str | None = None,
        request_id: str | None = None,
    ):
        del user_id, query, report, conversation_id, turn_id, request_id


class _DummyMCPSubgraph:
    async def arun(self, **kwargs):  # pragma: no cover - not used in this suite
        del kwargs
        return {"raw_signals": [], "errors": []}


class _CaptureMCPSubgraph:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def arun(self, **kwargs):
        self.calls.append(kwargs)
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

    def test_collect_signals_uses_hard_symbols_and_passes_soft_as_hint(self) -> None:
        capture_mcp = _CaptureMCPSubgraph()
        runner = ResearchGraphRunner(
            memory_service=_DummyMemoryService(),
            mcp_subgraph=capture_mcp,
            research_service=_DummyResearchService(),
            report_agent=_DummyReportAgent(),
        )
        state = {
            "user_id": "u1",
            "query": "请分析 BTC",
            "task_id": "task-1",
            "symbols": ["BTC"],
            "hard_symbols": ["BTC"],
            "soft_symbols": ["BTC", "ETH"],
            "errors": [],
            "workflow_steps": [],
        }

        asyncio.run(runner.collect_signals_via_mcp(state))

        self.assertEqual(len(capture_mcp.calls), 1)
        call = capture_mcp.calls[0]
        self.assertEqual(call["symbols"], ["BTC"])
        self.assertEqual(call["hint_symbols"], ["BTC", "ETH"])


if __name__ == "__main__":
    unittest.main()
