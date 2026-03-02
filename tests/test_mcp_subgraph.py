"""MCP 子图规则与判停测试。"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from typing import Any

from mcp import types

from app.agents.llm.base import BaseLLMClient
from app.graph.mcp_subgraph import MCPSignalSubgraphRunner
from app.models.schemas import RawSignal, SignalType
from app.tools.mcp_gateway import MCPExecutionResult


class _DummyLLM(BaseLLMClient):
    def generate(self, system_prompt: str, user_prompt: str, metadata=None) -> str:
        return '{"calls":[]}'


class _DummyGateway:
    def discover_tools(self):
        return {}, []

    def execute_calls(self, *, task_id: str, symbols: list[str], calls: list[dict[str, Any]]) -> MCPExecutionResult:
        return MCPExecutionResult(rows=[], successes=[], failures=[], errors=[])

    def normalize_rows_to_signals(self, *, task_id: str, rows: list[dict[str, Any]]) -> list[RawSignal]:
        signals: list[RawSignal] = []
        for row in rows:
            signals.append(
                RawSignal(
                    symbol=str(row.get("symbol", "BTC")),
                    source=str(row.get("source", "mcp:test")),
                    signal_type=SignalType(str(row.get("signal_type", SignalType.NEWS.value))),
                    value=row.get("value", {}),
                    raw_ref=str(row.get("raw_ref", "mcp://test/tool")),
                    published_at=datetime.now(timezone.utc),
                    metadata={},
                )
            )
        return signals


class MCPSubgraphRuleTestCase(unittest.TestCase):
    def test_apply_rules_respects_priority_and_injects_defaults(self) -> None:
        raw_calls = [
            {"server": "s1", "tool_name": "tool_banned", "arguments": {"market": "spot"}, "reason": "1"},
            {"server": "s1", "tool_name": "tool_sig", "arguments": {"market": "spot"}, "reason": "2"},
            {"server": "s1", "tool_name": "tool_ok", "arguments": {"market": "spot"}, "reason": "3"},
        ]
        sig_banned = MCPSignalSubgraphRunner.build_call_signature(
            server="s1",
            tool_name="tool_sig",
            arguments={"market": "spot", "page": 1},
        )
        rules = [
            {"type": "tool_ban", "server": "s1", "tool_name": "tool_banned"},
            {"type": "call_signature_ban", "server": "s1", "tool_name": "tool_sig", "call_signature": sig_banned},
        ]
        tool_schemas = {
            "s1": {
                "tool_banned": {"type": "object", "properties": {"market": {"type": "string"}}, "required": ["market"]},
                "tool_sig": {
                    "type": "object",
                    "properties": {
                        "market": {"type": "string"},
                        "page": {"type": "integer", "default": 1},
                    },
                    "required": ["market", "page"],
                },
                "tool_ok": {
                    "type": "object",
                    "properties": {
                        "market": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["market", "limit"],
                },
            }
        }

        result = MCPSignalSubgraphRunner.mcp_apply_rules_pure(
            raw_plan_calls=raw_calls,
            rules=rules,
            tool_schemas=tool_schemas,
        )

        self.assertEqual(len(result["filtered_plan_calls"]), 1)
        self.assertEqual(result["filtered_plan_calls"][0]["tool_name"], "tool_ok")
        self.assertEqual(result["filtered_plan_calls"][0]["arguments"]["limit"], 50)
        self.assertEqual(len(result["filtered_out_calls"]), 2)
        reasons = {item["reason"] for item in result["filtered_out_calls"]}
        self.assertEqual(reasons, {"tool_ban", "call_signature_ban"})

    def test_raw_plan_hash_is_canonical_and_ignores_reason(self) -> None:
        plan_a = [
            {
                "server": "s1",
                "tool_name": "tool",
                "arguments": {"b": 2, "a": 1},
                "reason": "first",
            }
        ]
        plan_b = [
            {
                "server": "s1",
                "tool_name": "tool",
                "arguments": {"a": 1, "b": 2},
                "reason": "second",
            }
        ]
        self.assertEqual(
            MCPSignalSubgraphRunner.build_raw_plan_hash(plan_a),
            MCPSignalSubgraphRunner.build_raw_plan_hash(plan_b),
        )

    def test_signal_hash_excludes_ref_and_publish_time(self) -> None:
        signal_a = {
            "symbol": "BTC",
            "source": "mcp:test",
            "signal_type": "news",
            "value": {"title": "ETF"},
            "raw_ref": "https://a.example",
            "published_at": "2026-03-02T00:00:00Z",
            "internal_id": "id-a",
        }
        signal_b = {
            "symbol": "BTC",
            "source": "mcp:test",
            "signal_type": "news",
            "value": {"title": "ETF"},
            "raw_ref": "https://b.example",
            "published_at": "2026-03-03T00:00:00Z",
            "internal_id": "id-b",
        }
        self.assertEqual(
            MCPSignalSubgraphRunner.build_signal_hash(signal_a),
            MCPSignalSubgraphRunner.build_signal_hash(signal_b),
        )

    def test_should_continue_hits_repeated_blind_plan(self) -> None:
        runner = MCPSignalSubgraphRunner(llm_client=_DummyLLM(), mcp_gateway=_DummyGateway(), max_rounds=4)
        decision = runner.mcp_should_continue(
            {
                "mcp_round": 2,
                "mcp_max_rounds": 4,
                "mcp_admissible_calls_count": 0,
                "mcp_round_successes": [],
                "mcp_failure_classes": {
                    "deterministic_failures_exist": False,
                    "transient_failures_exist": False,
                },
                "mcp_new_rules_added": False,
                "mcp_new_unique_signal_count": 0,
                "mcp_no_progress_streak": 1,
                "transient_grace_used": False,
                "mcp_plan_round_stats": [
                    {"raw_plan_hash": "same", "filtered_plan_count": 0},
                    {"raw_plan_hash": "same", "filtered_plan_count": 0},
                ],
                "mcp_raw_plan_hash_history": ["same", "same"],
            }
        )
        self.assertFalse(decision["mcp_should_continue"])
        self.assertEqual(decision["mcp_termination_reason"], "repeated_blind_plan")

    def test_prepare_discovers_tools_catalog(self) -> None:
        class _GatewayWithTools(_DummyGateway):
            def discover_tools(self):
                return {
                    "srv": [
                        types.Tool(
                            name="get_news",
                            description="fetch news",
                            inputSchema={
                                "type": "object",
                                "properties": {"limit": {"type": "integer", "default": 10}},
                                "required": ["limit"],
                            },
                        )
                    ]
                }, []

        runner = MCPSignalSubgraphRunner(llm_client=_DummyLLM(), mcp_gateway=_GatewayWithTools(), max_rounds=4)
        prepared = runner.mcp_prepare({"errors": [], "raw_signals": []})
        self.assertEqual(prepared["mcp_tool_catalog"]["srv"][0]["name"], "get_news")
        self.assertIn("limit", prepared["mcp_tool_schemas"]["srv"]["get_news"]["properties"])


if __name__ == "__main__":
    unittest.main()
