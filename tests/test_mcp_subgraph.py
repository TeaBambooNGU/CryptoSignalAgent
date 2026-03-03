"""MCP Agent 执行器测试。"""

from __future__ import annotations

import asyncio
import unittest
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from app.graph.mcp_subgraph import MCPSignalSubgraphRunner


class _DummyLLM:
    def invoke(self, messages):
        del messages
        return AIMessage(content="{}")


class _FakeTool:
    def __init__(self, name: str, description: str, schema: dict[str, Any]) -> None:
        self.name = name
        self.description = description
        self.args_schema = schema


class _FakeMCPClient:
    def __init__(self, connections: dict[str, dict[str, Any]], tool_name_prefix: bool = False) -> None:
        self.connections = connections
        self.tool_name_prefix = tool_name_prefix

    async def get_tools(self):
        prefix = "srv_" if self.tool_name_prefix else ""
        return [
            _FakeTool(
                name=f"{prefix}get_news",
                description="fetch news",
                schema={
                    "type": "object",
                    "properties": {"limit": {"type": "integer", "default": 10}},
                    "required": ["limit"],
                },
            )
        ]


class _FakeAgent:
    def __init__(self, messages: list[Any], *, sync_supported: bool = True) -> None:
        self._messages = messages
        self._sync_supported = sync_supported

    def invoke(self, payload: dict[str, Any]):
        del payload
        if not self._sync_supported:
            raise NotImplementedError("StructuredTool does not support sync invocation.")
        return {"messages": self._messages}

    async def ainvoke(self, payload: dict[str, Any]):
        del payload
        return {"messages": self._messages}


class MCPAgentRunnerTestCase(unittest.TestCase):
    def test_build_connections_from_settings(self) -> None:
        settings_servers = (
            {
                "name": "s1",
                "transport": "streamable_http",
                "url": "http://localhost:3000/mcp",
                "headers": {"Authorization": "Bearer x"},
            },
            {
                "name": "s2",
                "transport": "stdio",
                "command": "python",
                "args": ["server.py"],
                "env": {"A": "B"},
                "cwd": "/tmp",
            },
        )
        connections = MCPSignalSubgraphRunner.build_connections_from_settings(settings_servers)
        self.assertEqual(connections["s1"]["transport"], "streamable_http")
        self.assertEqual(connections["s1"]["url"], "http://localhost:3000/mcp")
        self.assertEqual(connections["s2"]["command"], "python")
        self.assertEqual(connections["s2"]["args"], ["server.py"])

    def test_run_collects_rows_from_tool_messages_and_payload(self) -> None:
        messages = [
            ToolMessage(name="srv_get_news", content='{"symbol":"BTC","title":"ETF"}', tool_call_id="call-1"),
            AIMessage(
                content=(
                    '{"raw_signals":[{"symbol":"ETH","source":"mcp:srv","signal_type":"price",'
                    '"value":{"price":3000},"raw_ref":"https://example.com"}],"errors":["agent-warning"]}'
                )
            ),
        ]

        runner = MCPSignalSubgraphRunner(
            llm=_DummyLLM(),
            mcp_connections={"srv": {"transport": "streamable_http", "url": "http://localhost:3000/mcp"}},
            mcp_client_factory=_FakeMCPClient,
            agent_factory=lambda **kwargs: _FakeAgent(messages),
        )

        result = asyncio.run(
            runner.arun(
                user_id="u1",
                query="分析 BTC ETH",
                task_id="task-1",
                symbols=["BTC", "ETH"],
                errors=[],
            )
        )

        self.assertEqual(result["mcp_tools_count"], 1)
        self.assertEqual(result["mcp_termination_reason"], "agent_completed")
        self.assertIn("agent-warning", result["errors"])
        self.assertGreaterEqual(len(result["raw_signals"]), 2)
        symbols = {item["symbol"] for item in result["raw_signals"]}
        self.assertIn("BTC", symbols)
        self.assertIn("ETH", symbols)

    def test_run_returns_parse_error_when_no_structured_output(self) -> None:
        messages = [AIMessage(content="采集完成")]
        runner = MCPSignalSubgraphRunner(
            llm=_DummyLLM(),
            mcp_connections={"srv": {"transport": "streamable_http", "url": "http://localhost:3000/mcp"}},
            mcp_client_factory=_FakeMCPClient,
            agent_factory=lambda **kwargs: _FakeAgent(messages),
        )

        result = asyncio.run(
            runner.arun(
                user_id="u1",
                query="分析 BTC",
                task_id="task-2",
                symbols=["BTC"],
                errors=[],
            )
        )

        self.assertEqual(result["raw_signals"], [])
        self.assertTrue(any("Agent 结论解析失败" in item for item in result["errors"]))

    def test_run_handles_missing_servers(self) -> None:
        runner = MCPSignalSubgraphRunner(
            llm=_DummyLLM(),
            mcp_connections={},
            mcp_client_factory=_FakeMCPClient,
            agent_factory=lambda **kwargs: _FakeAgent([]),
        )

        result = asyncio.run(
            runner.arun(
                user_id="u1",
                query="分析 BTC",
                task_id="task-3",
                symbols=["BTC"],
                errors=[],
            )
        )

        self.assertEqual(result["raw_signals"], [])
        self.assertEqual(result["mcp_tools_count"], 0)
        self.assertEqual(result["mcp_termination_reason"], "no_tools")
        self.assertTrue(any("未配置 MCP_SERVERS" in item for item in result["errors"]))

    def test_run_uses_async_agent_when_sync_not_supported(self) -> None:
        messages = [
            ToolMessage(name="srv_get_news", content='{"symbol":"BTC","title":"ETF"}', tool_call_id="call-1"),
            AIMessage(content='{"raw_signals":[],"errors":[]}'),
        ]
        runner = MCPSignalSubgraphRunner(
            llm=_DummyLLM(),
            mcp_connections={"srv": {"transport": "streamable_http", "url": "http://localhost:3000/mcp"}},
            mcp_client_factory=_FakeMCPClient,
            agent_factory=lambda **kwargs: _FakeAgent(messages, sync_supported=False),
        )

        result = asyncio.run(
            runner.arun(
                user_id="u1",
                query="分析 BTC",
                task_id="task-4",
                symbols=["BTC"],
                errors=[],
            )
        )

        self.assertEqual(result["mcp_termination_reason"], "agent_completed")
        self.assertGreaterEqual(len(result["raw_signals"]), 1)


if __name__ == "__main__":
    unittest.main()
