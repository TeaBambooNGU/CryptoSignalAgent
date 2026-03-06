"""MCP Agent 执行器测试。"""

from __future__ import annotations

import asyncio
import time
import unittest
from types import SimpleNamespace
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

    async def get_tools(self, *, server_name: str | None = None):
        name_prefix = ""
        if self.tool_name_prefix:
            name_prefix = f"{server_name}_" if server_name else "srv_"
        return [
            _FakeTool(
                name=f"{name_prefix}get_news",
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
        settings_servers = {
            "s1": {
                "type": "http",
                "url": "http://localhost:3000/mcp",
                "headers": {"Authorization": "Bearer x"},
            },
            "s2": {
                "type": "stdio",
                "command": "python",
                "args": ["server.py"],
                "env": {"A": "B"},
                "cwd": "/tmp",
            },
        }
        connections = MCPSignalSubgraphRunner.build_connections_from_settings(settings_servers)
        self.assertEqual(connections["s1"]["transport"], "streamable_http")
        self.assertEqual(connections["s1"]["url"], "http://localhost:3000/mcp")
        self.assertEqual(connections["s2"]["command"], "python")
        self.assertEqual(connections["s2"]["args"], ["server.py"])

    def test_build_agent_user_prompt_separates_target_and_hint_symbols(self) -> None:
        runner = MCPSignalSubgraphRunner(
            llm=_DummyLLM(),
            mcp_connections={"srv": {"transport": "streamable_http", "url": "http://localhost:3000/mcp"}},
            mcp_client_factory=_FakeMCPClient,
            agent_factory=lambda **kwargs: _FakeAgent([]),
        )
        prompt = runner._build_agent_user_prompt(
            user_id="u1",
            query="分析 BTC",
            symbols=["BTC"],
            hint_symbols=["BTC", "ETH"],
            task_id="task-0",
            server_name="srv",
            tool_catalog=[],
        )

        self.assertIn("target_symbols=BTC", prompt)
        self.assertIn("hint_symbols=BTC,ETH", prompt)

    def test_build_agent_passes_retry_middlewares(self) -> None:
        captured: dict[str, Any] = {}

        def _agent_factory(**kwargs):
            captured.update(kwargs)
            return _FakeAgent([])

        runner = MCPSignalSubgraphRunner(
            llm=_DummyLLM(),
            mcp_connections={"srv": {"transport": "streamable_http", "url": "http://localhost:3000/mcp"}},
            mcp_client_factory=_FakeMCPClient,
            agent_factory=_agent_factory,
        )

        runner._build_agent(tools=[], agent_name="agent", system_prompt="prompt")

        middleware = captured.get("middleware", [])
        self.assertEqual(len(middleware), 2)
        self.assertEqual(type(middleware[0]).__name__, "MCPToolErrorMiddleware")
        self.assertEqual(type(middleware[1]).__name__, "ToolRetryMiddleware")

    def test_retryable_error_classifier_distinguishes_transient_and_input_errors(self) -> None:
        runner = MCPSignalSubgraphRunner(
            llm=_DummyLLM(),
            mcp_connections={"srv": {"transport": "streamable_http", "url": "http://localhost:3000/mcp"}},
            mcp_client_factory=_FakeMCPClient,
            agent_factory=lambda **kwargs: _FakeAgent([]),
        )

        self.assertTrue(runner._is_retryable_tool_error(TimeoutError("socket timeout")))
        self.assertFalse(runner._is_retryable_tool_error(ValueError("missing required field: symbol")))

    def test_tool_error_middleware_returns_tool_message(self) -> None:
        runner = MCPSignalSubgraphRunner(
            llm=_DummyLLM(),
            mcp_connections={"srv": {"transport": "streamable_http", "url": "http://localhost:3000/mcp"}},
            mcp_client_factory=_FakeMCPClient,
            agent_factory=lambda **kwargs: _FakeAgent([]),
        )
        middleware = runner._build_agent_middleware()[0]
        request = SimpleNamespace(
            tool_call={"id": "call-1", "name": "srv_get_news", "args": {"limit": 10}},
            tool=_FakeTool(name="srv_get_news", description="fetch news", schema={"type": "object", "required": ["limit"]}),
        )

        async def _handler(_request):
            del _request
            raise ValueError("missing required field: symbol")

        message = asyncio.run(middleware.awrap_tool_call(request, _handler))

        self.assertIsInstance(message, ToolMessage)
        self.assertEqual(message.status, "error")
        self.assertEqual(message.tool_call_id, "call-1")
        self.assertIn('"error_type":"invalid_input"', str(message.content))

    def test_retry_middleware_retries_transient_tool_errors(self) -> None:
        runner = MCPSignalSubgraphRunner(
            llm=_DummyLLM(),
            mcp_connections={"srv": {"transport": "streamable_http", "url": "http://localhost:3000/mcp"}},
            mcp_client_factory=_FakeMCPClient,
            agent_factory=lambda **kwargs: _FakeAgent([]),
        )
        retry_middleware = runner._build_agent_middleware()[1]
        request = SimpleNamespace(
            tool_call={"id": "call-2", "name": "srv_get_news", "args": {}},
            tool=_FakeTool(name="srv_get_news", description="fetch news", schema={"type": "object"}),
        )
        attempts = {"count": 0}

        async def _handler(_request):
            del _request
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise TimeoutError("socket timeout")
            return ToolMessage(content='{"ok":true}', tool_call_id="call-2")

        message = asyncio.run(retry_middleware.awrap_tool_call(request, _handler))

        self.assertEqual(attempts["count"], 3)
        self.assertIsInstance(message, ToolMessage)
        self.assertEqual(str(message.content), '{"ok":true}')

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
        self.assertTrue(any("未配置 MCP Servers" in item for item in result["errors"]))

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

    def test_discover_tools_tolerates_single_server_failure(self) -> None:
        class _PartialFailMCPClient(_FakeMCPClient):
            async def get_tools(self, *, server_name: str | None = None):
                if server_name == "bad":
                    raise RuntimeError("server down")
                return await super().get_tools(server_name=server_name)

        messages = [
            ToolMessage(name="ok_get_news", content='{"symbol":"BTC","title":"ETF"}', tool_call_id="call-1"),
            AIMessage(content='{"raw_signals":[],"errors":[]}'),
        ]
        runner = MCPSignalSubgraphRunner(
            llm=_DummyLLM(),
            mcp_connections={
                "ok": {"transport": "streamable_http", "url": "http://localhost:3001/mcp"},
                "bad": {"transport": "streamable_http", "url": "http://localhost:3002/mcp"},
            },
            mcp_client_factory=_PartialFailMCPClient,
            agent_factory=lambda **kwargs: _FakeAgent(messages),
        )

        result = asyncio.run(
            runner.arun(
                user_id="u1",
                query="分析 BTC",
                task_id="task-5",
                symbols=["BTC"],
                errors=[],
            )
        )

        self.assertEqual(result["mcp_termination_reason"], "agent_completed")
        self.assertEqual(result["mcp_tools_count"], 1)
        self.assertTrue(any("MCP 工具发现失败(server=bad)" in item for item in result["errors"]))
        self.assertGreaterEqual(len(result["raw_signals"]), 1)

    def test_run_parallel_agents_across_servers(self) -> None:
        class _ParallelClient:
            def __init__(self, connections: dict[str, dict[str, Any]], tool_name_prefix: bool = False) -> None:
                self.tool_name_prefix = tool_name_prefix
                self.connections = connections

            async def get_tools(self, *, server_name: str | None = None):
                prefix = f"{server_name}_" if self.tool_name_prefix and server_name else ""
                return [_FakeTool(name=f"{prefix}get_news", description="fetch", schema={"type": "object"})]

        class _DelayedAgent:
            def __init__(self, messages: list[Any], delay_sec: float) -> None:
                self._messages = messages
                self._delay_sec = delay_sec

            async def ainvoke(self, payload: dict[str, Any]):
                del payload
                await asyncio.sleep(self._delay_sec)
                return {"messages": self._messages}

        def _agent_factory(**kwargs):
            tool = kwargs["tools"][0]
            messages = [
                ToolMessage(name=tool.name, content='{"symbol":"BTC","title":"parallel"}', tool_call_id=f"call-{tool.name}"),
                AIMessage(content='{"raw_signals":[],"errors":[]}'),
            ]
            return _DelayedAgent(messages=messages, delay_sec=0.2)

        runner = MCPSignalSubgraphRunner(
            llm=_DummyLLM(),
            mcp_connections={
                "s1": {"transport": "streamable_http", "url": "http://localhost:3011/mcp"},
                "s2": {"transport": "streamable_http", "url": "http://localhost:3012/mcp"},
                "s3": {"transport": "streamable_http", "url": "http://localhost:3013/mcp"},
            },
            mcp_client_factory=_ParallelClient,
            agent_factory=_agent_factory,
        )

        started = time.perf_counter()
        result = asyncio.run(
            runner.arun(
                user_id="u1",
                query="并行采集",
                task_id="task-parallel",
                symbols=["BTC"],
                errors=[],
            )
        )
        elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 0.45)
        self.assertEqual(result["mcp_tools_count"], 3)
        self.assertEqual(result["mcp_termination_reason"], "agent_completed")
        self.assertGreaterEqual(len(result["raw_signals"]), 3)

    def test_run_allows_llm_to_call_subset_of_tools(self) -> None:
        class _CustomClient:
            def __init__(self, connections: dict[str, dict[str, Any]], tool_name_prefix: bool = False) -> None:
                del connections
                self.tool_name_prefix = tool_name_prefix

            async def get_tools(self, *, server_name: str | None = None):
                prefix = f"{server_name}_" if self.tool_name_prefix and server_name else "srv_"
                return [
                    _FakeTool(name=f"{prefix}get_news", description="news", schema={"type": "object"}),
                    _FakeTool(name=f"{prefix}get_price", description="price", schema={"type": "object"}),
                ]

        messages = [
            ToolMessage(name="srv_get_news", content='{"symbol":"BTC","title":"agent"}', tool_call_id="call-1"),
            AIMessage(content='{"raw_signals":[],"errors":[]}'),
        ]
        runner = MCPSignalSubgraphRunner(
            llm=_DummyLLM(),
            mcp_connections={"srv": {"transport": "streamable_http", "url": "http://localhost:3000/mcp"}},
            mcp_client_factory=_CustomClient,
            agent_factory=lambda **kwargs: _FakeAgent(messages),
        )

        result = asyncio.run(
            runner.arun(
                user_id="u1",
                query="只看新闻",
                task_id="task-subset",
                symbols=["BTC"],
                errors=[],
            )
        )

        self.assertEqual(result["mcp_tools_count"], 2)
        self.assertEqual(result["mcp_termination_reason"], "agent_completed")
        self.assertGreaterEqual(len(result["raw_signals"]), 1)
        self.assertFalse(any("MCP 工具未完成调用" in item for item in result["errors"]))

    def test_run_uses_unknown_symbol_when_tool_output_has_no_symbol(self) -> None:
        messages = [
            ToolMessage(name="srv_get_news", content='{"title":"macro update"}', tool_call_id="call-1"),
            AIMessage(content='{"raw_signals":[],"errors":[]}'),
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
                query="宏观风险",
                task_id="task-unknown-1",
                symbols=[],
                errors=[],
            )
        )

        self.assertGreaterEqual(len(result["raw_signals"]), 1)
        self.assertEqual(result["raw_signals"][0]["symbol"], "UNKNOWN")

    def test_run_uses_unknown_symbol_when_payload_missing_symbol(self) -> None:
        messages = [
            AIMessage(
                content='{"raw_signals":[{"source":"mcp:srv","signal_type":"news","value":{"headline":"macro"},'
                '"raw_ref":"https://example.com/macro"}],"errors":[]}'
            )
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
                query="宏观风险",
                task_id="task-unknown-2",
                symbols=[],
                errors=[],
            )
        )

        self.assertGreaterEqual(len(result["raw_signals"]), 1)
        self.assertEqual(result["raw_signals"][0]["symbol"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
