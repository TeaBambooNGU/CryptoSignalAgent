"""MCP 客户端参数映射与新闻信号标准化测试。"""

from __future__ import annotations

import unittest

import httpx
from mcp import McpError, types

from app.agents.llm.base import BaseLLMClient
from app.config.settings import Settings
from app.models.schemas import SignalType
from app.tools.mcp_client import MCPClient, MCPServerSpec


class _FakeLLMClient(BaseLLMClient):
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text

    def generate(self, system_prompt: str, user_prompt: str, metadata=None) -> str:
        return self.response_text


class _SequenceFakeLLMClient(BaseLLMClient):
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.generate_calls = 0

    def generate(self, system_prompt: str, user_prompt: str, metadata=None) -> str:
        self.generate_calls += 1
        if not self._responses:
            raise RuntimeError("no more mock responses")
        return self._responses.pop(0)


class MCPClientNewsServerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.client = MCPClient(Settings())

    def test_item_to_row_prefers_requested_symbol_from_currencies(self) -> None:
        row = self.client._item_to_row(
            item={
                "title": "SEC update impacts BTC and ETH",
                "url": "https://example.com/news-1",
                "published_at": "2026-03-02T00:00:00Z",
                "currencies": [{"code": "ETH"}, {"code": "BTC"}],
                "signal_score": 71,
            },
            server_name="crypto-news-mcp",
            tool_name="get_research_signals",
            symbols=["BTC"],
            task_id="task-1",
        )
        self.assertEqual(row["symbol"], "BTC")
        self.assertEqual(row["raw_ref"], "https://example.com/news-1")
        self.assertEqual(row["published_at"], "2026-03-02T00:00:00Z")
        self.assertEqual(row["signal_type"], SignalType.NEWS.value)

    def test_normalize_raw_rows_keeps_news_type_and_publish_time(self) -> None:
        rows = [
            {
                "symbol": "ETH",
                "source": "mcp:crypto-news-mcp",
                "signal_type": "news",
                "value": {"title": "ETF flow update", "event_type": "etf"},
                "raw_ref": "https://example.com/news-2",
                "published_at": "2026-03-02T09:30:00Z",
                "metadata": {"tool": "get_news_digest"},
            }
        ]

        normalized = self.client._normalize_raw_rows(task_id="task-2", rows=rows)

        self.assertEqual(len(normalized), 1)
        signal = normalized[0]
        self.assertEqual(signal.symbol, "ETH")
        self.assertEqual(signal.signal_type, SignalType.NEWS)
        self.assertEqual(signal.raw_ref, "https://example.com/news-2")
        self.assertEqual(signal.published_at.isoformat(), "2026-03-02T09:30:00+00:00")

    def test_sanitize_for_log_masks_sensitive_fields(self) -> None:
        sanitized = self.client._sanitize_for_log(
            {
                "api_key": "key-123",
                "token": "secret-token",
                "token_id": "bitcoin",
                "nested": {"authorization": "Bearer xxx", "page": 1},
            }
        )
        self.assertEqual(sanitized["api_key"], "***")
        self.assertEqual(sanitized["token"], "***")
        self.assertEqual(sanitized["token_id"], "bitcoin")
        self.assertEqual(sanitized["nested"]["authorization"], "***")
        self.assertEqual(sanitized["nested"]["page"], 1)

    def test_summarize_tool_result_builds_safe_preview(self) -> None:
        result = types.CallToolResult(
            isError=False,
            structuredContent={"api_key": "secret", "data": [{"symbol": "BTC"}]},
            content=[
                types.TextContent(type="text", text='{"price": 100, "token":"x"}'),
                types.TextContent(type="text", text="plain text result"),
            ],
        )
        summary = self.client._summarize_tool_result(result=result, extracted_rows=2, post_processed_rows=1)
        self.assertEqual(summary["is_error"], False)
        self.assertEqual(summary["content_blocks"], 2)
        self.assertEqual(summary["extracted_rows"], 2)
        self.assertEqual(summary["post_processed_rows"], 1)
        self.assertEqual(summary["structured_content"]["type"], "dict")
        self.assertEqual(summary["structured_content"]["data_count"], 1)
        self.assertEqual(len(summary["text_preview"]), 2)

    def test_extract_tool_error_detail_reads_status_and_body(self) -> None:
        result = types.CallToolResult(
            isError=True,
            structuredContent={
                "status_code": 502,
                "response_body": {"detail": "upstream timeout"},
            },
            content=[types.TextContent(type="text", text='{"status":502,"body":"gateway timeout"}')],
        )
        detail = self.client._extract_tool_error_detail(result)
        self.assertEqual(detail["status_code"], 502)
        self.assertEqual(detail["response_body"], {"detail": "upstream timeout"})
        self.assertIn("gateway timeout", detail["content_text"][0])

    def test_build_exception_error_detail_keeps_mcp_error_data(self) -> None:
        exc = McpError(
            types.ErrorData(
                code=500,
                message="upstream failed",
                data={"status_code": 503, "response_body": "service unavailable"},
            )
        )
        detail = self.client._build_exception_error_detail(exc)
        self.assertEqual(detail["mcp_error_code"], 500)
        self.assertEqual(detail["status_code"], 503)
        self.assertEqual(detail["response_body"], "service unavailable")

    def test_plan_tool_calls_with_llm_returns_valid_calls(self) -> None:
        llm = _FakeLLMClient(
            """
            {
              "calls": [
                {
                  "tool_name": "get_coins_markets",
                  "arguments": {"vs_currency": "usd", "ids": "bitcoin,ethereum"},
                  "reason": "获取主流币行情"
                }
              ]
            }
            """
        )
        client = MCPClient(Settings(), llm_client=llm)
        tools = [
            types.Tool(
                name="get_coins_markets",
                description="query coins market data",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "vs_currency": {"type": "string"},
                        "ids": {"type": "string"},
                    },
                    "required": ["vs_currency"],
                },
            ),
        ]
        errors: list[str] = []
        calls = client._plan_tool_calls_with_llm(
            spec=MCPServerSpec(name="coingecko", transport="streamable_http", max_tools_per_server=2),
            tools=tools,
            query="请分析 BTC 和 ETH 行情",
            symbols=["BTC", "ETH"],
            errors=errors,
            historical_corrections=[],
            failure_feedback=[],
        )
        self.assertEqual(len(calls), 1)
        tool, arguments, reason = calls[0]
        self.assertEqual(tool.name, "get_coins_markets")
        self.assertEqual(arguments["vs_currency"], "usd")
        self.assertEqual(arguments["ids"], "bitcoin,ethereum")
        self.assertEqual(reason, "获取主流币行情")
        self.assertEqual(errors, [])

    def test_plan_tool_calls_with_llm_rejects_missing_required_arguments(self) -> None:
        llm = _FakeLLMClient(
            """
            {
              "calls": [
                {
                  "tool_name": "get_networks_onchain_dexes",
                  "arguments": {"page": 1},
                  "reason": "缺少 required network"
                }
              ]
            }
            """
        )
        client = MCPClient(Settings(), llm_client=llm)
        tools = [
            types.Tool(
                name="get_networks_onchain_dexes",
                description="query dex list by network",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "network": {"type": "string"},
                        "page": {"type": "integer"},
                    },
                    "required": ["network"],
                },
            ),
        ]
        errors: list[str] = []
        with self.assertRaises(ValueError):
            client._plan_tool_calls_with_llm(
                spec=MCPServerSpec(name="coingecko", transport="streamable_http", max_tools_per_server=2),
                tools=tools,
                query="给我 onchain dex 数据",
                symbols=["BTC"],
                errors=errors,
                historical_corrections=[],
                failure_feedback=[],
            )
        self.assertTrue(any("missing required `network`" in message for message in errors))

    def test_plan_tool_calls_with_llm_retries_after_invalid_json(self) -> None:
        llm = _SequenceFakeLLMClient(
            [
                "抱歉我先解释一下调用思路",
                """
                {
                  "calls": [
                    {
                      "tool_name": "get_coins_markets",
                      "arguments": {"vs_currency": "usd"},
                      "reason": "补齐关键行情"
                    }
                  ]
                }
                """,
            ]
        )
        client = MCPClient(Settings(), llm_client=llm)
        tools = [
            types.Tool(
                name="get_coins_markets",
                description="query coins market data",
                inputSchema={
                    "type": "object",
                    "properties": {"vs_currency": {"type": "string"}},
                    "required": ["vs_currency"],
                },
            )
        ]
        errors: list[str] = []
        calls = client._plan_tool_calls_with_llm(
            spec=MCPServerSpec(name="coingecko", transport="streamable_http", max_tools_per_server=1),
            tools=tools,
            query="给我 BTC 行情",
            symbols=["BTC"],
            errors=errors,
            historical_corrections=[],
            failure_feedback=[],
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(llm.generate_calls, 2)

    def test_plan_tool_calls_with_llm_rejects_unknown_argument(self) -> None:
        llm = _FakeLLMClient(
            """
            {
              "calls": [
                {
                  "tool_name": "get_coins_markets",
                  "arguments": {"vs_currency": "usd", "foo": "bar"},
                  "reason": "参数包含未知字段"
                }
              ]
            }
            """
        )
        client = MCPClient(Settings(), llm_client=llm)
        tools = [
            types.Tool(
                name="get_coins_markets",
                description="query coins market data",
                inputSchema={
                    "type": "object",
                    "properties": {"vs_currency": {"type": "string"}},
                    "required": ["vs_currency"],
                },
            )
        ]
        errors: list[str] = []
        with self.assertRaises(ValueError):
            client._plan_tool_calls_with_llm(
                spec=MCPServerSpec(name="coingecko", transport="streamable_http", max_tools_per_server=1),
                tools=tools,
                query="给我 BTC 行情",
                symbols=["BTC"],
                errors=errors,
                historical_corrections=[],
                failure_feedback=[],
            )
        self.assertTrue(any("unknown argument" in message for message in errors))

    def test_build_exception_error_detail_extracts_response_from_exception_group(self) -> None:
        request = httpx.Request("POST", "https://example.com/mcp")
        response = httpx.Response(500, request=request, text='{"detail":"upstream failed"}')
        exc = ExceptionGroup("group", [httpx.HTTPStatusError("boom", request=request, response=response)])
        detail = self.client._build_exception_error_detail(exc)
        self.assertEqual(detail["status_code"], 500)
        self.assertIn("upstream failed", detail["response_body"])
        self.assertTrue(isinstance(detail.get("sub_exceptions"), list))

    def test_retryable_server_exception_false_for_http_400(self) -> None:
        request = httpx.Request("POST", "https://example.com/mcp")
        response = httpx.Response(400, request=request, text="Protocol not found")
        exc = httpx.HTTPStatusError("bad request", request=request, response=response)
        self.assertFalse(self.client._is_retryable_server_exception(exc))

    def test_retryable_server_exception_true_for_http_500(self) -> None:
        request = httpx.Request("POST", "https://example.com/mcp")
        response = httpx.Response(500, request=request, text="Internal Server Error")
        exc = httpx.HTTPStatusError("server error", request=request, response=response)
        self.assertTrue(self.client._is_retryable_server_exception(exc))

    def test_is_deterministic_error_message_for_validation_error(self) -> None:
        message = (
            "1 validation error for get_research_signalsArguments "
            "currencies Input should be a valid list [type=list_type]"
        )
        self.assertTrue(self.client._is_deterministic_error_message(message))

    def test_parse_planner_payload_accepts_noisy_text_with_json_object(self) -> None:
        payload = self.client._parse_planner_payload(
            """
            下面是规划结果（请忽略这句）：
            {
              "calls": [
                {
                  "tool_name": "get_coins_markets",
                  "arguments": {"vs_currency": "usd"},
                  "reason": "获取行情"
                }
              ]
            }
            """
        )
        self.assertEqual(payload["calls"][0]["tool_name"], "get_coins_markets")

    def test_build_replan_feedback_includes_repeated_5xx_failure(self) -> None:
        repeated_failure = {
            "server": "coingecko",
            "tool_name": "get_coins_markets",
            "arguments": {"vs_currency": "usd", "symbols": "BTC,ETH"},
            "error_detail": {"status_code": 500, "error_message": "internal server error"},
            "deterministic": False,
        }
        feedback = self.client._build_replan_feedback(
            previous_failures=[repeated_failure],
            current_failures=[repeated_failure],
        )
        self.assertEqual(len(feedback), 1)
        self.assertEqual(feedback[0]["tool_name"], "get_coins_markets")

    def test_build_replan_feedback_skips_single_5xx_failure(self) -> None:
        single_failure = {
            "server": "coingecko",
            "tool_name": "get_coins_markets",
            "arguments": {"vs_currency": "usd", "symbols": "BTC,ETH"},
            "error_detail": {"status_code": 500, "error_message": "internal server error"},
            "deterministic": False,
        }
        feedback = self.client._build_replan_feedback(
            previous_failures=[],
            current_failures=[single_failure],
        )
        self.assertEqual(feedback, [])

    def test_finalize_failures_drops_resolved_deterministic_error(self) -> None:
        errors = [
            "MCP 工具返回错误: crypto-news-mcp/get_research_signals bad args",
            "MCP 工具返回错误: coingecko/get_coins_markets 500",
        ]
        failures = [
            {
                "server": "crypto-news-mcp",
                "tool_name": "get_research_signals",
                "arguments": {"currencies": "BTC,ETH"},
                "reason": "bad args",
                "error_detail": {"status_code": 400},
                "deterministic": True,
                "_error_index": 0,
                "_resolved": True,
            },
            {
                "server": "coingecko",
                "tool_name": "get_coins_markets",
                "arguments": {"symbols": "BTC,ETH"},
                "reason": "market",
                "error_detail": {"status_code": 500},
                "deterministic": False,
                "_error_index": 1,
                "_resolved": False,
            },
        ]
        final_errors, final_failures = self.client._finalize_failures(errors=errors, failures=failures)
        self.assertEqual(final_errors, [errors[1]])
        self.assertEqual(len(final_failures), 1)
        self.assertEqual(final_failures[0]["tool_name"], "get_coins_markets")

    def test_mark_failures_resolved_by_success_matches_tool_name(self) -> None:
        failures = [
            {
                "server": "crypto-news-mcp",
                "tool_name": "get_research_signals",
                "arguments": {"currencies": "BTC,ETH"},
                "reason": "bad args",
                "error_detail": {"status_code": 400},
                "deterministic": True,
                "_resolved": False,
            }
        ]
        self.client._mark_failures_resolved_by_success(
            failures=failures,
            success_tool_name="get_research_signals",
        )
        self.assertTrue(failures[0]["_resolved"])


if __name__ == "__main__":
    unittest.main()
