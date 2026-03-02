"""MCP 客户端参数映射与新闻信号标准化测试。"""

from __future__ import annotations

import unittest

from mcp import McpError, types

from app.config.settings import Settings
from app.models.schemas import SignalType
from app.tools.mcp_client import MCPClient, MCPServerSpec


class MCPClientNewsServerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.client = MCPClient(Settings())

    def test_suggest_currencies_array_argument(self) -> None:
        value = self.client._suggest_argument_value(
            name="currencies",
            schema={"type": "array"},
            query="latest news",
            symbols=["BTC", "ETH"],
        )
        self.assertEqual(value, ["BTC", "ETH"])

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

    def test_select_tools_prefers_intent_and_symbol_match(self) -> None:
        tools = [
            types.Tool(
                name="get_list_coins_categories",
                description="List all categories",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="get_id_coins",
                description="Get coin detail by id",
                inputSchema={"type": "object", "properties": {"id": {"type": "string"}}},
            ),
            types.Tool(
                name="get_news_digest",
                description="Return latest crypto news digest",
                inputSchema={"type": "object", "properties": {"region": {"type": "string"}}},
            ),
        ]
        selected = self.client._select_tools(
            spec=MCPServerSpec(name="coingecko", transport="streamable_http", max_tools_per_server=3),
            tools=tools,
            query="BTC 最新新闻",
            symbols=["BTC"],
        )
        self.assertEqual(selected[0].name, "get_news_digest")
        self.assertIn("get_id_coins", [tool.name for tool in selected])

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


if __name__ == "__main__":
    unittest.main()
