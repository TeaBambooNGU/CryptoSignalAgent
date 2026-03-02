"""标准 MCP 数据采集客户端。

V1 目标：
- 使用 MCP 官方协议（stdio / streamable HTTP / SSE）调用外部工具。
- 将 tool 返回结果统一映射为内部 RawSignal。
"""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from mcp import ClientSession, types
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.exceptions import McpError
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config.logging import get_logger, log_context
from app.config.settings import Settings
from app.models.schemas import RawSignal, SignalType

logger = get_logger(__name__)

URL_PATTERN = re.compile(r"https?://\\S+")
LOG_MAX_TEXT_CHARS = 256
LOG_MAX_LIST_ITEMS = 5
LOG_MAX_DEPTH = 4
ERROR_LOG_MAX_LIST_ITEMS = 50
SENSITIVE_MASK = "***"
SYMBOL_TO_CG_ID = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOGE": "dogecoin",
}


@dataclass(slots=True)
class MCPServerSpec:
    """MCP 服务器配置。"""

    name: str
    transport: str
    url: str = ""
    command: str = ""
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    cwd: str | None = None
    tool_allowlist: tuple[str, ...] = ()
    max_tools_per_server: int = 3


class MCPClient:
    """MCP 采集适配层。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def collect_signals(
        self,
        task_id: str,
        query: str,
        symbols: list[str],
    ) -> tuple[list[RawSignal], list[str]]:
        """通过标准 MCP 协议采集原始信号。"""

        errors: list[str] = []
        specs = self._load_server_specs()
        with log_context(component="mcp.collect"):
            logger.info("MCP 采集开始 servers=%s symbols=%s", len(specs), symbols or ["BTC"])
        if not specs:
            errors.append("未配置 MCP_SERVERS，跳过实时采集")
            return [], errors

        raw_rows: list[dict[str, Any]] = []
        for spec in specs:
            try:
                # 对服务级采集增加重试，覆盖临时网络抖动或上游瞬态故障。
                rows: list[dict[str, Any]] = []
                server_errors: list[str] = []
                for attempt in Retrying(
                    stop=stop_after_attempt(3),
                    wait=wait_exponential(multiplier=1, min=1, max=16),
                    retry=retry_if_exception_type(Exception),
                    reraise=True,
                ):
                    with attempt:
                        rows, server_errors = asyncio.run(
                            self._collect_from_server(
                                spec=spec,
                                task_id=task_id,
                                query=query,
                                symbols=symbols,
                            )
                        )
                errors.extend(server_errors)
                with log_context(component="mcp.collect"):
                    logger.info("MCP 服务完成 server=%s rows=%s errors=%s", spec.name, len(rows), len(server_errors))
                raw_rows.extend(rows)
            except Exception as exc:
                logger.exception("MCP 服务采集失败: %s", spec.name)
                errors.append(f"MCP 服务失败: {spec.name} ({type(exc).__name__})")

        normalized = self._normalize_raw_rows(task_id=task_id, rows=raw_rows)
        with log_context(component="mcp.collect"):
            logger.info("MCP 采集结束 raw_rows=%s normalized=%s errors=%s", len(raw_rows), len(normalized), len(errors))
        return normalized, errors

    def _load_server_specs(self) -> list[MCPServerSpec]:
        """从配置加载 MCP server 列表。"""

        specs: list[MCPServerSpec] = []

        for index, item in enumerate(self.settings.mcp_servers, start=1):
            name = str(item.get("name", f"server-{index}"))
            transport = str(item.get("transport", "streamable_http")).strip().lower()
            url = str(item.get("url", ""))
            command = str(item.get("command", ""))
            args_raw = item.get("args", [])
            if isinstance(args_raw, list):
                args = tuple(str(arg) for arg in args_raw)
            else:
                args = ()

            env_raw = item.get("env")
            env: dict[str, str] | None = None
            if isinstance(env_raw, dict):
                env = {str(k): str(v) for k, v in env_raw.items()}

            tool_allowlist_raw = item.get("tool_allowlist", [])
            tool_allowlist = (
                tuple(str(name) for name in tool_allowlist_raw)
                if isinstance(tool_allowlist_raw, list)
                else ()
            )

            max_tools = item.get("max_tools_per_server", 3)
            if not isinstance(max_tools, int):
                max_tools = 3

            specs.append(
                MCPServerSpec(
                    name=name,
                    transport=transport,
                    url=url,
                    command=command,
                    args=args,
                    env=env,
                    cwd=str(item.get("cwd")) if item.get("cwd") else None,
                    tool_allowlist=tool_allowlist,
                    max_tools_per_server=max(1, max_tools),
                )
            )

        with log_context(component="mcp.collect"):
            logger.info("MCP 服务配置加载完成 servers=%s", len(specs))
        return specs

    async def _collect_from_server(
        self,
        spec: MCPServerSpec,
        task_id: str,
        query: str,
        symbols: list[str],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """连接单个 MCP 服务并采集数据。"""

        async with self._open_session(spec) as session:
            tools_result = await session.list_tools()
            selected_tools = self._select_tools(
                spec=spec,
                tools=tools_result.tools,
                query=query,
                symbols=symbols,
            )
            if not selected_tools:
                return [], [f"MCP 无可用工具: {spec.name}"]

            with log_context(component="mcp.collect"):
                logger.info("工具筛选完成 server=%s selected_tools=%s", spec.name, len(selected_tools))
            rows: list[dict[str, Any]] = []
            errors: list[str] = []
            for tool in selected_tools[: spec.max_tools_per_server]:
                arguments = self._build_tool_arguments(tool=tool, query=query, symbols=symbols)
                safe_arguments = self._sanitize_for_log(arguments)
                with log_context(component="mcp.collect"):
                    logger.info(
                        "MCP 工具调用开始 server=%s tool=%s arguments=%s",
                        spec.name,
                        tool.name,
                        self._format_log_payload(safe_arguments),
                    )
                try:
                    result = await session.call_tool(tool.name, arguments=arguments)
                except Exception as exc:
                    error_detail = self._build_exception_error_detail(exc)
                    errors.append(
                        "MCP 工具异常: "
                        f"{spec.name}/{tool.name} {self._format_log_payload(error_detail)}"
                    )
                    logger.warning(
                        "MCP 工具调用失败 server=%s tool=%s arguments=%s error=%s",
                        spec.name,
                        tool.name,
                        self._format_log_payload(safe_arguments),
                        self._format_log_payload(error_detail),
                    )
                    continue

                if result.isError:
                    error_detail = self._extract_tool_error_detail(result)
                    errors.append(
                        "MCP 工具返回错误: "
                        f"{spec.name}/{tool.name} {self._format_log_payload(error_detail)}"
                    )
                    with log_context(component="mcp.collect"):
                        logger.warning(
                            "MCP 工具返回错误 server=%s tool=%s arguments=%s error=%s",
                            spec.name,
                            tool.name,
                            self._format_log_payload(safe_arguments),
                            self._format_log_payload(error_detail),
                        )
                    continue

                extracted_rows = self._extract_rows_from_tool_result(
                    result=result,
                    server_name=spec.name,
                    tool_name=tool.name,
                    symbols=symbols,
                    task_id=task_id,
                )
                tool_rows = self._post_process_rows(rows=extracted_rows, symbols=symbols)
                rows.extend(tool_rows)
                with log_context(component="mcp.collect"):
                    logger.info(
                        "工具调用完成 server=%s tool=%s tool_rows=%s total_rows=%s result=%s",
                        spec.name,
                        tool.name,
                        len(tool_rows),
                        len(rows),
                        self._format_log_payload(
                            self._summarize_tool_result(
                                result=result,
                                extracted_rows=len(extracted_rows),
                                post_processed_rows=len(tool_rows),
                            )
                        ),
                    )

            return rows, errors

    def _post_process_rows(self, rows: list[dict[str, Any]], symbols: list[str]) -> list[dict[str, Any]]:
        """对工具返回行做轻量过滤，避免无关大批量数据淹没主信号。"""

        if not rows:
            return rows

        target_symbols = {symbol.upper() for symbol in symbols}
        filtered: list[dict[str, Any]] = []
        for row in rows:
            signal_type = str(row.get("signal_type", SignalType.NEWS.value))
            symbol = str(row.get("symbol", "")).upper()
            if (
                target_symbols
                and symbol
                and symbol not in target_symbols
                and signal_type in {SignalType.PRICE.value, SignalType.SENTIMENT.value}
            ):
                continue
            filtered.append(row)

        # 单工具最多保留 120 条，降低噪声与写入压力。
        return (filtered or rows)[:120]

    @asynccontextmanager
    async def _open_session(self, spec: MCPServerSpec):
        """按 transport 打开 MCP 会话。"""

        transport = spec.transport
        if transport == "stdio":
            if not spec.command:
                raise ValueError(f"stdio MCP 服务缺少 command: {spec.name}")
            params = StdioServerParameters(
                command=spec.command,
                args=list(spec.args),
                env=spec.env,
                cwd=spec.cwd,
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session
            return

        if transport == "sse":
            if not spec.url:
                raise ValueError(f"sse MCP 服务缺少 url: {spec.name}")
            async with sse_client(spec.url, timeout=20, sse_read_timeout=120) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session
            return

        # 默认走 streamable HTTP
        if not spec.url:
            raise ValueError(f"streamable_http MCP 服务缺少 url: {spec.name}")
        async with streamablehttp_client(
            spec.url,
            timeout=20,
            sse_read_timeout=120,
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    def _select_tools(
        self,
        spec: MCPServerSpec,
        tools: list[types.Tool],
        query: str,
        symbols: list[str],
    ) -> list[types.Tool]:
        """按服务器配置或关键词策略筛选工具。"""

        if spec.tool_allowlist:
            allow = set(spec.tool_allowlist)
            return [tool for tool in tools if tool.name in allow]

        intents = self._infer_query_intents(query)
        scored_tools: list[tuple[float, types.Tool]] = []
        for tool in tools:
            score = self._score_tool(tool=tool, intents=intents, symbols=symbols)
            if score > 0:
                scored_tools.append((score, tool))

        if scored_tools:
            scored_tools.sort(key=lambda item: (-item[0], item[1].name))
            candidate_limit = max(spec.max_tools_per_server * 3, spec.max_tools_per_server)
            return [tool for _, tool in scored_tools[:candidate_limit]]

        return tools[:3]

    def _infer_query_intents(self, query: str) -> set[str]:
        """从查询语义提取意图标签。"""

        lowered = query.lower()
        intents: set[str] = set()
        if any(token in lowered for token in ("news", "headline", "digest", "brief", "article", "新闻", "快讯")):
            intents.add("news")
        if any(token in lowered for token in ("chain", "onchain", "tvl", "protocol", "defi", "链上", "协议")):
            intents.add("onchain")
        if any(token in lowered for token in ("price", "market", "funding", "ticker", "行情", "价格")):
            intents.add("price")
        if not intents:
            intents.add("market")
        return intents

    def _score_tool(self, *, tool: types.Tool, intents: set[str], symbols: list[str]) -> float:
        """根据意图与 symbol 可控性对工具打分。"""

        text = f"{tool.name} {tool.description or ''}".lower()
        schema = tool.inputSchema if isinstance(tool.inputSchema, dict) else {}
        properties = schema.get("properties", {})
        property_names = {str(name).lower() for name in properties}

        symbol_params = {"symbol", "symbols", "ticker", "tickers", "id", "coin_id", "ids", "coin_ids", "currencies"}
        query_params = {"query", "keyword", "search"}

        score = 0.0

        intent_tokens: dict[str, tuple[str, ...]] = {
            "news": ("news", "digest", "brief", "headline", "article", "research", "catalyst"),
            "onchain": ("onchain", "chain", "protocol", "defi", "tvl"),
            "price": ("price", "market", "ticker", "funding", "ohlc", "volume"),
            "market": ("market", "coin", "token", "trend", "signal"),
        }
        for intent in intents:
            tokens = intent_tokens.get(intent, ())
            if any(token in text for token in tokens):
                score += 2.5

        if property_names & symbol_params:
            score += 2.0
        if property_names & query_params:
            score += 1.0
        if symbols and any(symbol.lower() in text for symbol in symbols):
            score += 0.5

        tool_name = tool.name.lower()
        if any(token in tool_name for token in ("list_coins_categories", "new_coins_list", "all_categories")):
            score -= 4.0
        if "list" in tool_name and not (property_names & symbol_params or property_names & query_params):
            score -= 2.0

        return score

    def _build_tool_arguments(self, tool: types.Tool, query: str, symbols: list[str]) -> dict[str, Any]:
        """根据 inputSchema 生成调用参数。"""

        schema = tool.inputSchema if isinstance(tool.inputSchema, dict) else {}
        properties = schema.get("properties", {})
        required = schema.get("required", []) or []

        args: dict[str, Any] = {}
        for name, info in properties.items():
            value = self._suggest_argument_value(
                name=name,
                schema=info if isinstance(info, dict) else {},
                query=query,
                symbols=symbols,
            )
            if value is not None:
                args[name] = value

        # required 参数兜底，避免遗漏。
        for name in required:
            if name in args:
                continue
            value = self._suggest_argument_value(name=name, schema={}, query=query, symbols=symbols)
            if value is not None:
                args[name] = value

        return args

    def _suggest_argument_value(
        self,
        name: str,
        schema: dict[str, Any],
        query: str,
        symbols: list[str],
    ) -> Any | None:
        """按参数名语义推断参数值。"""

        lowered = name.lower()
        enums = schema.get("enum")
        if isinstance(enums, list) and enums:
            return enums[0]
        if "default" in schema:
            return schema["default"]

        mapped_ids = [SYMBOL_TO_CG_ID.get(symbol.upper(), symbol.lower()) for symbol in symbols] or ["bitcoin"]

        if lowered in {"vs_currency", "currency", "quote_currency"}:
            return "usd"
        if lowered in {"id", "coin_id", "token_id", "asset_id", "protocol"}:
            return mapped_ids[0]
        if lowered in {"ids", "coin_ids", "token_ids"}:
            return ",".join(mapped_ids)
        if lowered in {"currencies"}:
            if schema.get("type") == "array":
                return symbols or ["BTC", "ETH"]
            return ",".join(symbols) if symbols else "BTC,ETH"
        if lowered in {"symbol", "ticker"}:
            return (symbols[0] if symbols else "BTC").lower()
        if lowered in {"symbols", "tickers"}:
            return ",".join(symbols).lower() if symbols else "btc,eth"
        if lowered in {"region", "lang", "language"}:
            return "en"
        if lowered in {"kind"}:
            return "news"
        if lowered in {"news_filter"}:
            return "important"
        if lowered in {"lookback_hours"}:
            return 24
        if lowered in {"top_k"}:
            return 10
        if lowered in {"min_score"}:
            return 0
        if lowered in {"public_mode"}:
            return False
        if lowered in {"horizon"}:
            return "intraday"
        if "query" in lowered or "search" in lowered or "keyword" in lowered:
            return query
        if lowered in {"per_page", "limit", "count", "size"}:
            return 10
        if lowered in {"page", "offset"}:
            return 1
        if lowered in {"from", "start", "from_timestamp"}:
            return int((datetime.now(timezone.utc) - timedelta(hours=24)).timestamp())
        if lowered in {"to", "end", "to_timestamp"}:
            return int(datetime.now(timezone.utc).timestamp())
        if lowered == "date":
            return datetime.now(timezone.utc).strftime("%d-%m-%Y")
        if lowered == "site":
            return "coindesk"

        value_type = schema.get("type")
        if value_type == "integer":
            return 1
        if value_type == "number":
            return 1
        if value_type == "boolean":
            return False

        return None

    def _extract_rows_from_tool_result(
        self,
        result: types.CallToolResult,
        server_name: str,
        tool_name: str,
        symbols: list[str],
        task_id: str,
    ) -> list[dict[str, Any]]:
        """将 CallToolResult 转换为标准化前的字典行。"""

        if result.isError:
            return []

        extracted_items: list[Any] = []
        if result.structuredContent is not None:
            extracted_items.extend(self._flatten_item(result.structuredContent))

        for block in result.content:
            if isinstance(block, types.TextContent):
                text = block.text.strip()
                parsed = self._try_parse_json(text)
                if parsed is not None:
                    extracted_items.extend(self._flatten_item(parsed))
                else:
                    extracted_items.append(text)

        rows: list[dict[str, Any]] = []
        for item in extracted_items:
            rows.append(
                self._item_to_row(
                    item=item,
                    server_name=server_name,
                    tool_name=tool_name,
                    symbols=symbols,
                    task_id=task_id,
                )
            )
        return rows

    def _summarize_tool_result(
        self,
        *,
        result: types.CallToolResult,
        extracted_rows: int,
        post_processed_rows: int,
    ) -> dict[str, Any]:
        """构建工具返回摘要，避免日志中出现超大原始 payload。"""

        content_types = [getattr(block, "type", type(block).__name__) for block in result.content]
        text_previews: list[str] = []
        for block in result.content:
            if isinstance(block, types.TextContent):
                text_previews.append(self._truncate_log_text(block.text.strip()))

        return {
            "is_error": result.isError,
            "content_blocks": len(result.content),
            "content_types": content_types,
            "text_preview": text_previews[:2],
            "structured_content": self._summarize_payload(result.structuredContent),
            "extracted_rows": extracted_rows,
            "post_processed_rows": post_processed_rows,
        }

    def _summarize_payload(self, value: Any) -> Any:
        """输出紧凑 payload 摘要，避免日志被大字段淹没。"""

        if value is None:
            return None
        if isinstance(value, dict):
            keys = [str(key) for key in list(value.keys())[:8]]
            summary: dict[str, Any] = {"type": "dict", "keys": keys}
            for list_key in ("items", "data", "results", "coins", "news"):
                data = value.get(list_key)
                if isinstance(data, list):
                    summary[f"{list_key}_count"] = len(data)
            if "count" in value and isinstance(value.get("count"), (int, float)):
                summary["count"] = value["count"]
            return summary
        if isinstance(value, list):
            return {"type": "list", "length": len(value)}
        if isinstance(value, str):
            return self._truncate_log_text(value)
        if isinstance(value, (int, float, bool)):
            return value
        return self._truncate_log_text(str(value))

    def _sanitize_for_log(
        self,
        value: Any,
        *,
        depth: int = 0,
        max_text_chars: int | None = LOG_MAX_TEXT_CHARS,
        max_list_items: int = LOG_MAX_LIST_ITEMS,
    ) -> Any:
        """日志输出前做脱敏与截断。"""

        if depth >= LOG_MAX_DEPTH:
            return "<max_depth>"

        if isinstance(value, dict):
            sanitized: dict[str, Any] = {}
            for key, raw_value in value.items():
                key_text = str(key)
                if self._is_sensitive_key(key_text):
                    sanitized[key_text] = SENSITIVE_MASK
                else:
                    sanitized[key_text] = self._sanitize_for_log(
                        raw_value,
                        depth=depth + 1,
                        max_text_chars=max_text_chars,
                        max_list_items=max_list_items,
                    )
            return sanitized

        if isinstance(value, list):
            items = [
                self._sanitize_for_log(
                    item,
                    depth=depth + 1,
                    max_text_chars=max_text_chars,
                    max_list_items=max_list_items,
                )
                for item in value[:max_list_items]
            ]
            if len(value) > max_list_items:
                items.append(f"...({len(value) - max_list_items} more items)")
            return items

        if isinstance(value, tuple):
            return self._sanitize_for_log(
                list(value),
                depth=depth + 1,
                max_text_chars=max_text_chars,
                max_list_items=max_list_items,
            )

        if isinstance(value, str):
            return self._truncate_log_text(value, max_text_chars=max_text_chars)

        if isinstance(value, (int, float, bool)) or value is None:
            return value

        return self._truncate_log_text(str(value), max_text_chars=max_text_chars)

    def _is_sensitive_key(self, key: str) -> bool:
        """识别需要脱敏的字段名。"""

        lowered = key.strip().lower()
        if lowered.endswith("_id") and lowered not in {"session_id", "client_id"}:
            return False

        sensitive_tokens = (
            "api_key",
            "apikey",
            "secret",
            "password",
            "passwd",
            "authorization",
            "bearer",
            "private_key",
            "client_secret",
            "access_token",
            "refresh_token",
            "id_token",
            "cookie",
            "token",
        )
        return any(token in lowered for token in sensitive_tokens)

    def _truncate_log_text(self, text: str, *, max_text_chars: int | None = LOG_MAX_TEXT_CHARS) -> str:
        """截断超长日志文本，控制单条日志体积。"""

        cleaned = text.replace("\n", "\\n")
        if max_text_chars is None or len(cleaned) <= max_text_chars:
            return cleaned
        return f"{cleaned[:max_text_chars]}...(truncated)"

    def _format_log_payload(self, payload: Any) -> str:
        """将日志 payload 序列化为单行字符串。"""

        try:
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            return self._truncate_log_text(str(payload))

    def _extract_tool_error_detail(self, result: types.CallToolResult) -> dict[str, Any]:
        """提取工具错误详情，保留状态码与响应体。"""

        detail: dict[str, Any] = {"is_error": True}
        if result.structuredContent is not None:
            detail["structured_content"] = self._sanitize_for_log(
                result.structuredContent,
                max_text_chars=None,
                max_list_items=ERROR_LOG_MAX_LIST_ITEMS,
            )
            self._merge_status_body_from_payload(
                detail,
                result.structuredContent,
                source_key_prefix="structured_content",
            )

        text_blocks: list[str] = []
        for block in result.content:
            if isinstance(block, types.TextContent):
                text = block.text.strip()
                if not text:
                    continue
                text_blocks.append(self._truncate_log_text(text, max_text_chars=None))
                parsed = self._try_parse_json(text)
                if parsed is not None:
                    self._merge_status_body_from_payload(detail, parsed, source_key_prefix="content_json")

        if text_blocks:
            detail["content_text"] = text_blocks

        if "status_code" not in detail and "response_body" not in detail:
            detail["error_message"] = text_blocks[0] if text_blocks else "unknown_error"
        return detail

    def _build_exception_error_detail(self, exc: Exception) -> dict[str, Any]:
        """提取异常详情，避免丢失状态码与响应体。"""

        detail: dict[str, Any] = {
            "exception_type": type(exc).__name__,
            "message": self._truncate_log_text(str(exc), max_text_chars=None),
        }

        if isinstance(exc, McpError):
            detail["mcp_error_code"] = exc.error.code
            detail["mcp_error_message"] = exc.error.message
            detail["mcp_error_data"] = self._sanitize_for_log(
                exc.error.data,
                max_text_chars=None,
                max_list_items=ERROR_LOG_MAX_LIST_ITEMS,
            )
            self._merge_status_body_from_payload(detail, exc.error.data, source_key_prefix="mcp_error_data")

        response = getattr(exc, "response", None)
        if response is not None:
            status_code = getattr(response, "status_code", None)
            if status_code is not None:
                detail["status_code"] = status_code
            body = None
            try:
                body = response.text
            except Exception:
                body = None
            if body:
                detail["response_body"] = self._truncate_log_text(str(body), max_text_chars=None)

        if "status_code" not in detail:
            status_code = getattr(exc, "status_code", None)
            if status_code is not None:
                detail["status_code"] = status_code

        if "response_body" not in detail:
            body = getattr(exc, "body", None)
            if body:
                detail["response_body"] = self._truncate_log_text(str(body), max_text_chars=None)

        return detail

    def _merge_status_body_from_payload(
        self,
        detail: dict[str, Any],
        payload: Any,
        *,
        source_key_prefix: str,
        depth: int = 0,
    ) -> None:
        """从任意 payload 中递归提取状态码与响应体。"""

        if depth > 6:
            return

        if isinstance(payload, dict):
            for key, value in payload.items():
                lowered = str(key).strip().lower()

                if lowered in {"status", "status_code", "http_status", "http_status_code"}:
                    parsed = self._coerce_status_code(value)
                    if parsed is not None and "status_code" not in detail:
                        detail["status_code"] = parsed
                        detail["status_code_source"] = f"{source_key_prefix}.{key}"

                if lowered in {"response", "response_body", "body", "raw_body", "error_body"}:
                    if "response_body" not in detail:
                        detail["response_body"] = self._sanitize_for_log(value, max_text_chars=None)
                        detail["response_body_source"] = f"{source_key_prefix}.{key}"

                if isinstance(value, (dict, list)):
                    self._merge_status_body_from_payload(
                        detail,
                        value,
                        source_key_prefix=f"{source_key_prefix}.{key}",
                        depth=depth + 1,
                    )
            return

        if isinstance(payload, list):
            for index, item in enumerate(payload):
                if isinstance(item, (dict, list)):
                    self._merge_status_body_from_payload(
                        detail,
                        item,
                        source_key_prefix=f"{source_key_prefix}[{index}]",
                        depth=depth + 1,
                    )

    def _coerce_status_code(self, value: Any) -> int | None:
        """尽量将状态码字段转为 int。"""

        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            digits = "".join(char for char in value if char.isdigit())
            if digits:
                try:
                    return int(digits)
                except ValueError:
                    return None
        return None

    def _flatten_item(self, data: Any) -> list[Any]:
        """将结构化内容扁平化为 item 列表。"""

        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("data", "items", "result", "results", "coins", "news"):
                value = data.get(key)
                if isinstance(value, list):
                    return value
            return [data]
        return [data]

    def _try_parse_json(self, text: str) -> Any | None:
        """尝试将文本解析为 JSON。"""

        if not text:
            return None
        if not (text.startswith("{") or text.startswith("[")):
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _item_to_row(
        self,
        item: Any,
        server_name: str,
        tool_name: str,
        symbols: list[str],
        task_id: str,
    ) -> dict[str, Any]:
        """将单条 item 映射为 RawSignal 输入结构。"""

        symbol = symbols[0].upper() if symbols else "BTC"
        raw_ref = ""
        published_at = datetime.now(timezone.utc).isoformat()

        if isinstance(item, dict):
            symbol = self._extract_symbol_from_item(item=item, requested_symbols=symbols, fallback=symbol)
            raw_ref = str(item.get("url") or item.get("link") or item.get("source_url") or "")
            published_at = self._extract_published_at(item) or published_at
            value = item
        else:
            text = str(item)
            value = text
            matched_symbols = [sym for sym in symbols if sym.upper() in text.upper()]
            if matched_symbols:
                symbol = matched_symbols[0].upper()
            url_match = URL_PATTERN.search(text)
            raw_ref = url_match.group(0) if url_match else ""

        signal_type = self._infer_signal_type(tool_name=tool_name, server_name=server_name, value=value)
        return {
            "symbol": symbol,
            "source": f"mcp:{server_name}",
            "signal_type": signal_type,
            "value": value,
            "raw_ref": raw_ref or f"mcp://{server_name}/{tool_name}",
            "published_at": published_at,
            "task_id": task_id,
            "metadata": {"tool": tool_name},
        }

    def _infer_signal_type(self, tool_name: str, server_name: str, value: Any) -> str:
        """根据工具名与值内容推断信号类型。"""

        name = f"{server_name} {tool_name}".lower()
        tool_name_lower = tool_name.lower()
        server_name_lower = server_name.lower()

        if any(token in tool_name_lower for token in ("news", "digest", "brief", "headline", "article", "catalyst")):
            return SignalType.NEWS.value
        if "research_signals" in tool_name_lower:
            return SignalType.NEWS.value
        if any(token in server_name_lower for token in ("news", "cryptopanic")):
            return SignalType.NEWS.value
        if any(token in name for token in ("news", "rss", "headline", "article")):
            return SignalType.NEWS.value
        if any(token in name for token in ("chain", "tvl", "protocol", "onchain")):
            return SignalType.ONCHAIN.value
        if any(token in name for token in ("sentiment", "social", "twitter", "x_")):
            return SignalType.SENTIMENT.value

        if isinstance(value, dict):
            text = json.dumps(value, ensure_ascii=False).lower()
            if any(token in text for token in ("event_type", "sentiment", "signal_score", "latency_minutes")):
                return SignalType.NEWS.value
            if any(token in text for token in ("tvl", "active_address", "onchain", "protocol")):
                return SignalType.ONCHAIN.value
        return SignalType.PRICE.value

    def _extract_symbol_from_item(
        self,
        *,
        item: dict[str, Any],
        requested_symbols: list[str],
        fallback: str,
    ) -> str:
        """从返回项中提取最可能的 symbol。"""

        direct_symbol = item.get("symbol") or item.get("base") or item.get("coin") or item.get("token")
        if isinstance(direct_symbol, str) and direct_symbol.strip():
            return direct_symbol.strip().upper()

        candidates: list[str] = []
        currencies = item.get("currencies")
        if isinstance(currencies, list):
            for value in currencies:
                if isinstance(value, str) and value.strip():
                    candidates.append(value.strip().upper())
                elif isinstance(value, dict):
                    token = value.get("code") or value.get("symbol") or value.get("currency")
                    if isinstance(token, str) and token.strip():
                        candidates.append(token.strip().upper())
        elif isinstance(currencies, str) and currencies.strip():
            candidates.extend(token.strip().upper() for token in currencies.split(",") if token.strip())

        requested = {symbol.upper() for symbol in requested_symbols}
        for candidate in candidates:
            if candidate in requested:
                return candidate

        if candidates:
            return candidates[0]
        return fallback.upper()

    def _extract_published_at(self, item: dict[str, Any]) -> str | None:
        """从返回项提取发布时间并统一为 ISO 字符串。"""

        for key in ("published_at", "created_at", "updated_at", "timestamp", "time"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, (int, float)):
                return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
        return None

    def _normalize_raw_rows(self, task_id: str, rows: list[dict[str, Any]]) -> list[RawSignal]:
        """将 MCP 响应映射为统一 RawSignal 结构。"""

        normalized: list[RawSignal] = []
        for item in rows:
            symbol = str(item.get("symbol", "UNKNOWN")).upper()
            source = str(item.get("source", "mcp"))
            raw_type = str(item.get("signal_type", "news")).lower()
            if raw_type not in {member.value for member in SignalType}:
                raw_type = SignalType.NEWS.value

            published_at = datetime.now(timezone.utc)
            raw_published = item.get("published_at")
            if isinstance(raw_published, str):
                try:
                    published_at = datetime.fromisoformat(raw_published.replace("Z", "+00:00"))
                except Exception:
                    published_at = datetime.now(timezone.utc)

            normalized.append(
                RawSignal(
                    symbol=symbol,
                    source=source,
                    signal_type=SignalType(raw_type),
                    value=item.get("value", item),
                    raw_ref=str(item.get("raw_ref", item.get("url", source))),
                    published_at=published_at,
                    metadata={"task_id": task_id, "raw": item, **(item.get("metadata") or {})},
                )
            )

        return normalized
