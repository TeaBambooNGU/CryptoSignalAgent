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
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config.logging import get_logger, log_context
from app.config.settings import Settings
from app.models.schemas import RawSignal, SignalType

logger = get_logger(__name__)

URL_PATTERN = re.compile(r"https?://\\S+")
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
                for attempt in Retrying(
                    stop=stop_after_attempt(3),
                    wait=wait_exponential(multiplier=1, min=1, max=16),
                    retry=retry_if_exception_type(Exception),
                    reraise=True,
                ):
                    with attempt:
                        rows = asyncio.run(
                            self._collect_from_server(
                                spec=spec,
                                task_id=task_id,
                                query=query,
                                symbols=symbols,
                            )
                        )
                with log_context(component="mcp.collect"):
                    logger.info("MCP 服务完成 server=%s rows=%s", spec.name, len(rows))
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
    ) -> list[dict[str, Any]]:
        """连接单个 MCP 服务并采集数据。"""

        async with self._open_session(spec) as session:
            tools_result = await session.list_tools()
            selected_tools = self._select_tools(spec=spec, tools=tools_result.tools)
            if not selected_tools:
                return []

            with log_context(component="mcp.collect"):
                logger.info("工具筛选完成 server=%s selected_tools=%s", spec.name, len(selected_tools))
            rows: list[dict[str, Any]] = []
            for tool in selected_tools[: spec.max_tools_per_server]:
                arguments = self._build_tool_arguments(tool=tool, query=query, symbols=symbols)
                try:
                    result = await session.call_tool(tool.name, arguments=arguments)
                except Exception as exc:
                    logger.warning(
                        "MCP 工具调用失败 server=%s tool=%s error=%s",
                        spec.name,
                        tool.name,
                        type(exc).__name__,
                    )
                    continue

                rows.extend(
                    self._post_process_rows(
                        rows=self._extract_rows_from_tool_result(
                            result=result,
                            server_name=spec.name,
                            tool_name=tool.name,
                            symbols=symbols,
                            task_id=task_id,
                        ),
                        symbols=symbols,
                    )
                )
                with log_context(component="mcp.collect"):
                    logger.info("工具调用完成 server=%s tool=%s rows=%s", spec.name, tool.name, len(rows))

            return rows

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

    def _select_tools(self, spec: MCPServerSpec, tools: list[types.Tool]) -> list[types.Tool]:
        """按服务器配置或关键词策略筛选工具。"""

        if spec.tool_allowlist:
            allow = set(spec.tool_allowlist)
            return [tool for tool in tools if tool.name in allow]

        keywords = (
            "market",
            "price",
            "coin",
            "token",
            "chain",
            "protocol",
            "news",
            "trend",
            "tvl",
            "recent",
            "funding",
            "onchain",
            "search",
        )
        selected = [tool for tool in tools if any(keyword in tool.name.lower() for keyword in keywords)]
        if selected:
            return selected
        return tools[:3]

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
        if lowered in {"symbol", "ticker"}:
            return (symbols[0] if symbols else "BTC").lower()
        if lowered in {"symbols", "tickers"}:
            return ",".join(symbols).lower() if symbols else "btc,eth"
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

        if isinstance(item, dict):
            symbol = str(
                item.get("symbol")
                or item.get("base")
                or item.get("coin")
                or item.get("token")
                or symbol
            ).upper()
            raw_ref = str(item.get("url") or item.get("link") or item.get("source_url") or "")
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
            "published_at": datetime.now(timezone.utc).isoformat(),
            "task_id": task_id,
            "metadata": {"tool": tool_name},
        }

    def _infer_signal_type(self, tool_name: str, server_name: str, value: Any) -> str:
        """根据工具名与值内容推断信号类型。"""

        name = f"{server_name} {tool_name}".lower()
        if any(token in name for token in ("news", "rss", "headline", "article")):
            return SignalType.NEWS.value
        if any(token in name for token in ("chain", "tvl", "protocol", "onchain")):
            return SignalType.ONCHAIN.value
        if any(token in name for token in ("sentiment", "social", "twitter", "x_")):
            return SignalType.SENTIMENT.value

        if isinstance(value, dict):
            text = json.dumps(value, ensure_ascii=False).lower()
            if any(token in text for token in ("tvl", "active_address", "onchain", "protocol")):
                return SignalType.ONCHAIN.value
        return SignalType.PRICE.value

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
