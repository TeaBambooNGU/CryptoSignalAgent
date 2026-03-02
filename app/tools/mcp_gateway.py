"""官方 MCP 网关。

仅负责：
- 按配置连接 MCP server（stdio/sse/streamable_http）。
- 发现工具 schema。
- 执行已经过规则引擎过滤的工具调用。

不负责：
- 规划策略。
- 规则决策。
- 收敛判停。
"""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from mcp import ClientSession, types
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.exceptions import McpError

from app.config.logging import get_logger
from app.config.settings import Settings
from app.models.schemas import RawSignal, SignalType

logger = get_logger(__name__)

URL_PATTERN = re.compile(r"https?://\S+")


@dataclass(slots=True)
class MCPServerSpec:
    name: str
    transport: str
    url: str = ""
    command: str = ""
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    cwd: str | None = None
    tool_allowlist: tuple[str, ...] = ()


@dataclass(slots=True)
class MCPExecutionResult:
    rows: list[dict[str, Any]]
    successes: list[dict[str, Any]]
    failures: list[dict[str, Any]]
    errors: list[str]


class OfficialMCPGateway:
    """官方 MCP 调用适配层。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def load_server_specs(self) -> list[MCPServerSpec]:
        specs: list[MCPServerSpec] = []
        for index, item in enumerate(self.settings.mcp_servers, start=1):
            name = str(item.get("name", f"server-{index}")).strip() or f"server-{index}"
            transport = str(item.get("transport", "streamable_http")).strip().lower() or "streamable_http"
            args_raw = item.get("args", [])
            args = tuple(str(value) for value in args_raw) if isinstance(args_raw, list) else ()
            env_raw = item.get("env")
            env = {str(k): str(v) for k, v in env_raw.items()} if isinstance(env_raw, dict) else None
            tool_allowlist_raw = item.get("tool_allowlist", [])
            tool_allowlist = tuple(str(name) for name in tool_allowlist_raw) if isinstance(tool_allowlist_raw, list) else ()
            specs.append(
                MCPServerSpec(
                    name=name,
                    transport=transport,
                    url=str(item.get("url", "")),
                    command=str(item.get("command", "")),
                    args=args,
                    env=env,
                    cwd=str(item.get("cwd")) if item.get("cwd") else None,
                    tool_allowlist=tool_allowlist,
                )
            )
        return specs

    def discover_tools(self) -> tuple[dict[str, list[types.Tool]], list[str]]:
        """发现所有 MCP 服务工具。"""

        specs = self.load_server_specs()
        if not specs:
            return {}, ["未配置 MCP_SERVERS，跳过实时采集"]

        tool_map: dict[str, list[types.Tool]] = {}
        errors: list[str] = []
        for spec in specs:
            try:
                tools = asyncio.run(self._discover_tools_for_server(spec))
                if spec.tool_allowlist:
                    allow = set(spec.tool_allowlist)
                    tools = [tool for tool in tools if tool.name in allow]
                tool_map[spec.name] = tools
            except Exception as exc:  # pragma: no cover - 线上诊断分支
                detail = self._build_exception_error_detail(exc)
                message = f"MCP 工具发现失败: {spec.name} {self._safe_json(detail)}"
                errors.append(message)
                logger.exception("%s", message)

        return tool_map, errors

    async def _discover_tools_for_server(self, spec: MCPServerSpec) -> list[types.Tool]:
        async with self._open_session(spec) as session:
            listed = await session.list_tools()
            return listed.tools

    def execute_calls(
        self,
        *,
        task_id: str,
        symbols: list[str],
        calls: list[dict[str, Any]],
    ) -> MCPExecutionResult:
        """执行过滤后的工具调用。"""

        if not calls:
            return MCPExecutionResult(rows=[], successes=[], failures=[], errors=[])

        specs = {spec.name: spec for spec in self.load_server_specs()}
        grouped_calls: dict[str, list[dict[str, Any]]] = {}
        failures: list[dict[str, Any]] = []
        errors: list[str] = []

        for call in calls:
            server = str(call.get("server", "")).strip()
            if not server or server not in specs:
                failure = {
                    "server": server,
                    "tool_name": str(call.get("tool_name", "")).strip(),
                    "arguments": call.get("arguments", {}),
                    "call_signature": str(call.get("call_signature", "")),
                    "deterministic": True,
                    "reason": "unknown_server",
                    "error_detail": {"status_code": 400, "error_message": "unknown server"},
                }
                failures.append(failure)
                errors.append(f"MCP 执行失败: unknown server `{server}`")
                continue
            grouped_calls.setdefault(server, []).append(call)

        rows: list[dict[str, Any]] = []
        successes: list[dict[str, Any]] = []

        for server_name, call_items in grouped_calls.items():
            spec = specs[server_name]
            try:
                server_rows, server_successes, server_failures, server_errors = asyncio.run(
                    self._execute_calls_for_server(
                        spec=spec,
                        task_id=task_id,
                        symbols=symbols,
                        calls=call_items,
                    )
                )
                rows.extend(server_rows)
                successes.extend(server_successes)
                failures.extend(server_failures)
                errors.extend(server_errors)
            except Exception as exc:  # pragma: no cover - 线上诊断分支
                detail = self._build_exception_error_detail(exc)
                failures.extend(
                    {
                        "server": server_name,
                        "tool_name": str(call.get("tool_name", "")).strip(),
                        "arguments": call.get("arguments", {}),
                        "call_signature": str(call.get("call_signature", "")),
                        "deterministic": self._is_deterministic_error_detail(detail),
                        "reason": "server_exception",
                        "error_detail": detail,
                    }
                    for call in call_items
                )
                errors.append(f"MCP 服务失败: {server_name} {self._safe_json(detail)}")

        return MCPExecutionResult(rows=rows, successes=successes, failures=failures, errors=errors)

    async def _execute_calls_for_server(
        self,
        *,
        spec: MCPServerSpec,
        task_id: str,
        symbols: list[str],
        calls: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
        rows: list[dict[str, Any]] = []
        successes: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        errors: list[str] = []

        async with self._open_session(spec) as session:
            for call in calls:
                tool_name = str(call.get("tool_name", "")).strip()
                arguments = call.get("arguments", {})
                if not isinstance(arguments, dict):
                    arguments = {}
                reason = str(call.get("reason", "")).strip()
                call_signature = str(call.get("call_signature", "")).strip()

                try:
                    result = await session.call_tool(tool_name, arguments=arguments)
                except Exception as exc:
                    detail = self._build_exception_error_detail(exc)
                    failure = {
                        "server": spec.name,
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "call_signature": call_signature,
                        "reason": reason,
                        "deterministic": self._is_deterministic_error_detail(detail),
                        "error_detail": detail,
                    }
                    failures.append(failure)
                    errors.append(f"MCP 工具异常: {spec.name}/{tool_name} {self._safe_json(detail)}")
                    continue

                if result.isError:
                    detail = self._extract_tool_error_detail(result)
                    failure = {
                        "server": spec.name,
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "call_signature": call_signature,
                        "reason": reason,
                        "deterministic": self._is_deterministic_error_detail(detail),
                        "error_detail": detail,
                    }
                    failures.append(failure)
                    errors.append(f"MCP 工具返回错误: {spec.name}/{tool_name} {self._safe_json(detail)}")
                    continue

                extracted_rows = self._extract_rows_from_tool_result(
                    result=result,
                    server_name=spec.name,
                    tool_name=tool_name,
                    symbols=symbols,
                    task_id=task_id,
                )
                rows.extend(extracted_rows)
                success_status = "success_zero_rows" if len(extracted_rows) == 0 else "success"
                successes.append(
                    {
                        "server": spec.name,
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "call_signature": call_signature,
                        "reason": reason,
                        "rows": len(extracted_rows),
                        "status": success_status,
                    }
                )

        return rows, successes, failures, errors

    def normalize_rows_to_signals(self, *, task_id: str, rows: list[dict[str, Any]]) -> list[RawSignal]:
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

            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            normalized.append(
                RawSignal(
                    symbol=symbol,
                    source=source,
                    signal_type=SignalType(raw_type),
                    value=item.get("value", item),
                    raw_ref=str(item.get("raw_ref", item.get("url", source))),
                    published_at=published_at,
                    metadata={"task_id": task_id, "raw": item, **metadata},
                )
            )
        return normalized

    @asynccontextmanager
    async def _open_session(self, spec: MCPServerSpec):
        transport = spec.transport
        if transport == "stdio":
            if not spec.command:
                raise ValueError(f"stdio MCP 服务缺少 command: {spec.name}")
            params = StdioServerParameters(command=spec.command, args=list(spec.args), env=spec.env, cwd=spec.cwd)
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

        if not spec.url:
            raise ValueError(f"streamable_http MCP 服务缺少 url: {spec.name}")
        async with streamablehttp_client(spec.url, timeout=20, sse_read_timeout=120) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    def _extract_rows_from_tool_result(
        self,
        *,
        result: types.CallToolResult,
        server_name: str,
        tool_name: str,
        symbols: list[str],
        task_id: str,
    ) -> list[dict[str, Any]]:
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

        return [
            self._item_to_row(
                item=item,
                server_name=server_name,
                tool_name=tool_name,
                symbols=symbols,
                task_id=task_id,
            )
            for item in extracted_items
        ]

    def _flatten_item(self, data: Any) -> list[Any]:
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
        *,
        item: Any,
        server_name: str,
        tool_name: str,
        symbols: list[str],
        task_id: str,
    ) -> dict[str, Any]:
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

    def _infer_signal_type(self, *, tool_name: str, server_name: str, value: Any) -> str:
        name = f"{server_name} {tool_name}".lower()
        if any(token in name for token in ("news", "digest", "headline", "article", "rss")):
            return SignalType.NEWS.value
        if any(token in name for token in ("chain", "tvl", "protocol", "onchain")):
            return SignalType.ONCHAIN.value
        if any(token in name for token in ("sentiment", "social", "twitter", "x_")):
            return SignalType.SENTIMENT.value
        if isinstance(value, dict):
            text = json.dumps(value, ensure_ascii=False).lower()
            if any(token in text for token in ("event_type", "signal_score", "news")):
                return SignalType.NEWS.value
            if any(token in text for token in ("tvl", "onchain", "active_address")):
                return SignalType.ONCHAIN.value
        return SignalType.PRICE.value

    def _extract_symbol_from_item(self, *, item: dict[str, Any], requested_symbols: list[str], fallback: str) -> str:
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

        requested = {symbol.upper() for symbol in requested_symbols}
        for candidate in candidates:
            if candidate in requested:
                return candidate
        if candidates:
            return candidates[0]
        return fallback.upper()

    def _extract_published_at(self, item: dict[str, Any]) -> str | None:
        for key in ("published_at", "created_at", "updated_at", "timestamp", "time"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, (int, float)):
                return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
        return None

    def _extract_tool_error_detail(self, result: types.CallToolResult) -> dict[str, Any]:
        detail: dict[str, Any] = {"is_error": True}
        if result.structuredContent is not None:
            detail["structured_content"] = result.structuredContent
            self._merge_status_body_from_payload(detail, result.structuredContent)

        text_blocks: list[str] = []
        for block in result.content:
            if not isinstance(block, types.TextContent):
                continue
            text = block.text.strip()
            if not text:
                continue
            text_blocks.append(text)
            parsed = self._try_parse_json(text)
            if parsed is not None:
                self._merge_status_body_from_payload(detail, parsed)

        if text_blocks:
            detail["content_text"] = text_blocks
        if "status_code" not in detail and "response_body" not in detail:
            detail["error_message"] = text_blocks[0] if text_blocks else "unknown_error"
        return detail

    def _build_exception_error_detail(self, exc: BaseException) -> dict[str, Any]:
        detail: dict[str, Any] = {
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
        if isinstance(exc, ExceptionGroup):
            children = [self._build_exception_error_detail(item) for item in exc.exceptions[:8]]
            detail["sub_exceptions"] = children
            for child in children:
                if "status_code" in child and "status_code" not in detail:
                    detail["status_code"] = child["status_code"]
                if "response_body" in child and "response_body" not in detail:
                    detail["response_body"] = child["response_body"]

        if isinstance(exc, McpError):
            detail["mcp_error_code"] = exc.error.code
            detail["mcp_error_message"] = exc.error.message
            detail["mcp_error_data"] = exc.error.data
            self._merge_status_body_from_payload(detail, exc.error.data)

        response = getattr(exc, "response", None)
        if response is not None:
            status_code = getattr(response, "status_code", None)
            if status_code is not None:
                detail["status_code"] = status_code
            try:
                body = response.text
            except Exception:
                body = None
            if body:
                detail["response_body"] = body

        if "status_code" not in detail:
            status_code = getattr(exc, "status_code", None)
            if status_code is not None:
                detail["status_code"] = status_code

        if "response_body" not in detail:
            body = getattr(exc, "body", None)
            if body:
                detail["response_body"] = body

        return detail

    def _merge_status_body_from_payload(self, detail: dict[str, Any], payload: Any, depth: int = 0) -> None:
        if depth > 6:
            return
        if isinstance(payload, dict):
            for key, value in payload.items():
                lowered = str(key).lower().strip()
                if lowered in {"status", "status_code", "http_status", "http_status_code"} and "status_code" not in detail:
                    parsed = self._coerce_status_code(value)
                    if parsed is not None:
                        detail["status_code"] = parsed
                if lowered in {"response", "response_body", "body", "raw_body", "error_body"} and "response_body" not in detail:
                    detail["response_body"] = value
                if isinstance(value, (dict, list)):
                    self._merge_status_body_from_payload(detail, value, depth + 1)
            return

        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, (dict, list)):
                    self._merge_status_body_from_payload(detail, item, depth + 1)

    def _coerce_status_code(self, value: Any) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            digits = "".join(ch for ch in value if ch.isdigit())
            if digits:
                try:
                    return int(digits)
                except ValueError:
                    return None
        return None

    def _is_deterministic_error_detail(self, detail: dict[str, Any]) -> bool:
        status_code = self._coerce_status_code(detail.get("status_code"))
        if status_code is not None and 400 <= status_code < 500 and status_code != 429:
            return True

        message_parts: list[str] = []
        for key in ("error_message", "message", "response_body", "content_text"):
            value = detail.get(key)
            if value is None:
                continue
            message_parts.append(str(value))
        message = " | ".join(message_parts).lower()
        deterministic_tokens = (
            "invalid parameter",
            "invalid params",
            "validation error",
            "input should be",
            "missing required",
            "unknown argument",
            "not in enum",
            "protocol not found",
            "coin not found",
            "symbol not found",
            "bad request",
        )
        return any(token in message for token in deterministic_tokens)

    def _safe_json(self, value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            return str(value)


__all__ = ["MCPServerSpec", "MCPExecutionResult", "OfficialMCPGateway"]
