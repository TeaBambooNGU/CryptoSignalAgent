"""MCP 信号采集执行器（create_agent 版本）。

设计目标：
- 使用 LangGraph/LangChain 官方 `create_agent` 进行工具调用。
- 去除原 MCP 子图的规则引擎与多轮判停复杂逻辑。
- 保持对外 `run(...)` 接口稳定，供主工作流复用。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Callable, TypedDict

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage

from app.config.logging import get_logger, log_context

logger = get_logger(__name__)

URL_PATTERN = re.compile(r"https?://\\S+")
VALID_SIGNAL_TYPES = {"price", "news", "sentiment", "onchain"}


class MCPSubgraphState(TypedDict, total=False):
    """MCP 采集结果状态（精简版）。"""

    raw_signals: list[dict[str, Any]]
    errors: list[str]
    mcp_tools_count: int
    mcp_termination_reason: str


class MCPSignalSubgraphRunner:
    """基于 `create_agent` 的 MCP 采集执行器。"""

    def __init__(
        self,
        *,
        llm: BaseChatModel,
        mcp_connections: dict[str, dict[str, Any]],
        max_rounds: int = 4,
        mcp_client_factory: Callable[..., Any] | None = None,
        agent_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.llm = llm
        self.mcp_connections = mcp_connections
        self.max_rounds = max(1, int(max_rounds))
        if mcp_client_factory is None:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            mcp_client_factory = MultiServerMCPClient
        self.mcp_client_factory = mcp_client_factory
        self.agent_factory = agent_factory or create_agent

    @staticmethod
    def build_connections_from_settings(mcp_servers: tuple[dict[str, Any], ...]) -> dict[str, dict[str, Any]]:
        """将 settings 的 MCP 配置转换为 MultiServerMCPClient 入参。"""

        connections: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(mcp_servers, start=1):
            name = str(item.get("name", f"server-{index}")).strip() or f"server-{index}"
            transport = str(item.get("transport", "streamable_http")).strip().lower() or "streamable_http"
            connection: dict[str, Any] = {"transport": transport}
            if transport == "stdio":
                connection["command"] = str(item.get("command", ""))
                args = item.get("args", [])
                connection["args"] = [str(value) for value in args] if isinstance(args, list) else []
                if isinstance(item.get("env"), dict):
                    connection["env"] = {str(k): str(v) for k, v in item["env"].items()}
                if item.get("cwd"):
                    connection["cwd"] = str(item["cwd"])
            else:
                connection["url"] = str(item.get("url", ""))
                if isinstance(item.get("headers"), dict):
                    connection["headers"] = {str(k): str(v) for k, v in item["headers"].items()}
            connections[name] = connection
        return connections

    async def arun(
        self,
        *,
        user_id: str,
        query: str,
        task_id: str,
        symbols: list[str],
        errors: list[str] | None,
    ) -> MCPSubgraphState:
        """执行 MCP 采集并返回原始信号。"""

        merged_errors = list(errors or [])
        catalog, tools, discovery_errors = await self._discover_tools_with_official_client()
        merged_errors.extend(discovery_errors)

        if not tools:
            return {
                "raw_signals": [],
                "errors": self._dedupe_strings(merged_errors),
                "mcp_tools_count": 0,
                "mcp_termination_reason": "no_tools",
            }

        try:
            agent = self._build_agent(tools=tools)
            agent_input = {
                "messages": [
                    {
                        "role": "user",
                        "content": self._build_agent_user_prompt(
                            user_id=user_id,
                            query=query,
                            symbols=symbols,
                            task_id=task_id,
                            tool_catalog=catalog,
                        ),
                    }
                ]
            }
            if not callable(getattr(agent, "ainvoke", None)):
                raise RuntimeError("MCP agent 不支持 ainvoke，无法保证 async-only 工具调用链路")
            # MCP adapter tools通常只支持异步调用，强制走 ainvoke 避免 sync tool path。
            result = await agent.ainvoke(agent_input)
        except Exception as exc:
            merged_errors.append(f"MCP Agent 执行失败: {self._safe_json(self._build_exception_error_detail(exc))}")
            return {
                "raw_signals": [],
                "errors": self._dedupe_strings(merged_errors),
                "mcp_tools_count": len(tools),
                "mcp_termination_reason": "agent_failed",
            }

        messages = result.get("messages", []) if isinstance(result, dict) else []
        with log_context(component="mcp.agent", task_id=task_id, user_id=user_id):
            tool_rows = self._extract_rows_from_messages(messages=messages, symbols=symbols, task_id=task_id)
            payload, payload_error = self._parse_agent_payload(messages)
            payload_rows = payload.get("raw_signals", []) if isinstance(payload, dict) else []
            payload_errors = payload.get("errors", []) if isinstance(payload, dict) else []

            rows = self._merge_rows(
                tool_rows=tool_rows,
                payload_rows=payload_rows,
                symbols=symbols,
                task_id=task_id,
            )

            if payload_error and not rows:
                merged_errors.append(f"Agent 结论解析失败: {payload_error}")
            merged_errors.extend(payload_errors)

            logger.info("MCP Agent 采集完成 tools=%s raw_signals=%s", len(tools), len(rows))
            return {
                "raw_signals": rows,
                "errors": self._dedupe_strings(merged_errors),
                "mcp_tools_count": len(tools),
                "mcp_termination_reason": "agent_completed",
            }

    def run(
        self,
        *,
        user_id: str,
        query: str,
        task_id: str,
        symbols: list[str],
        errors: list[str] | None,
    ) -> MCPSubgraphState:
        """同步包装器（兼容旧调用方）。"""

        return asyncio.run(
            self.arun(
                user_id=user_id,
                query=query,
                task_id=task_id,
                symbols=symbols,
                errors=errors,
            )
        )

    async def _discover_tools_with_official_client(
        self,
    ) -> tuple[dict[str, list[dict[str, Any]]], list[Any], list[str]]:
        if not self.mcp_connections:
            return {}, [], ["未配置 MCP_SERVERS，跳过实时采集"]

        errors: list[str] = []
        tool_catalog: dict[str, list[dict[str, Any]]] = {server: [] for server in self.mcp_connections.keys()}
        runtime_tools: list[Any] = []

        try:
            client = self.mcp_client_factory(self.mcp_connections, tool_name_prefix=True)
            tools = await client.get_tools()
        except Exception as exc:
            detail = self._build_exception_error_detail(exc)
            errors.append(f"MCP 工具发现失败: {self._safe_json(detail)}")
            return tool_catalog, runtime_tools, errors

        for tool in tools:
            full_name = str(getattr(tool, "name", "")).strip()
            server, tool_name = self._split_prefixed_tool_name(full_name)
            if not server or not tool_name:
                errors.append(f"MCP 工具名无法解析 server/tool: {full_name}")
                continue
            schema = self._extract_tool_schema(tool)
            tool_catalog.setdefault(server, []).append(
                {
                    "name": tool_name,
                    "description": str(getattr(tool, "description", ""))[:220],
                    "schema": schema,
                }
            )
            runtime_tools.append(tool)

        return tool_catalog, runtime_tools, errors

    def _build_agent(self, *, tools: list[Any]):
        return self.agent_factory(
            model=self.llm,
            tools=tools,
            system_prompt=self._build_agent_system_prompt(),
            name="mcp_signal_agent",
        )

    def _build_agent_system_prompt(self) -> str:
        return (
            "你是加密市场信号采集 Agent。"
            "你可以自行选择和调用 MCP 工具。"
            "优先返回结构化数据，并尽量覆盖用户关注 symbols。"
            "最后一条回复必须是 JSON 对象，不要使用 markdown 代码块。"
            "JSON 格式为 {\"raw_signals\": [...], \"errors\": [...]}。"
            "raw_signals 每项包含 symbol/source/signal_type/value/raw_ref/published_at/metadata 字段。"
            "signal_type 仅允许 price/news/sentiment/onchain。"
        )

    def _build_agent_user_prompt(
        self,
        *,
        user_id: str,
        query: str,
        symbols: list[str],
        task_id: str,
        tool_catalog: dict[str, list[dict[str, Any]]],
    ) -> str:
        symbols_text = ",".join(symbols) if symbols else "BTC,ETH"
        return (
            f"user_id={user_id}\n"
            f"task_id={task_id}\n"
            f"query={query}\n"
            f"target_symbols={symbols_text}\n"
            f"tool_call_budget_hint={self.max_rounds}\n\n"
            "请先按需调用工具，再输出最终 JSON。"
            "如果某个工具失败，把错误摘要写入 errors。"
            "不要输出解释性文本，只输出最终 JSON。\n\n"
            f"可用工具目录：{self._safe_json(tool_catalog)}"
        )

    def _split_prefixed_tool_name(self, full_name: str) -> tuple[str, str]:
        for server in sorted(self.mcp_connections.keys(), key=len, reverse=True):
            prefix = f"{server}_"
            if full_name.startswith(prefix):
                return server, full_name[len(prefix) :]
        return "", ""

    def _extract_tool_schema(self, tool: Any) -> dict[str, Any]:
        schema = getattr(tool, "args_schema", None)
        if isinstance(schema, dict):
            return schema
        input_schema = getattr(tool, "inputSchema", None)
        if isinstance(input_schema, dict):
            return input_schema
        return {"type": "object", "properties": {}, "required": []}

    def _parse_agent_payload(self, messages: list[Any]) -> tuple[dict[str, Any], str | None]:
        final_text = ""
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                final_text = self._extract_text(message.content).strip()
                break

        if not final_text:
            return {}, "Agent 未输出最终文本"

        payload = self._try_parse_json_object(final_text)
        if not isinstance(payload, dict):
            return {}, "Agent 输出不是 JSON 对象"

        raw_signals = payload.get("raw_signals", [])
        normalized_rows = raw_signals if isinstance(raw_signals, list) else []

        payload_errors = payload.get("errors", [])
        normalized_errors = [str(item) for item in payload_errors if str(item).strip()] if isinstance(payload_errors, list) else []

        return {"raw_signals": normalized_rows, "errors": normalized_errors}, None

    def _extract_rows_from_messages(self, *, messages: list[Any], symbols: list[str], task_id: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, ToolMessage):
                continue

            full_tool_name = str(getattr(message, "name", "")).strip()
            server, tool_name = self._split_prefixed_tool_name(full_tool_name)
            if not server:
                server = "unknown"
            if not tool_name:
                tool_name = full_tool_name or "unknown_tool"

            extracted_items: list[Any] = []
            artifact = getattr(message, "artifact", None)
            if isinstance(artifact, dict):
                structured = artifact.get("structured_content")
                if structured is not None:
                    extracted_items.extend(self._flatten_item(structured))

            extracted_items.extend(self._extract_items_from_content(message.content))
            for item in extracted_items:
                rows.append(
                    self._item_to_row(
                        item=item,
                        server_name=server,
                        tool_name=tool_name,
                        symbols=symbols,
                        task_id=task_id,
                    )
                )
        return rows

    def _merge_rows(
        self,
        *,
        tool_rows: list[dict[str, Any]],
        payload_rows: list[Any],
        symbols: list[str],
        task_id: str,
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        hash_set: set[str] = set()

        for item in [*tool_rows, *payload_rows]:
            row = self._normalize_signal_row(item=item, symbols=symbols, task_id=task_id)
            if row is None:
                continue
            signal_hash = self.build_signal_hash(row)
            if signal_hash in hash_set:
                continue
            hash_set.add(signal_hash)
            merged.append(row)
        return merged

    def _normalize_signal_row(self, *, item: Any, symbols: list[str], task_id: str) -> dict[str, Any] | None:
        if not isinstance(item, dict):
            return None

        symbol = str(item.get("symbol") or (symbols[0] if symbols else "BTC")).upper()
        source = str(item.get("source") or "mcp:agent")
        signal_type = str(item.get("signal_type") or "price").lower()
        if signal_type not in VALID_SIGNAL_TYPES:
            signal_type = "price"

        value = item.get("value", item)
        raw_ref = str(item.get("raw_ref") or item.get("url") or item.get("link") or "mcp://agent")
        published_at = self._normalize_published_at(item.get("published_at"))
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}

        return {
            "symbol": symbol,
            "source": source,
            "signal_type": signal_type,
            "value": value,
            "raw_ref": raw_ref,
            "published_at": published_at,
            "task_id": str(item.get("task_id") or task_id),
            "metadata": metadata,
        }

    def _extract_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
                        continue
                chunks.append(str(block))
            return "".join(chunks)
        if isinstance(content, dict):
            text = content.get("text")
            if isinstance(text, str):
                return text
        return str(content)

    def _try_parse_json_object(self, text: str) -> dict[str, Any] | None:
        stripped = text.strip()
        if not stripped:
            return None
        stripped = re.sub(r"^```(?:json)?\\s*|\\s*```$", "", stripped, flags=re.IGNORECASE).strip()
        decoder = json.JSONDecoder()
        for index, char in enumerate(stripped):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(stripped[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    def _extract_items_from_content(self, content: Any) -> list[Any]:
        extracted: list[Any] = []
        if content is None:
            return extracted
        if isinstance(content, str):
            parsed = self._try_parse_json(content.strip())
            if parsed is not None:
                extracted.extend(self._flatten_item(parsed))
            elif content.strip():
                extracted.append(content.strip())
            return extracted
        if isinstance(content, list):
            for block in content:
                extracted.extend(self._extract_items_from_content(block))
            return extracted
        if isinstance(content, dict):
            block_type = str(content.get("type", "")).strip().lower()
            if block_type == "text" and isinstance(content.get("text"), str):
                text = content["text"].strip()
                parsed = self._try_parse_json(text)
                if parsed is not None:
                    extracted.extend(self._flatten_item(parsed))
                elif text:
                    extracted.append(text)
                return extracted
            extracted.append(content)
            return extracted
        extracted.append(content)
        return extracted

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
            return "news"
        if any(token in name for token in ("chain", "tvl", "protocol", "onchain")):
            return "onchain"
        if any(token in name for token in ("sentiment", "social", "twitter", "x_")):
            return "sentiment"
        if isinstance(value, dict):
            text = json.dumps(value, ensure_ascii=False).lower()
            if any(token in text for token in ("event_type", "signal_score", "news")):
                return "news"
            if any(token in text for token in ("tvl", "onchain", "active_address")):
                return "onchain"
        return "price"

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

    def _normalize_published_at(self, value: Any) -> str:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
        return datetime.now(timezone.utc).isoformat()

    def _build_exception_error_detail(self, exc: BaseException) -> dict[str, Any]:
        detail: dict[str, Any] = {
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
        status_code = self._coerce_status_code(getattr(exc, "status_code", None))
        if status_code is not None:
            detail["status_code"] = status_code
        if isinstance(exc, ExceptionGroup):
            children = [self._build_exception_error_detail(item) for item in exc.exceptions[:8]]
            detail["sub_exceptions"] = children
        return detail

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

    def _safe_json(self, value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            return str(value)

    def _dedupe_strings(self, values: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            deduped.append(text)
        return deduped

    @staticmethod
    def _canonical_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def build_signal_hash(signal: dict[str, Any]) -> str:
        core = {
            "symbol": signal.get("symbol"),
            "source": signal.get("source"),
            "signal_type": signal.get("signal_type"),
            "value": signal.get("value"),
            "metadata": signal.get("metadata", {}),
        }
        payload = MCPSignalSubgraphRunner._canonical_json(core)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["MCPSubgraphState", "MCPSignalSubgraphRunner"]
