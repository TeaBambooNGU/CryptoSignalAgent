"""MCP 全量巡检脚本。

目标：
1) 访问已配置 MCP server。
2) 打印并落盘每次工具调用的入参和原始响应。
3) 不脱敏、不摘要、不截断，完整保留原始数据，便于排查。
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from typing import Any

from mcp import McpError, types

# 允许使用 `uv run python scripts/init_milvus.py` 直接运行。
# 直接运行脚本时，Python 默认只把 `scripts/` 加入 sys.path，
# 因此这里显式补充项目根目录，确保可导入 `app` 包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import Settings
from app.graph.mcp_subgraph import MCPSignalSubgraphRunner

# 固定执行配置（无 CLI 参数）
QUERY = "请提供 BTC 与 ETH 的最新市场、链上与新闻数据"
SYMBOLS = ["BTC", "ETH"]
CALL_ALL_TOOLS = True
MAX_TOOLS_PER_SERVER = 5  # 仅在 CALL_ALL_TOOLS=False 时生效


def _to_jsonable(value: Any) -> Any:
    """将对象转换为可 JSON 序列化结构（不做裁剪）。"""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]

    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(by_alias=True, mode="json", exclude_none=False)
        except TypeError:
            dumped = value.model_dump()
        return _to_jsonable(dumped)

    return str(value)


def _serialize_exception(exc: Exception) -> dict[str, Any]:
    """序列化异常，尽量保留 HTTP 状态码与响应体。"""

    payload: dict[str, Any] = {
        "type": type(exc).__name__,
        "message": str(exc),
        "repr": repr(exc),
    }

    if isinstance(exc, McpError):
        payload["mcp_error"] = _to_jsonable(exc.error)

    response = getattr(exc, "response", None)
    if response is not None:
        response_payload: dict[str, Any] = {
            "status_code": getattr(response, "status_code", None),
            "headers": dict(getattr(response, "headers", {})),
        }
        try:
            response_payload["text"] = response.text
        except Exception as response_exc:  # pragma: no cover - 防御性兜底
            response_payload["text_error"] = f"{type(response_exc).__name__}: {response_exc}"
        payload["http_response"] = response_payload

    request = getattr(exc, "request", None)
    if request is not None:
        payload["http_request"] = {
            "method": getattr(request, "method", None),
            "url": str(getattr(request, "url", "")),
            "headers": dict(getattr(request, "headers", {})),
        }

    if hasattr(exc, "__dict__"):
        extra_attrs: dict[str, Any] = {}
        for key, value in exc.__dict__.items():
            if key in {"response", "request"}:
                continue
            extra_attrs[str(key)] = _to_jsonable(value)
        if extra_attrs:
            payload["attributes"] = extra_attrs

    return payload


def _write_json_block(handle, title: str, data: Any) -> None:
    """写入 markdown JSON 代码块。"""

    handle.write(f"#### {title}\n\n")
    handle.write("```json\n")
    handle.write(json.dumps(_to_jsonable(data), ensure_ascii=False, indent=2))
    handle.write("\n```\n\n")
    handle.flush()


def _build_inspect_arguments(tool: types.Tool) -> dict[str, Any]:
    """巡检脚本内置参数猜测，仅用于尽量提高工具可调用率。"""

    schema = tool.inputSchema if isinstance(tool.inputSchema, dict) else {}
    properties = schema.get("properties", {}) if isinstance(schema.get("properties", {}), dict) else {}
    required = schema.get("required", []) or []
    now = datetime.now(timezone.utc)

    def suggest(name: str, prop_schema: dict[str, Any]) -> Any | None:
        lowered = name.lower()
        enums = prop_schema.get("enum")
        if isinstance(enums, list) and enums:
            return enums[0]
        if "default" in prop_schema:
            return prop_schema["default"]
        if lowered in {"vs_currency", "currency"}:
            return "usd"
        if lowered in {"id", "coin_id"}:
            return "bitcoin"
        if lowered in {"ids", "coin_ids"}:
            return "bitcoin,ethereum"
        if lowered in {"symbol"}:
            return "btc"
        if lowered in {"symbols"}:
            return "btc,eth"
        if lowered in {"query", "keyword", "search"}:
            return QUERY
        if lowered in {"from", "start"}:
            return int((now - timedelta(hours=24)).timestamp())
        if lowered in {"to", "end"}:
            return int(now.timestamp())
        if lowered in {"page", "offset"}:
            return 1
        if lowered in {"per_page", "limit", "count"}:
            return 10
        if lowered == "date":
            return now.strftime("%Y-%m-%d")
        return None

    arguments: dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        schema_item = prop_schema if isinstance(prop_schema, dict) else {}
        value = suggest(str(prop_name), schema_item)
        if value is not None:
            arguments[str(prop_name)] = value

    for req_name in required:
        req_text = str(req_name)
        if req_text in arguments:
            continue
        value = suggest(req_text, {})
        if value is not None:
            arguments[req_text] = value

    return arguments


async def _inspect_server(
    *,
    client,
    server_name: str,
    spec,
    output_handle,
) -> None:
    """检查单个 server，逐个调用工具并落盘。"""

    output_handle.write(f"## Server: {server_name}\n\n")
    output_handle.write(f"- transport: `{spec.get('transport', '')}`\n")
    if spec.get("url"):
        output_handle.write(f"- url: `{spec['url']}`\n")
    if spec.get("command"):
        output_handle.write(f"- command: `{spec['command']}`\n")
    output_handle.write("\n")
    output_handle.flush()

    try:
        async with client.session(server_name) as session:
            tools_result = await session.list_tools()
            tools = list(tools_result.tools)
            _write_json_block(output_handle, "tools/list", tools_result)

            if CALL_ALL_TOOLS:
                tools_to_call = tools
            else:
                tools_to_call = tools[:MAX_TOOLS_PER_SERVER]

            output_handle.write(f"### Calls ({len(tools_to_call)})\n\n")
            output_handle.flush()

            for index, tool in enumerate(tools_to_call, start=1):
                arguments = _build_inspect_arguments(tool)
                output_handle.write(f"### Tool #{index}: `{tool.name}`\n\n")
                _write_json_block(
                    output_handle,
                    "request",
                    {
                        "server": server_name,
                        "tool": tool.name,
                        "arguments": arguments,
                    },
                )
                try:
                    result = await session.call_tool(tool.name, arguments=arguments)
                    _write_json_block(output_handle, "response", result)
                except Exception as exc:
                    _write_json_block(output_handle, "error", _serialize_exception(exc))
    except Exception as exc:
        _write_json_block(output_handle, "server_error", _serialize_exception(exc))

    output_handle.write("\n---\n\n")
    output_handle.flush()


async def main() -> None:
    settings = Settings.from_env()
    connections = MCPSignalSubgraphRunner.build_connections_from_settings(settings.mcp_servers)
    if not connections:
        raise SystemExit("未配置 MCP_SERVERS，无法巡检。")

    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(connections, tool_name_prefix=True)
    specs = list(connections.items())

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = log_dir / f"mcp_inspect_{timestamp}.md"

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# MCP Inspect Report\n\n")
        handle.write(f"- generated_at_utc: `{datetime.now(timezone.utc).isoformat()}`\n")
        handle.write(f"- query: `{QUERY}`\n")
        handle.write(f"- symbols: `{','.join(SYMBOLS)}`\n")
        handle.write(f"- call_all_tools: `{CALL_ALL_TOOLS}`\n")
        handle.write(f"- server_count: `{len(specs)}`\n\n")
        handle.write("> 说明：本报告不做脱敏、不做摘要、不做截断，保留原始请求与响应。\n\n")
        handle.write("---\n\n")
        handle.flush()

        for server_name, spec in specs:
            print(f"[inspect] start server={server_name}")
            await _inspect_server(client=client, server_name=server_name, spec=spec, output_handle=handle)
            print(f"[inspect] done server={server_name}")

    print(f"[inspect] report saved: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
