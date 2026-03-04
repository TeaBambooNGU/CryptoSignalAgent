"""MCP Server 可用性验证脚本。

验证内容：
1. 能否建立 MCP 会话并 initialize。
2. 能否 list_tools。
3. 至少成功 call 一个工具。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from typing import Any

from mcp import ClientSession, types
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import Settings
from app.graph.mcp_subgraph import MCPSignalSubgraphRunner


@dataclass(slots=True)
class ServerSpec:
    name: str
    transport: str
    url: str = ""
    command: str = ""
    args: tuple[str, ...] = ()


def load_specs() -> list[ServerSpec]:
    settings = Settings.from_env()
    connections = MCPSignalSubgraphRunner.build_connections_from_settings(settings.mcp_servers)
    specs: list[ServerSpec] = []
    for name, item in connections.items():
        if not isinstance(item, dict):
            continue
        specs.append(
            ServerSpec(
                name=str(name),
                transport=str(item.get("transport", "streamable_http")).lower(),
                url=str(item.get("url", "")),
                command=str(item.get("command", "")),
                args=tuple(str(arg) for arg in item.get("args", []) if isinstance(arg, (str, int, float))),
            )
        )
    return specs


def suggest_arguments(tool: types.Tool) -> dict[str, Any]:
    schema = tool.inputSchema if isinstance(tool.inputSchema, dict) else {}
    properties = schema.get("properties", {})
    required = schema.get("required", []) or []

    now = datetime.now(timezone.utc)

    def pick(name: str) -> Any | None:
        lowered = name.lower()
        if lowered in {"vs_currency", "currency"}:
            return "usd"
        if lowered in {"id", "coin_id", "token_id", "asset_id", "protocol"}:
            return "bitcoin"
        if lowered in {"ids", "coin_ids", "token_ids"}:
            return "bitcoin,ethereum"
        if lowered in {"symbol", "ticker"}:
            return "btc"
        if lowered in {"symbols", "tickers"}:
            return "btc,eth"
        if lowered in {"query", "search", "keyword"}:
            return "bitcoin market"
        if lowered in {"site"}:
            return "coindesk"
        if lowered in {"limit", "per_page", "count", "size"}:
            return 10
        if lowered in {"page", "offset"}:
            return 1
        if lowered in {"from", "start", "from_timestamp"}:
            return int((now - timedelta(hours=24)).timestamp())
        if lowered in {"to", "end", "to_timestamp"}:
            return int(now.timestamp())
        if lowered == "date":
            return now.strftime("%d-%m-%Y")
        return None

    args: dict[str, Any] = {}
    for name in properties:
        value = pick(name)
        if value is not None:
            args[name] = value

    for name in required:
        if name not in args:
            value = pick(name)
            if value is not None:
                args[name] = value

    return args


async def verify_one(spec: ServerSpec) -> tuple[bool, str]:
    try:
        if spec.transport == "stdio":
            params = StdioServerParameters(command=spec.command, args=list(spec.args))
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await _verify_session(session)

        if spec.transport == "sse":
            async with sse_client(spec.url, timeout=20, sse_read_timeout=120) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await _verify_session(session)

        async with streamablehttp_client(spec.url, timeout=20, sse_read_timeout=120) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await _verify_session(session)
    except Exception as exc:
        return False, f"连接失败: {type(exc).__name__}: {exc}"


async def _verify_session(session: ClientSession) -> tuple[bool, str]:
    tools = await session.list_tools()
    if not tools.tools:
        return False, "list_tools 成功，但无可用工具"

    sorted_tools = sorted(
        tools.tools,
        key=lambda tool: len((tool.inputSchema or {}).get("required", []))
        if isinstance(tool.inputSchema, dict)
        else 0,
    )

    for tool in sorted_tools[:8]:
        args = suggest_arguments(tool)
        try:
            result = await session.call_tool(tool.name, arguments=args)
        except Exception:
            continue
        if result.isError:
            continue

        sample = ""
        for block in result.content[:1]:
            if isinstance(block, types.TextContent):
                sample = block.text[:120].replace("\n", " ")
        return True, f"工具调用成功: {tool.name} | sample={sample}"

    return False, "list_tools 成功，但工具调用均失败"


async def main() -> None:
    specs = load_specs()
    if not specs:
        print("未配置 MCP Servers（请检查 .mcp.json），无法验证")
        return

    print(f"开始验证 {len(specs)} 个 MCP 服务器...\n")
    success_count = 0
    for spec in specs:
        ok, message = await verify_one(spec)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {spec.name}: {message}")
        if ok:
            success_count += 1

    print(f"\n验证完成: {success_count}/{len(specs)} 可用")
    if success_count != len(specs):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
