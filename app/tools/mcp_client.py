"""标准 MCP 数据采集客户端。

V1 目标：
- 使用 MCP 官方协议（stdio / streamable HTTP / SSE）调用外部工具。
- 将 tool 返回结果统一映射为内部 RawSignal。
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from mcp import ClientSession, types
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.exceptions import McpError

from app.agents.llm.base import BaseLLMClient
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
PLANNER_MAX_DESCRIPTION_CHARS = 220
PLANNER_MAX_PROPERTIES_PER_TOOL = 16
PLANNER_MAX_RETRY_ATTEMPTS = 2
PLANNER_MAX_HISTORY_ITEMS = 6
PLANNER_MAX_FEEDBACK_ITEMS = 8
SERVER_MAX_RETRY_ATTEMPTS = 3
SERVER_REPLAN_MAX_ROUNDS = 2
SERVER_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
REPLAN_REPEAT_5XX_THRESHOLD = 2


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


@dataclass(slots=True)
class MCPCollectResult:
    """MCP 采集结果（含诊断信息）。"""

    signals: list[RawSignal]
    errors: list[str]
    failures: list[dict[str, Any]]
    successes: list[dict[str, Any]]


class MCPClient:
    """MCP 采集适配层。"""

    def __init__(self, settings: Settings, llm_client: BaseLLMClient | None = None) -> None:
        self.settings = settings
        self.llm_client = llm_client

    def collect_signals(
        self,
        task_id: str,
        query: str,
        symbols: list[str],
    ) -> tuple[list[RawSignal], list[str]]:
        """通过标准 MCP 协议采集原始信号。"""

        result = self.collect_signals_detailed(
            task_id=task_id,
            query=query,
            symbols=symbols,
            planning_context=None,
        )
        return result.signals, result.errors

    def collect_signals_detailed(
        self,
        *,
        task_id: str,
        query: str,
        symbols: list[str],
        planning_context: dict[str, Any] | None,
    ) -> MCPCollectResult:
        """通过标准 MCP 协议采集原始信号，并返回诊断信息。"""

        errors: list[str] = []
        failures: list[dict[str, Any]] = []
        successes: list[dict[str, Any]] = []
        specs = self._load_server_specs()
        with log_context(component="mcp.collect"):
            logger.info("MCP 采集开始 servers=%s symbols=%s", len(specs), symbols or ["BTC"])
        if not specs:
            errors.append("未配置 MCP_SERVERS，跳过实时采集")
            return MCPCollectResult(signals=[], errors=errors, failures=failures, successes=successes)

        historical_corrections = planning_context.get("historical_corrections", []) if planning_context else []
        server_failures_raw = planning_context.get("server_failures", {}) if planning_context else {}
        server_failures: dict[str, list[dict[str, Any]]] = {}
        if isinstance(server_failures_raw, dict):
            for server_name, items in server_failures_raw.items():
                if isinstance(items, list):
                    server_failures[str(server_name)] = [item for item in items if isinstance(item, dict)]

        raw_rows: list[dict[str, Any]] = []
        for spec in specs:
            server_feedback = [item for item in server_failures.get(spec.name, []) if isinstance(item, dict)]
            exception_signature_counts: dict[str, int] = {}
            attempt_context: dict[str, Any] = {}
            try:
                rows: list[dict[str, Any]] = []
                server_errors: list[str] = []
                server_failures_result: list[dict[str, Any]] = []
                server_successes_result: list[dict[str, Any]] = []
                last_exc: Exception | None = None

                for attempt in range(1, SERVER_MAX_RETRY_ATTEMPTS + 1):
                    try:
                        rows, server_errors, server_failures_result, server_successes_result = asyncio.run(
                            self._collect_from_server(
                                spec=spec,
                                task_id=task_id,
                                query=query,
                                symbols=symbols,
                                historical_corrections=historical_corrections,
                                failure_feedback=server_feedback,
                                attempt_context=attempt_context,
                            )
                        )
                        break
                    except Exception as exc:
                        last_exc = exc
                        error_detail = self._build_exception_error_detail(exc)
                        failure_tool_name = str(attempt_context.get("tool_name") or "__server__")
                        failure_arguments = (
                            attempt_context.get("arguments")
                            if isinstance(attempt_context.get("arguments"), dict)
                            else {}
                        )
                        failure_reason = str(attempt_context.get("reason") or "server_exception")
                        transient_failure = {
                            "server": spec.name,
                            "tool_name": failure_tool_name,
                            "arguments": failure_arguments,
                            "reason": failure_reason,
                            "error_detail": error_detail,
                            "deterministic": self._is_deterministic_error_detail(error_detail),
                        }
                        if self._should_replan_on_failure(item=transient_failure):
                            server_feedback.append(transient_failure)

                        signature = self._build_failure_signature(transient_failure)
                        if signature:
                            exception_signature_counts[signature] = exception_signature_counts.get(signature, 0) + 1

                        retryable = self._is_retryable_server_exception(exc)
                        repeated_5xx = (
                            signature
                            and exception_signature_counts.get(signature, 0) >= REPLAN_REPEAT_5XX_THRESHOLD
                            and self._is_failure_status_5xx(error_detail)
                        )

                        if not retryable or attempt >= SERVER_MAX_RETRY_ATTEMPTS:
                            raise

                        if repeated_5xx:
                            logger.warning(
                                "MCP 服务重试前注入重复5xx反馈 server=%s signature=%s attempt=%s",
                                spec.name,
                                self._truncate_log_text(signature, max_text_chars=180),
                                attempt + 1,
                            )
                        else:
                            logger.warning("MCP 服务重试 server=%s attempt=%s", spec.name, attempt + 1)

                        backoff_seconds = min(16, 2 ** (attempt - 1))
                        time.sleep(backoff_seconds)
                else:
                    if last_exc is not None:
                        raise last_exc

                errors.extend(server_errors)
                failures.extend(server_failures_result)
                successes.extend(server_successes_result)
                with log_context(component="mcp.collect"):
                    logger.info("MCP 服务完成 server=%s rows=%s errors=%s", spec.name, len(rows), len(server_errors))
                raw_rows.extend(rows)
            except Exception as exc:
                error_detail = self._build_exception_error_detail(exc)
                failure_tool_name = str(attempt_context.get("tool_name") or "__server__")
                failure_arguments = (
                    attempt_context.get("arguments")
                    if isinstance(attempt_context.get("arguments"), dict)
                    else {}
                )
                failure_reason = str(attempt_context.get("reason") or "server_collect")
                logger.exception(
                    "MCP 服务采集失败: %s detail=%s",
                    spec.name,
                    self._format_log_payload(error_detail),
                )
                errors.append(f"MCP 服务失败: {spec.name} {self._format_log_payload(error_detail)}")
                failures.append(
                    {
                        "server": spec.name,
                        "tool_name": failure_tool_name,
                        "arguments": failure_arguments,
                        "reason": failure_reason,
                        "error_detail": error_detail,
                        "deterministic": self._is_deterministic_error_detail(error_detail),
                    }
                )

        normalized = self._normalize_raw_rows(task_id=task_id, rows=raw_rows)
        with log_context(component="mcp.collect"):
            logger.info("MCP 采集结束 raw_rows=%s normalized=%s errors=%s", len(raw_rows), len(normalized), len(errors))
        return MCPCollectResult(signals=normalized, errors=errors, failures=failures, successes=successes)

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
        historical_corrections: list[dict[str, Any]],
        failure_feedback: list[dict[str, Any]],
        attempt_context: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]], list[dict[str, Any]]]:
        """连接单个 MCP 服务并采集数据。"""

        if attempt_context is not None:
            attempt_context.clear()
            attempt_context.update({"tool_name": "__server__", "arguments": {}, "reason": "session_open"})

        async with self._open_session(spec) as session:
            tools_result = await session.list_tools()
            rows: list[dict[str, Any]] = []
            errors: list[str] = []
            failures: list[dict[str, Any]] = []
            successes: list[dict[str, Any]] = []
            planning_feedback: list[dict[str, Any]] = [item for item in failure_feedback if isinstance(item, dict)]
            seen_call_signatures: set[str] = set()

            for planning_round in range(SERVER_REPLAN_MAX_ROUNDS):
                tool_calls = self._plan_tool_calls_with_llm(
                    spec=spec,
                    tools=tools_result.tools,
                    query=query,
                    symbols=symbols,
                    errors=errors,
                    historical_corrections=historical_corrections,
                    failure_feedback=planning_feedback,
                )
                if not tool_calls:
                    return [], [f"MCP 无可用工具: {spec.name}"], failures, successes

                with log_context(component="mcp.collect"):
                    logger.info(
                        "LLM 规划完成 server=%s planned_calls=%s round=%s",
                        spec.name,
                        len(tool_calls),
                        planning_round + 1,
                    )

                round_failures: list[dict[str, Any]] = []
                executed_calls = 0
                for tool, arguments, rationale in tool_calls:
                    call_signature = self._build_call_signature(tool_name=tool.name, arguments=arguments)
                    if call_signature in seen_call_signatures:
                        errors.append(
                            f"MCP 规划重复调用已跳过: {spec.name}/{tool.name} {self._format_log_payload(self._sanitize_for_log(arguments))}"
                        )
                        continue
                    seen_call_signatures.add(call_signature)

                    executed_calls += 1
                    safe_arguments = self._sanitize_for_log(arguments)
                    if attempt_context is not None:
                        attempt_context.update(
                            {
                                "tool_name": tool.name,
                                "arguments": arguments,
                                "reason": rationale,
                            }
                        )
                    with log_context(component="mcp.collect"):
                        logger.info(
                            "MCP 工具调用开始 server=%s tool=%s arguments=%s rationale=%s round=%s",
                            spec.name,
                            tool.name,
                            self._format_log_payload(safe_arguments),
                            self._truncate_log_text(rationale or ""),
                            planning_round + 1,
                        )
                    try:
                        result = await session.call_tool(tool.name, arguments=arguments)
                    except Exception as exc:
                        error_detail = self._build_exception_error_detail(exc)
                        error_index = len(errors)
                        errors.append(
                            "MCP 工具异常: "
                            f"{spec.name}/{tool.name} {self._format_log_payload(error_detail)}"
                        )
                        logger.warning(
                            "MCP 工具调用失败 server=%s tool=%s arguments=%s rationale=%s round=%s error=%s",
                            spec.name,
                            tool.name,
                            self._format_log_payload(safe_arguments),
                            self._truncate_log_text(rationale or ""),
                            planning_round + 1,
                            self._format_log_payload(error_detail),
                        )
                        failure_item = {
                            "server": spec.name,
                            "tool_name": tool.name,
                            "arguments": arguments,
                            "reason": rationale,
                            "error_detail": error_detail,
                            "deterministic": self._is_deterministic_error_detail(error_detail),
                            "_error_index": error_index,
                            "_resolved": False,
                        }
                        failures.append(failure_item)
                        round_failures.append(failure_item)
                        continue

                    if result.isError:
                        error_detail = self._extract_tool_error_detail(result)
                        error_index = len(errors)
                        errors.append(
                            "MCP 工具返回错误: "
                            f"{spec.name}/{tool.name} {self._format_log_payload(error_detail)}"
                        )
                        with log_context(component="mcp.collect"):
                            logger.warning(
                                "MCP 工具返回错误 server=%s tool=%s arguments=%s rationale=%s round=%s error=%s",
                                spec.name,
                                tool.name,
                                self._format_log_payload(safe_arguments),
                                self._truncate_log_text(rationale or ""),
                                planning_round + 1,
                                self._format_log_payload(error_detail),
                            )
                        failure_item = {
                            "server": spec.name,
                            "tool_name": tool.name,
                            "arguments": arguments,
                            "reason": rationale,
                            "error_detail": error_detail,
                            "deterministic": self._is_deterministic_error_detail(error_detail),
                            "_error_index": error_index,
                            "_resolved": False,
                        }
                        failures.append(failure_item)
                        round_failures.append(failure_item)
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
                            "工具调用完成 server=%s tool=%s tool_rows=%s total_rows=%s rationale=%s round=%s result=%s",
                            spec.name,
                            tool.name,
                            len(tool_rows),
                            len(rows),
                            self._truncate_log_text(rationale or ""),
                            planning_round + 1,
                            self._format_log_payload(
                                self._summarize_tool_result(
                                    result=result,
                                    extracted_rows=len(extracted_rows),
                                    post_processed_rows=len(tool_rows),
                                )
                            ),
                        )
                    successes.append(
                        {
                            "server": spec.name,
                            "tool_name": tool.name,
                            "arguments": arguments,
                            "reason": rationale,
                            "rows": len(tool_rows),
                        }
                    )
                    self._mark_failures_resolved_by_success(failures=failures, success_tool_name=tool.name)

                if planning_round + 1 >= SERVER_REPLAN_MAX_ROUNDS:
                    break
                if executed_calls == 0:
                    break

                replan_feedback = self._build_replan_feedback(
                    previous_failures=planning_feedback,
                    current_failures=round_failures,
                )
                if not replan_feedback:
                    break
                planning_feedback = self._merge_feedback_items(existing=planning_feedback, incoming=replan_feedback)
                logger.info(
                    "触发单服务重规划 server=%s round=%s feedback=%s",
                    spec.name,
                    planning_round + 2,
                    len(replan_feedback),
                )

            final_errors, final_failures = self._finalize_failures(errors=errors, failures=failures)
            if len(final_failures) < len(failures):
                logger.info(
                    "折叠已修复失败 server=%s resolved=%s remaining=%s",
                    spec.name,
                    len(failures) - len(final_failures),
                    len(final_failures),
                )
            return rows, final_errors, final_failures, successes

    def _mark_failures_resolved_by_success(self, *, failures: list[dict[str, Any]], success_tool_name: str) -> None:
        """同一工具后续调用成功时，标记此前确定性失败为已修复。"""

        for item in failures:
            if not isinstance(item, dict):
                continue
            if item.get("_resolved"):
                continue
            if not bool(item.get("deterministic")):
                continue
            if str(item.get("tool_name", "")).strip() != success_tool_name:
                continue
            item["_resolved"] = True

    def _finalize_failures(
        self,
        *,
        errors: list[str],
        failures: list[dict[str, Any]],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """剔除同轮已修复失败，避免把已解决问题继续向上游放大。"""

        removed_error_indexes: set[int] = set()
        output_failures: list[dict[str, Any]] = []
        for item in failures:
            if not isinstance(item, dict):
                continue
            if item.get("_resolved"):
                error_index = item.get("_error_index")
                if isinstance(error_index, int):
                    removed_error_indexes.add(error_index)
                continue
            payload = dict(item)
            payload.pop("_resolved", None)
            payload.pop("_error_index", None)
            output_failures.append(payload)

        output_errors = [text for index, text in enumerate(errors) if index not in removed_error_indexes]
        return output_errors, output_failures

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

    def _plan_tool_calls_with_llm(
        self,
        *,
        spec: MCPServerSpec,
        tools: list[types.Tool],
        query: str,
        symbols: list[str],
        errors: list[str],
        historical_corrections: list[dict[str, Any]],
        failure_feedback: list[dict[str, Any]],
    ) -> list[tuple[types.Tool, dict[str, Any], str]]:
        """使用 LLM 规划当前 server 的工具调用与参数。"""

        if self.llm_client is None:
            raise RuntimeError("MCPClient 未注入 llm_client，无法执行工具规划")

        if not tools:
            return []

        candidate_tools = self._filter_tools_for_planner(spec=spec, tools=tools)
        if not candidate_tools:
            return []

        base_user_prompt = self._build_planner_user_prompt(
            server_name=spec.name,
            query=query,
            symbols=symbols,
            tools=candidate_tools,
            max_calls=spec.max_tools_per_server,
            historical_corrections=historical_corrections,
            failure_feedback=failure_feedback,
        )
        planner_error: Exception | None = None
        planner_text = ""

        for attempt in range(PLANNER_MAX_RETRY_ATTEMPTS):
            user_prompt = base_user_prompt
            if attempt > 0 and planner_error is not None:
                user_prompt = self._build_planner_repair_user_prompt(
                    original_prompt=base_user_prompt,
                    previous_output=planner_text,
                    error_message=str(planner_error),
                )

            planner_text = self.llm_client.generate(
                system_prompt=self._build_planner_system_prompt(),
                user_prompt=user_prompt,
                metadata={
                    "component": "mcp_tool_planner",
                    "server": spec.name,
                    "attempt": attempt + 1,
                },
            )
            try:
                plan_payload = self._parse_planner_payload(planner_text)
                return self._validate_and_normalize_plan(
                    spec=spec,
                    tools=candidate_tools,
                    plan_payload=plan_payload,
                    errors=errors,
                )
            except ValueError as exc:
                planner_error = exc

        raise ValueError(f"LLM 规划失败: {spec.name} {planner_error}")

    def _filter_tools_for_planner(self, *, spec: MCPServerSpec, tools: list[types.Tool]) -> list[types.Tool]:
        """根据 allowlist 过滤候选工具，不做规则推断。"""

        if not spec.tool_allowlist:
            return tools
        allow = set(spec.tool_allowlist)
        return [tool for tool in tools if tool.name in allow]

    def _build_planner_system_prompt(self) -> str:
        """构建 MCP 工具规划器系统提示词。"""

        return (
            "你是严谨的 MCP 工具调用规划器。"
            "你的任务是从候选工具中选择最相关工具，并为每个工具生成可执行 arguments。"
            "你必须严格遵守工具 schema，避免虚构字段。"
            "你只能输出一个 JSON 对象，且必须能被 json.loads 直接解析。"
            "禁止输出 markdown 代码块、前后缀说明、注释、自然语言解释。"
            '输出顶层必须是 {"calls":[...]}。'
        )

    def _build_planner_user_prompt(
        self,
        *,
        server_name: str,
        query: str,
        symbols: list[str],
        tools: list[types.Tool],
        max_calls: int,
        historical_corrections: list[dict[str, Any]],
        failure_feedback: list[dict[str, Any]],
    ) -> str:
        """构建 MCP 工具规划器用户提示词。"""

        tool_summaries = self._summarize_tools_for_planning(tools)
        symbols_text = ",".join(symbols) if symbols else "BTC,ETH"
        tools_text = json.dumps(tool_summaries, ensure_ascii=False, indent=2)
        correction_text = json.dumps(
            self._summarize_historical_corrections(server_name=server_name, corrections=historical_corrections),
            ensure_ascii=False,
            indent=2,
        )
        feedback_text = json.dumps(
            self._summarize_failure_feedback(feedback=failure_feedback),
            ensure_ascii=False,
            indent=2,
        )
        return (
            f"server={server_name}\n"
            f"user_query={query}\n"
            f"target_symbols={symbols_text}\n"
            f"max_calls={max_calls}\n\n"
            "请输出 JSON：\n"
            "{\n"
            '  "calls":[\n'
            "    {\n"
            '      "tool_name":"候选工具名",\n'
            '      "arguments":{"参数名":"参数值"},\n'
            '      "reason":"不超过30字"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "强约束：\n"
            "1) 仅可使用候选工具名；\n"
            "2) 必须补齐工具 required 字段；\n"
            "3) 不确定必填 path 参数时，不要选择该工具；\n"
            "4) 优先返回能直接覆盖目标 symbols 的工具；\n"
            "5) calls 数量必须在 1..max_calls。\n"
            "6) 若存在“最近失败反馈”，必须优先避免重复使用同一组失败参数。\n\n"
            "输出格式约束（必须同时满足）：\n"
            "A) 回复内容只能是 JSON 对象本体；\n"
            "B) 不要输出 ```json 或 ```；\n"
            "C) 不要输出任何解释性文字。\n\n"
            f"历史纠错经验:\n{correction_text}\n\n"
            f"最近失败反馈:\n{feedback_text}\n\n"
            f"候选工具:\n{tools_text}"
        )

    def _summarize_historical_corrections(
        self,
        *,
        server_name: str,
        corrections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """压缩长期纠错记忆，供 planner 参考。"""

        summarized: list[dict[str, Any]] = []
        for item in corrections[:PLANNER_MAX_HISTORY_ITEMS]:
            if not isinstance(item, dict):
                continue
            if str(item.get("server", "")).strip() not in {server_name, ""}:
                continue
            summarized.append(
                {
                    "server": str(item.get("server", "")),
                    "failed_tool": str(item.get("failed_tool", "")),
                    "failed_arguments": self._sanitize_for_log(item.get("failed_arguments", {})),
                    "error_signature": self._truncate_log_text(str(item.get("error_signature", ""))),
                    "fixed_tool": str(item.get("fixed_tool", "")),
                    "fixed_arguments": self._sanitize_for_log(item.get("fixed_arguments", {})),
                }
            )
        return summarized

    def _summarize_failure_feedback(self, *, feedback: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """压缩最近失败反馈，避免 prompt 过长。"""

        summarized: list[dict[str, Any]] = []
        for item in feedback[:PLANNER_MAX_FEEDBACK_ITEMS]:
            if not isinstance(item, dict):
                continue
            error_detail = item.get("error_detail", {})
            signature = self._build_error_signature(error_detail if isinstance(error_detail, dict) else {})
            summarized.append(
                {
                    "tool_name": str(item.get("tool_name", "")),
                    "arguments": self._sanitize_for_log(item.get("arguments", {})),
                    "error_signature": signature,
                }
            )
        return summarized

    def _build_planner_repair_user_prompt(
        self,
        *,
        original_prompt: str,
        previous_output: str,
        error_message: str,
    ) -> str:
        """当规划输出不合法时，要求 LLM 按错误原因自修复。"""

        return (
            "上一次输出未通过校验，请严格修复。\n"
            f"校验错误: {error_message}\n\n"
            "请重新输出符合约束的 JSON 对象，禁止输出额外解释。\n"
            "你的回复将被 json.loads 直接解析。\n"
            "禁止 markdown 代码块、禁止任何前后缀文本，只能输出 JSON 对象。\n"
            "如果某工具无法满足 required 字段，请直接删除该调用。\n\n"
            f"原始任务:\n{original_prompt}\n\n"
            f"上次输出:\n{previous_output}"
        )

    def _summarize_tools_for_planning(self, tools: list[types.Tool]) -> list[dict[str, Any]]:
        """压缩工具 schema，避免 planner prompt 过长。"""

        summaries: list[dict[str, Any]] = []
        for tool in tools:
            schema = tool.inputSchema if isinstance(tool.inputSchema, dict) else {}
            properties = schema.get("properties", {})
            required = schema.get("required", []) or []

            property_summaries: dict[str, Any] = {}
            required_names = [str(name) for name in required]
            ordered_property_names = list(dict.fromkeys([*required_names, *[str(name) for name in properties.keys()]]))
            if len(ordered_property_names) > PLANNER_MAX_PROPERTIES_PER_TOOL:
                ordered_property_names = ordered_property_names[:PLANNER_MAX_PROPERTIES_PER_TOOL]

            for prop_name in ordered_property_names:
                prop_schema = properties.get(prop_name, {})
                if not isinstance(prop_schema, dict):
                    property_summaries[str(prop_name)] = {}
                    continue
                item: dict[str, Any] = {}
                if "type" in prop_schema:
                    item["type"] = prop_schema["type"]
                enums = prop_schema.get("enum")
                if isinstance(enums, list) and enums:
                    item["enum"] = enums[:10]
                if "default" in prop_schema:
                    item["default"] = prop_schema["default"]
                if "description" in prop_schema:
                    item["description"] = self._truncate_log_text(
                        str(prop_schema["description"]),
                        max_text_chars=PLANNER_MAX_DESCRIPTION_CHARS,
                    )
                property_summaries[str(prop_name)] = item

            summaries.append(
                {
                    "name": tool.name,
                    "description": self._truncate_log_text(
                        tool.description or "",
                        max_text_chars=PLANNER_MAX_DESCRIPTION_CHARS,
                    ),
                    "required": [str(name) for name in required],
                    "properties": property_summaries,
                    "total_properties": len(properties),
                }
            )
        return summaries

    def _parse_planner_payload(self, planner_text: str) -> dict[str, Any]:
        """解析 LLM 输出的 JSON 规划结果。"""

        cleaned = planner_text.strip()
        if not cleaned:
            raise ValueError("LLM 规划输出为空")

        candidates: list[str] = [cleaned]
        stripped_fence = re.sub(r"^```(?:json)?\\s*|\\s*```$", "", cleaned, flags=re.IGNORECASE).strip()
        if stripped_fence and stripped_fence != cleaned:
            candidates.append(stripped_fence)

        decoder = json.JSONDecoder()
        text = stripped_fence or cleaned
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj

        for candidate in candidates:
            parsed = self._try_parse_json(candidate)
            if isinstance(parsed, dict):
                return parsed

        raise ValueError("LLM 规划输出不是有效 JSON 对象（仅允许 JSON object）")

    def _validate_and_normalize_plan(
        self,
        *,
        spec: MCPServerSpec,
        tools: list[types.Tool],
        plan_payload: dict[str, Any],
        errors: list[str],
    ) -> list[tuple[types.Tool, dict[str, Any], str]]:
        """校验 planner 结果并转换为可执行调用列表。"""

        calls = plan_payload.get("calls")
        if not isinstance(calls, list) or not calls:
            raise ValueError("LLM 规划缺少 calls 列表")

        tool_map = {tool.name: tool for tool in tools}
        normalized_calls: list[tuple[types.Tool, dict[str, Any], str]] = []
        seen_tools: set[str] = set()

        for item in calls:
            if not isinstance(item, dict):
                continue

            tool_name = str(item.get("tool_name", "")).strip()
            if not tool_name or tool_name not in tool_map:
                errors.append(f"MCP 规划忽略未知工具: {spec.name}/{tool_name or '<empty>'}")
                continue
            if tool_name in seen_tools:
                continue

            raw_arguments = item.get("arguments", {})
            if raw_arguments is None:
                raw_arguments = {}
            if not isinstance(raw_arguments, dict):
                errors.append(f"MCP 规划参数格式非法: {spec.name}/{tool_name}")
                continue

            tool = tool_map[tool_name]
            coerced_args, validation_errors = self._validate_arguments_against_schema(tool=tool, arguments=raw_arguments)
            if validation_errors:
                errors.extend(f"MCP 规划参数无效: {spec.name}/{tool_name} {message}" for message in validation_errors)
                continue

            reason = str(item.get("reason", "")).strip()
            normalized_calls.append((tool, coerced_args, reason))
            seen_tools.add(tool_name)

            if len(normalized_calls) >= spec.max_tools_per_server:
                break

        if not normalized_calls:
            raise ValueError(f"LLM 规划无有效调用: {spec.name}")
        return normalized_calls

    def _validate_arguments_against_schema(
        self,
        *,
        tool: types.Tool,
        arguments: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        """按 inputSchema 校验并规范化 arguments。"""

        schema = tool.inputSchema if isinstance(tool.inputSchema, dict) else {}
        properties = schema.get("properties", {}) if isinstance(schema.get("properties", {}), dict) else {}
        required = schema.get("required", []) or []

        normalized: dict[str, Any] = {}
        errors: list[str] = []

        for key, value in arguments.items():
            key_text = str(key)
            if properties and key_text not in properties:
                errors.append(f"{key_text}: unknown argument")
                continue
            prop_schema = properties.get(key_text, {})
            if not isinstance(prop_schema, dict):
                prop_schema = {}
            coerced, error = self._coerce_argument_value(value=value, schema=prop_schema)
            if error:
                errors.append(f"{key_text}: {error}")
                continue
            normalized[key_text] = coerced

        for req in required:
            req_name = str(req)
            value = normalized.get(req_name)
            if value is None or (isinstance(value, str) and not value.strip()):
                errors.append(f"missing required `{req_name}`")

        for key, prop_schema in properties.items():
            key_text = str(key)
            if key_text not in normalized or not isinstance(prop_schema, dict):
                continue
            enums = prop_schema.get("enum")
            if isinstance(enums, list) and enums and normalized[key_text] not in enums:
                errors.append(f"{key_text}: not in enum")

        return normalized, errors

    def _coerce_argument_value(self, *, value: Any, schema: dict[str, Any]) -> tuple[Any, str | None]:
        """按 schema 对 LLM 参数进行最小必要类型转换。"""

        value_type = schema.get("type")
        if value_type == "integer":
            if isinstance(value, bool):
                return value, "expected integer"
            if isinstance(value, int):
                return value, None
            if isinstance(value, float):
                return int(value), None
            if isinstance(value, str) and value.strip().lstrip("-").isdigit():
                return int(value.strip()), None
            return value, "expected integer"

        if value_type == "number":
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value, None
            if isinstance(value, str):
                try:
                    return float(value.strip()), None
                except ValueError:
                    return value, "expected number"
            return value, "expected number"

        if value_type == "boolean":
            if isinstance(value, bool):
                return value, None
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "1", "yes"}:
                    return True, None
                if lowered in {"false", "0", "no"}:
                    return False, None
            return value, "expected boolean"

        if value_type == "array":
            if isinstance(value, list):
                return value, None
            if isinstance(value, str):
                tokens = [token.strip() for token in value.split(",") if token.strip()]
                if tokens:
                    return tokens, None
            return value, "expected array"

        return value, None

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

    def _build_exception_error_detail(self, exc: BaseException, *, depth: int = 0) -> dict[str, Any]:
        """提取异常详情，避免丢失状态码与响应体。"""

        detail: dict[str, Any] = {
            "exception_type": type(exc).__name__,
            "message": self._truncate_log_text(str(exc), max_text_chars=None),
        }

        if isinstance(exc, ExceptionGroup):
            sub_details: list[dict[str, Any]] = []
            for index, sub_exc in enumerate(exc.exceptions[:ERROR_LOG_MAX_LIST_ITEMS]):
                sub_detail = self._build_exception_error_detail(sub_exc, depth=depth + 1)
                sub_detail["index"] = index
                sub_details.append(sub_detail)
                if "status_code" not in detail and "status_code" in sub_detail:
                    detail["status_code"] = sub_detail["status_code"]
                    detail["status_code_source"] = f"sub_exceptions[{index}]"
                if "response_body" not in detail and "response_body" in sub_detail:
                    detail["response_body"] = sub_detail["response_body"]
                    detail["response_body_source"] = f"sub_exceptions[{index}]"
            detail["sub_exceptions"] = sub_details
            if len(exc.exceptions) > ERROR_LOG_MAX_LIST_ITEMS:
                detail["sub_exceptions_truncated"] = len(exc.exceptions) - ERROR_LOG_MAX_LIST_ITEMS

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

    def _is_retryable_server_exception(self, exc: BaseException) -> bool:
        """判断服务级异常是否值得重试，避免确定性错误反复重放。"""

        if isinstance(exc, ExceptionGroup):
            if not exc.exceptions:
                return True
            return any(self._is_retryable_server_exception(item) for item in exc.exceptions)

        # 参数校验类错误通常是确定性失败，重试无收益。
        if isinstance(exc, (ValueError, TypeError)):
            return False

        status_code = self._coerce_status_code(self._find_attr_in_exception_tree(exc, "status_code"))
        if status_code is None:
            response = self._find_attr_in_exception_tree(exc, "response")
            if response is not None:
                status_code = self._coerce_status_code(getattr(response, "status_code", None))
        response_text = self._find_response_text_in_exception_tree(exc)
        if self._is_deterministic_error_message(response_text):
            return False

        if status_code is not None:
            # 4xx 代表请求参数/语义错误，重试无意义；429 保留重试。
            if 400 <= status_code < 500 and status_code != 429:
                return False
            return status_code in {429, 500, 502, 503, 504}

        # MCP 标准错误码：-3260x 多为请求参数错误。
        if isinstance(exc, McpError) and exc.error.code in {-32600, -32601, -32602}:
            return False

        return True

    def _build_call_signature(self, *, tool_name: str, arguments: dict[str, Any]) -> str:
        """构建工具调用签名，避免同轮重复执行同参数调用。"""

        try:
            args_text = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            args_text = str(arguments)
        return f"{tool_name}|args={args_text}"

    def _build_failure_signature(self, item: dict[str, Any]) -> str:
        """构建失败签名，用于识别重复失败模式。"""

        server = str(item.get("server", "")).strip()
        tool_name = str(item.get("tool_name", "")).strip()
        arguments = item.get("arguments", {})
        error_detail = item.get("error_detail", {})
        status_code = self._coerce_status_code(error_detail.get("status_code")) if isinstance(error_detail, dict) else None
        return f"{server}|{self._build_call_signature(tool_name=tool_name, arguments=arguments if isinstance(arguments, dict) else {})}|status={status_code or 'unknown'}"

    def _is_failure_status_5xx(self, detail: dict[str, Any]) -> bool:
        """判断失败是否属于 5xx 上游错误。"""

        status_code = self._coerce_status_code(detail.get("status_code"))
        if status_code is not None:
            return 500 <= status_code < 600

        body_text = ""
        for key in ("error_message", "message", "response_body"):
            value = detail.get(key)
            if value is None:
                continue
            body_text += f" {value}"
        lowered = body_text.lower()
        return "status=500" in lowered or "status=502" in lowered or "status=503" in lowered or "status=504" in lowered

    def _should_replan_on_failure(self, *, item: dict[str, Any]) -> bool:
        """判断失败是否可直接回灌给下一轮 planner。"""

        if bool(item.get("deterministic")):
            return True
        error_detail = item.get("error_detail", {})
        if not isinstance(error_detail, dict):
            return False
        status_code = self._coerce_status_code(error_detail.get("status_code"))
        if status_code is not None and status_code in SERVER_RETRYABLE_STATUS_CODES:
            return True
        return False

    def _build_replan_feedback(
        self,
        *,
        previous_failures: list[dict[str, Any]],
        current_failures: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """从失败集中挑选可用于重规划的反馈。"""

        if not current_failures:
            return []

        signature_counts: dict[str, int] = {}
        for item in [*previous_failures, *current_failures]:
            if not isinstance(item, dict):
                continue
            signature = self._build_failure_signature(item)
            signature_counts[signature] = signature_counts.get(signature, 0) + 1

        selected: list[dict[str, Any]] = []
        for item in current_failures:
            if not isinstance(item, dict):
                continue
            if bool(item.get("deterministic")):
                selected.append(item)
                continue
            error_detail = item.get("error_detail", {})
            if not isinstance(error_detail, dict):
                continue
            if not self._is_failure_status_5xx(error_detail):
                continue
            signature = self._build_failure_signature(item)
            if signature_counts.get(signature, 0) >= REPLAN_REPEAT_5XX_THRESHOLD:
                selected.append(item)

        return self._merge_feedback_items(existing=[], incoming=selected)

    def _merge_feedback_items(
        self,
        *,
        existing: list[dict[str, Any]],
        incoming: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """合并反馈并按失败签名去重，控制提示词体积。"""

        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in [*existing, *incoming]:
            if not isinstance(item, dict):
                continue
            signature = self._build_failure_signature(item)
            if signature in seen:
                continue
            seen.add(signature)
            merged.append(item)
            if len(merged) >= PLANNER_MAX_FEEDBACK_ITEMS:
                break
        return merged

    def _find_attr_in_exception_tree(self, exc: BaseException, attr_name: str, *, depth: int = 0) -> Any | None:
        """从异常树（含 ExceptionGroup）中查找属性。"""

        if depth > 8:
            return None
        value = getattr(exc, attr_name, None)
        if value is not None:
            return value
        if isinstance(exc, ExceptionGroup):
            for sub_exc in exc.exceptions:
                found = self._find_attr_in_exception_tree(sub_exc, attr_name, depth=depth + 1)
                if found is not None:
                    return found
        return None

    def _find_response_text_in_exception_tree(self, exc: BaseException, *, depth: int = 0) -> str:
        """从异常树中提取响应文本（若存在）。"""

        if depth > 8:
            return ""
        response = getattr(exc, "response", None)
        if response is not None:
            try:
                text = response.text
                if text:
                    return str(text)
            except Exception:
                pass
        body = getattr(exc, "body", None)
        if body:
            return str(body)
        if isinstance(exc, ExceptionGroup):
            for sub_exc in exc.exceptions:
                text = self._find_response_text_in_exception_tree(sub_exc, depth=depth + 1)
                if text:
                    return text
        return ""

    def _is_deterministic_error_message(self, message: str) -> bool:
        """基于错误文本判定是否为确定性错误。"""

        if not message:
            return False
        lowered = message.lower()
        deterministic_tokens = (
            "invalid parameter",
            "invalid params",
            "validation error",
            "pydantic",
            "input should be",
            "list_type",
            "missing required",
            "required parameter",
            "unknown argument",
            "not in enum",
            "protocol not found",
            "coin not found",
            "symbol not found",
            "bad request",
        )
        return any(token in lowered for token in deterministic_tokens)

    def _is_deterministic_error_detail(self, detail: dict[str, Any]) -> bool:
        """根据错误详情判断是否为确定性参数/语义错误。"""

        status_code = self._coerce_status_code(detail.get("status_code"))
        if status_code is not None and 400 <= status_code < 500 and status_code != 429:
            return True

        message_parts: list[str] = []
        for key in ("error_message", "message", "response_body"):
            value = detail.get(key)
            if value is None:
                continue
            message_parts.append(str(value))
        message = " | ".join(message_parts)
        return self._is_deterministic_error_message(message)

    def _build_error_signature(self, detail: dict[str, Any]) -> str:
        """构建可复用的错误签名，便于 LLM 下轮避坑。"""

        status_code = self._coerce_status_code(detail.get("status_code"))
        body = detail.get("response_body")
        message = detail.get("error_message") or detail.get("message") or ""
        body_text = ""
        if body is not None:
            body_text = str(body)
        core = f"status={status_code or 'unknown'} message={message} body={body_text}"
        return self._truncate_log_text(core, max_text_chars=220)

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
