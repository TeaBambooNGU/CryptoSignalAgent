"""MCP 信号采集子图。

设计目标：
- LLM 仅负责规划 `raw_plan_calls`。
- 规则引擎强制执行，仅执行 `filtered_plan`。
- 退出条件基于结构化状态，可证明收敛。
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.llm.base import BaseLLMClient
from app.config.logging import get_logger, log_context
from app.tools.mcp_gateway import OfficialMCPGateway

logger = get_logger(__name__)

PLANNER_MAX_CALLS = 16
PLANNER_MAX_FEEDBACK_ROUNDS = 2
PLANNER_MAX_FEEDBACK_ITEMS = 8
TECHNICAL_DEFAULTS: dict[str, Any] = {
    "page": 1,
    "page_size": 50,
    "limit": 50,
    "offset": 0,
    "timeout": 20,
}
UNKNOWN_TOOL_REASON = "tool_not_available"
MISSING_REQUIRED_REASON = "missing_required_fields"


class MCPSubgraphState(TypedDict, total=False):
    user_id: str
    query: str
    task_id: str
    symbols: list[str]
    errors: list[str]
    raw_signals: list[dict[str, Any]]

    mcp_round: int
    mcp_max_rounds: int
    transient_grace_used: bool

    mcp_raw_plan_calls: list[dict[str, Any]]
    mcp_filtered_plan_calls: list[dict[str, Any]]
    mcp_admissible_calls_count: int
    mcp_filtered_out_calls: list[dict[str, Any]]
    mcp_ban_reasons: list[str]
    mcp_active_constraints_summary: list[str]

    mcp_round_rows: list[dict[str, Any]]
    mcp_round_successes: list[dict[str, Any]]
    mcp_round_failures: list[dict[str, Any]]
    mcp_round_signal_docs: list[dict[str, Any]]
    mcp_failure_classes: dict[str, Any]
    mcp_rules: list[dict[str, Any]]
    mcp_new_rules_added: bool

    mcp_new_unique_signal_count: int
    mcp_signal_hash_set: list[str]
    mcp_raw_plan_hash_history: list[str]
    mcp_no_progress_streak: int

    mcp_should_continue: bool
    mcp_termination_reason: str

    mcp_tool_catalog: dict[str, list[dict[str, Any]]]
    mcp_tool_schemas: dict[str, dict[str, dict[str, Any]]]
    mcp_feedback_history: list[dict[str, Any]]
    mcp_plan_round_stats: list[dict[str, Any]]
    mcp_unique_progress_history: list[int]


class MCPSignalSubgraphRunner:
    """LangGraph MCP 子图执行器。"""

    def __init__(self, *, llm_client: BaseLLMClient, mcp_gateway: OfficialMCPGateway, max_rounds: int = 4) -> None:
        self.llm_client = llm_client
        self.mcp_gateway = mcp_gateway
        self.max_rounds = max(1, int(max_rounds))
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(MCPSubgraphState)

        builder.add_node("mcp_prepare", self.mcp_prepare)
        builder.add_node("mcp_plan", self.mcp_plan)
        builder.add_node("mcp_apply_rules", self.mcp_apply_rules)
        builder.add_node("mcp_tool_call", self.mcp_tool_call)
        builder.add_node("mcp_collect_round", self.mcp_collect_round)
        builder.add_node("mcp_classify_failures", self.mcp_classify_failures)
        builder.add_node("mcp_reflect_rules", self.mcp_reflect_rules)
        builder.add_node("mcp_should_continue", self.mcp_should_continue)
        builder.add_node("mcp_finalize", self.mcp_finalize)

        builder.add_edge(START, "mcp_prepare")
        builder.add_edge("mcp_prepare", "mcp_plan")
        builder.add_edge("mcp_plan", "mcp_apply_rules")
        builder.add_edge("mcp_apply_rules", "mcp_tool_call")
        builder.add_edge("mcp_tool_call", "mcp_collect_round")
        builder.add_edge("mcp_collect_round", "mcp_classify_failures")
        builder.add_edge("mcp_classify_failures", "mcp_reflect_rules")
        builder.add_edge("mcp_reflect_rules", "mcp_should_continue")
        builder.add_conditional_edges(
            "mcp_should_continue",
            self._route_after_should_continue,
            {
                "continue": "mcp_plan",
                "stop": "mcp_finalize",
            },
        )
        builder.add_edge("mcp_finalize", END)
        return builder.compile()

    def _route_after_should_continue(self, state: MCPSubgraphState) -> str:
        return "continue" if state.get("mcp_should_continue", False) else "stop"

    def run(
        self,
        *,
        user_id: str,
        query: str,
        task_id: str,
        symbols: list[str],
        errors: list[str] | None,
    ) -> MCPSubgraphState:
        initial_state: MCPSubgraphState = {
            "user_id": user_id,
            "query": query,
            "task_id": task_id,
            "symbols": symbols,
            "errors": list(errors or []),
            "raw_signals": [],
            "mcp_max_rounds": self.max_rounds,
        }
        return self.graph.invoke(initial_state)

    def mcp_prepare(self, state: MCPSubgraphState) -> MCPSubgraphState:
        tool_map, discovery_errors = self.mcp_gateway.discover_tools()

        tool_catalog: dict[str, list[dict[str, Any]]] = {}
        tool_schemas: dict[str, dict[str, dict[str, Any]]] = {}
        for server, tools in tool_map.items():
            summaries: list[dict[str, Any]] = []
            schemas: dict[str, dict[str, Any]] = {}
            for tool in tools:
                schema = tool.inputSchema if isinstance(tool.inputSchema, dict) else {}
                schemas[tool.name] = schema
                summaries.append(self._summarize_tool(tool_name=tool.name, description=tool.description, schema=schema))
            tool_catalog[server] = summaries
            tool_schemas[server] = schemas

        return {
            "errors": [*state.get("errors", []), *discovery_errors],
            "raw_signals": state.get("raw_signals", []),
            "mcp_round": 0,
            "mcp_max_rounds": int(state.get("mcp_max_rounds", self.max_rounds)),
            "transient_grace_used": False,
            "mcp_raw_plan_calls": [],
            "mcp_filtered_plan_calls": [],
            "mcp_admissible_calls_count": 0,
            "mcp_filtered_out_calls": [],
            "mcp_ban_reasons": [],
            "mcp_active_constraints_summary": [],
            "mcp_round_rows": [],
            "mcp_round_successes": [],
            "mcp_round_failures": [],
            "mcp_round_signal_docs": [],
            "mcp_failure_classes": {
                "deterministic_failures": [],
                "transient_failures": [],
                "deterministic_failures_exist": False,
                "transient_failures_exist": False,
            },
            "mcp_rules": [],
            "mcp_new_rules_added": False,
            "mcp_new_unique_signal_count": 0,
            "mcp_signal_hash_set": [],
            "mcp_raw_plan_hash_history": [],
            "mcp_no_progress_streak": 0,
            "mcp_should_continue": True,
            "mcp_termination_reason": "",
            "mcp_tool_catalog": tool_catalog,
            "mcp_tool_schemas": tool_schemas,
            "mcp_feedback_history": [],
            "mcp_plan_round_stats": [],
            "mcp_unique_progress_history": [],
        }

    def mcp_plan(self, state: MCPSubgraphState) -> MCPSubgraphState:
        round_no = int(state.get("mcp_round", 0)) + 1
        symbols = state.get("symbols", [])
        feedback = self._build_recent_feedback_for_prompt(state.get("mcp_feedback_history", []))
        user_prompt = self._build_planner_prompt(
            query=state.get("query", ""),
            symbols=symbols,
            tool_catalog=state.get("mcp_tool_catalog", {}),
            max_calls=PLANNER_MAX_CALLS,
            feedback=feedback,
            round_no=round_no,
            max_rounds=int(state.get("mcp_max_rounds", self.max_rounds)),
        )

        planner_text = self.llm_client.generate(
            system_prompt=self._planner_system_prompt(),
            user_prompt=user_prompt,
            metadata={
                "component": "mcp_subgraph_planner",
                "round": round_no,
            },
        )
        errors = list(state.get("errors", []))
        try:
            payload = self._parse_planner_payload(planner_text)
            raw_calls = self._normalize_raw_plan_calls(payload)
        except ValueError as exc:
            errors.append(f"MCP 规划输出无效: {exc}")
            raw_calls = []

        raw_hash = self.build_raw_plan_hash(raw_calls)
        return {
            "mcp_round": round_no,
            "mcp_raw_plan_calls": raw_calls,
            "mcp_raw_plan_hash_history": [*state.get("mcp_raw_plan_hash_history", []), raw_hash],
            "errors": errors,
        }

    def mcp_apply_rules(self, state: MCPSubgraphState) -> MCPSubgraphState:
        result = self.mcp_apply_rules_pure(
            raw_plan_calls=state.get("mcp_raw_plan_calls", []),
            rules=state.get("mcp_rules", []),
            tool_schemas=state.get("mcp_tool_schemas", {}),
        )
        round_no = int(state.get("mcp_round", 0))
        feedback_entry = {
            "round": round_no,
            "filtered_out_calls": result["filtered_out_calls"][:PLANNER_MAX_FEEDBACK_ITEMS],
            "ban_reasons": result["ban_reasons"][:PLANNER_MAX_FEEDBACK_ITEMS],
            "active_constraints_summary": result["active_constraints_summary"][:PLANNER_MAX_FEEDBACK_ITEMS],
        }
        feedback_history = [*state.get("mcp_feedback_history", []), feedback_entry][-PLANNER_MAX_FEEDBACK_ROUNDS:]
        raw_hash_history = state.get("mcp_raw_plan_hash_history", [])
        current_hash = raw_hash_history[-1] if raw_hash_history else ""
        plan_round_stats = [
            *state.get("mcp_plan_round_stats", []),
            {
                "round": round_no,
                "raw_plan_hash": current_hash,
                "filtered_plan_count": len(result["filtered_plan_calls"]),
            },
        ]
        return {
            "mcp_filtered_plan_calls": result["filtered_plan_calls"],
            "mcp_admissible_calls_count": len(result["filtered_plan_calls"]),
            "mcp_filtered_out_calls": result["filtered_out_calls"],
            "mcp_ban_reasons": result["ban_reasons"],
            "mcp_active_constraints_summary": result["active_constraints_summary"],
            "mcp_feedback_history": feedback_history,
            "mcp_plan_round_stats": plan_round_stats,
        }

    def mcp_tool_call(self, state: MCPSubgraphState) -> MCPSubgraphState:
        calls = state.get("mcp_filtered_plan_calls", [])
        if not calls:
            return {
                "mcp_round_rows": [],
                "mcp_round_successes": [],
                "mcp_round_failures": [],
                "mcp_round_signal_docs": [],
            }

        with log_context(component="mcp.subgraph.tool_call", round=state.get("mcp_round", 0)):
            result = self.mcp_gateway.execute_calls(
                task_id=state.get("task_id", ""),
                symbols=state.get("symbols", []),
                calls=calls,
            )
        raw_signals = [
            item.model_dump(mode="json")
            for item in self.mcp_gateway.normalize_rows_to_signals(task_id=state.get("task_id", ""), rows=result.rows)
        ]
        return {
            "mcp_round_rows": result.rows,
            "mcp_round_successes": result.successes,
            "mcp_round_failures": result.failures,
            "mcp_round_signal_docs": raw_signals,
            "errors": [*state.get("errors", []), *result.errors],
        }

    def mcp_collect_round(self, state: MCPSubgraphState) -> MCPSubgraphState:
        existing_signals = list(state.get("raw_signals", []))
        round_signals = state.get("mcp_round_signal_docs", [])
        signal_hash_set = set(str(item) for item in state.get("mcp_signal_hash_set", []))

        new_unique_count = 0
        for signal in round_signals:
            signal_hash = self.build_signal_hash(signal)
            if signal_hash in signal_hash_set:
                continue
            signal_hash_set.add(signal_hash)
            existing_signals.append(signal)
            new_unique_count += 1

        previous_streak = int(state.get("mcp_no_progress_streak", 0))
        no_progress_streak = previous_streak + 1 if new_unique_count == 0 else 0
        return {
            "raw_signals": existing_signals,
            "mcp_new_unique_signal_count": new_unique_count,
            "mcp_signal_hash_set": sorted(signal_hash_set),
            "mcp_no_progress_streak": no_progress_streak,
            "mcp_unique_progress_history": [*state.get("mcp_unique_progress_history", []), new_unique_count],
        }

    def mcp_classify_failures(self, state: MCPSubgraphState) -> MCPSubgraphState:
        failures = state.get("mcp_round_failures", [])
        deterministic = [item for item in failures if bool(item.get("deterministic"))]
        transient = [item for item in failures if not bool(item.get("deterministic"))]
        return {
            "mcp_failure_classes": {
                "deterministic_failures": deterministic,
                "transient_failures": transient,
                "deterministic_failures_exist": bool(deterministic),
                "transient_failures_exist": bool(transient),
            }
        }

    def mcp_reflect_rules(self, state: MCPSubgraphState) -> MCPSubgraphState:
        existing_rules = list(state.get("mcp_rules", []))
        deterministic_failures = state.get("mcp_failure_classes", {}).get("deterministic_failures", [])
        round_successes = state.get("mcp_round_successes", [])
        new_rules: list[dict[str, Any]] = []

        existing_signature_bans = {
            str(item.get("call_signature", ""))
            for item in existing_rules
            if str(item.get("type", "")) == "call_signature_ban"
        }
        for failure in deterministic_failures:
            signature = str(failure.get("call_signature", "")).strip()
            if not signature or signature in existing_signature_bans:
                continue
            existing_signature_bans.add(signature)
            new_rules.append(
                {
                    "type": "call_signature_ban",
                    "server": str(failure.get("server", "")),
                    "tool_name": str(failure.get("tool_name", "")),
                    "call_signature": signature,
                    "reason": "deterministic_failure",
                }
            )

        fail_count_by_tool: dict[tuple[str, str], int] = {}
        for failure in deterministic_failures:
            key = (str(failure.get("server", "")), str(failure.get("tool_name", "")))
            fail_count_by_tool[key] = fail_count_by_tool.get(key, 0) + 1
        success_tools = {(str(item.get("server", "")), str(item.get("tool_name", ""))) for item in round_successes}
        existing_tool_bans = {
            (str(item.get("server", "")), str(item.get("tool_name", "")))
            for item in existing_rules
            if str(item.get("type", "")) == "tool_ban"
        }
        for key, count in fail_count_by_tool.items():
            if count < 2:
                continue
            if key in success_tools or key in existing_tool_bans:
                continue
            existing_tool_bans.add(key)
            new_rules.append(
                {
                    "type": "tool_ban",
                    "server": key[0],
                    "tool_name": key[1],
                    "reason": "repeated_deterministic_failures",
                }
            )

        return {
            "mcp_rules": [*existing_rules, *new_rules],
            "mcp_new_rules_added": bool(new_rules),
        }

    def mcp_should_continue(self, state: MCPSubgraphState) -> MCPSubgraphState:
        max_rounds = int(state.get("mcp_max_rounds", self.max_rounds))
        round_no = int(state.get("mcp_round", 0))
        admissible_count = int(state.get("mcp_admissible_calls_count", 0))
        round_success_count = len(state.get("mcp_round_successes", []))
        failure_classes = state.get("mcp_failure_classes", {})
        deterministic_failures_exist = bool(failure_classes.get("deterministic_failures_exist"))
        transient_failures_exist = bool(failure_classes.get("transient_failures_exist"))
        no_new_rules_added = not bool(state.get("mcp_new_rules_added", False))
        new_unique_signal_count = int(state.get("mcp_new_unique_signal_count", 0))
        no_progress_streak = int(state.get("mcp_no_progress_streak", 0))
        transient_grace_used = bool(state.get("transient_grace_used", False))

        plan_round_stats = state.get("mcp_plan_round_stats", [])
        repeated_blind_plan = self._is_repeated_blind_plan(plan_round_stats)
        plan_oscillation = self._is_plan_oscillation(state.get("mcp_raw_plan_hash_history", []), new_unique_signal_count)

        if round_no >= max_rounds:
            return {"mcp_should_continue": False, "mcp_termination_reason": "max_rounds_reached"}

        if admissible_count == 0:
            reason = "repeated_blind_plan" if repeated_blind_plan else "no_admissible_calls"
            return {"mcp_should_continue": False, "mcp_termination_reason": reason}

        if round_success_count == 0:
            if transient_failures_exist and not transient_grace_used:
                return {
                    "mcp_should_continue": True,
                    "mcp_termination_reason": "",
                    "transient_grace_used": True,
                }
            return {"mcp_should_continue": False, "mcp_termination_reason": "no_success_calls"}

        if deterministic_failures_exist and no_new_rules_added:
            return {"mcp_should_continue": False, "mcp_termination_reason": "deterministic_no_progress"}

        if new_unique_signal_count == 0:
            if plan_oscillation:
                return {"mcp_should_continue": False, "mcp_termination_reason": "plan_oscillation"}
            if no_progress_streak >= 2:
                return {"mcp_should_continue": False, "mcp_termination_reason": "no_progress_streak"}
            return {"mcp_should_continue": False, "mcp_termination_reason": "no_new_unique_signals"}

        if repeated_blind_plan:
            return {"mcp_should_continue": False, "mcp_termination_reason": "repeated_blind_plan"}

        if plan_oscillation:
            return {"mcp_should_continue": False, "mcp_termination_reason": "plan_oscillation"}

        return {"mcp_should_continue": True, "mcp_termination_reason": ""}

    def mcp_finalize(self, state: MCPSubgraphState) -> MCPSubgraphState:
        return {
            "raw_signals": state.get("raw_signals", []),
            "errors": state.get("errors", []),
            "mcp_should_continue": False,
            "mcp_round": state.get("mcp_round", 0),
            "mcp_max_rounds": state.get("mcp_max_rounds", self.max_rounds),
            "transient_grace_used": state.get("transient_grace_used", False),
            "mcp_raw_plan_calls": state.get("mcp_raw_plan_calls", []),
            "mcp_filtered_plan_calls": state.get("mcp_filtered_plan_calls", []),
            "mcp_admissible_calls_count": state.get("mcp_admissible_calls_count", 0),
            "mcp_filtered_out_calls": state.get("mcp_filtered_out_calls", []),
            "mcp_ban_reasons": state.get("mcp_ban_reasons", []),
            "mcp_active_constraints_summary": state.get("mcp_active_constraints_summary", []),
            "mcp_round_rows": state.get("mcp_round_rows", []),
            "mcp_round_successes": state.get("mcp_round_successes", []),
            "mcp_round_failures": state.get("mcp_round_failures", []),
            "mcp_failure_classes": state.get("mcp_failure_classes", {}),
            "mcp_rules": state.get("mcp_rules", []),
            "mcp_new_rules_added": state.get("mcp_new_rules_added", False),
            "mcp_new_unique_signal_count": state.get("mcp_new_unique_signal_count", 0),
            "mcp_signal_hash_set": state.get("mcp_signal_hash_set", []),
            "mcp_raw_plan_hash_history": state.get("mcp_raw_plan_hash_history", []),
            "mcp_no_progress_streak": state.get("mcp_no_progress_streak", 0),
            "mcp_termination_reason": state.get("mcp_termination_reason", ""),
        }

    @staticmethod
    def mcp_apply_rules_pure(
        *,
        raw_plan_calls: list[dict[str, Any]],
        rules: list[dict[str, Any]],
        tool_schemas: dict[str, dict[str, dict[str, Any]]],
    ) -> dict[str, Any]:
        """纯函数：根据规则对 raw plan 做过滤/修正。"""

        tool_ban_rules = [item for item in rules if str(item.get("type", "")) == "tool_ban"]
        call_signature_ban_rules = [item for item in rules if str(item.get("type", "")) == "call_signature_ban"]
        field_patch_rules = [item for item in rules if str(item.get("type", "")) == "field_patch"]
        call_signature_bans = {str(item.get("call_signature", "")) for item in call_signature_ban_rules}

        filtered_plan_calls: list[dict[str, Any]] = []
        filtered_out_calls: list[dict[str, Any]] = []
        ban_reason_counter: dict[str, int] = {}
        schema_default_injections = 0
        technical_default_injections = 0

        for call in raw_plan_calls:
            normalized = MCPSignalSubgraphRunner._normalize_single_raw_call(call)
            server = normalized["server"]
            tool_name = normalized["tool_name"]
            arguments = dict(normalized["arguments"])

            schema = tool_schemas.get(server, {}).get(tool_name)
            if schema is None:
                filtered_out_calls.append(
                    {
                        **normalized,
                        "reason": UNKNOWN_TOOL_REASON,
                    }
                )
                ban_reason_counter[UNKNOWN_TOOL_REASON] = ban_reason_counter.get(UNKNOWN_TOOL_REASON, 0) + 1
                continue

            if MCPSignalSubgraphRunner._is_tool_banned(tool_ban_rules=tool_ban_rules, server=server, tool_name=tool_name):
                reason = "tool_ban"
                filtered_out_calls.append({**normalized, "reason": reason})
                ban_reason_counter[reason] = ban_reason_counter.get(reason, 0) + 1
                continue

            patched_arguments = MCPSignalSubgraphRunner._apply_field_patches(
                arguments=arguments,
                server=server,
                tool_name=tool_name,
                field_patch_rules=field_patch_rules,
            )
            schema_defaults, technical_defaults = MCPSignalSubgraphRunner._inject_defaults(
                arguments=patched_arguments,
                schema=schema,
            )
            schema_default_injections += schema_defaults
            technical_default_injections += technical_defaults

            missing_required = MCPSignalSubgraphRunner._find_missing_required(arguments=patched_arguments, schema=schema)
            if missing_required:
                filtered_out_calls.append(
                    {
                        **normalized,
                        "arguments": patched_arguments,
                        "reason": MISSING_REQUIRED_REASON,
                        "missing_required": missing_required,
                    }
                )
                ban_reason_counter[MISSING_REQUIRED_REASON] = ban_reason_counter.get(MISSING_REQUIRED_REASON, 0) + 1
                continue

            call_signature = MCPSignalSubgraphRunner.build_call_signature(
                server=server,
                tool_name=tool_name,
                arguments=patched_arguments,
            )
            if call_signature in call_signature_bans:
                reason = "call_signature_ban"
                filtered_out_calls.append(
                    {
                        **normalized,
                        "arguments": patched_arguments,
                        "call_signature": call_signature,
                        "reason": reason,
                    }
                )
                ban_reason_counter[reason] = ban_reason_counter.get(reason, 0) + 1
                continue

            filtered_plan_calls.append(
                {
                    **normalized,
                    "arguments": patched_arguments,
                    "call_signature": call_signature,
                }
            )

        active_constraints_summary = [
            f"tool_ban_rules={len(tool_ban_rules)}",
            f"call_signature_ban_rules={len(call_signature_ban_rules)}",
            f"field_patch_rules={len(field_patch_rules)}",
            f"default_injection(schema={schema_default_injections},technical={technical_default_injections})",
        ]
        ban_reasons = [f"{reason}:{count}" for reason, count in sorted(ban_reason_counter.items())]
        return {
            "filtered_plan_calls": filtered_plan_calls,
            "filtered_out_calls": filtered_out_calls,
            "ban_reasons": ban_reasons,
            "active_constraints_summary": active_constraints_summary,
        }

    @staticmethod
    def _normalize_single_raw_call(call: dict[str, Any]) -> dict[str, Any]:
        arguments = call.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}
        return {
            "server": str(call.get("server", "")).strip(),
            "tool_name": str(call.get("tool_name", "")).strip(),
            "arguments": arguments,
            "reason": str(call.get("reason", "")).strip(),
        }

    @staticmethod
    def _is_tool_banned(*, tool_ban_rules: list[dict[str, Any]], server: str, tool_name: str) -> bool:
        for rule in tool_ban_rules:
            if str(rule.get("server", "")) != server:
                continue
            if str(rule.get("tool_name", "")) != tool_name:
                continue
            return True
        return False

    @staticmethod
    def _apply_field_patches(
        *,
        arguments: dict[str, Any],
        server: str,
        tool_name: str,
        field_patch_rules: list[dict[str, Any]],
    ) -> dict[str, Any]:
        patched = dict(arguments)
        for rule in field_patch_rules:
            if str(rule.get("server", "")) != server:
                continue
            if str(rule.get("tool_name", "")) != tool_name:
                continue
            field_name = str(rule.get("field", "")).strip()
            if not field_name:
                continue
            operation = str(rule.get("operation", "set")).strip().lower()
            if operation == "remove":
                patched.pop(field_name, None)
                continue
            patched[field_name] = rule.get("value")
        return patched

    @staticmethod
    def _inject_defaults(*, arguments: dict[str, Any], schema: dict[str, Any]) -> tuple[int, int]:
        properties = schema.get("properties", {}) if isinstance(schema.get("properties"), dict) else {}
        required = schema.get("required", []) if isinstance(schema.get("required"), list) else []
        schema_injected = 0
        technical_injected = 0

        # schema default 允许注入（required/optional 都可）。
        for field_name, field_schema in properties.items():
            key = str(field_name)
            if key in arguments:
                continue
            if not isinstance(field_schema, dict):
                continue
            if "default" not in field_schema:
                continue
            arguments[key] = field_schema.get("default")
            schema_injected += 1

        # technical default 仅在 required 缺失时注入，避免业务语义猜值。
        for field in required:
            field_name = str(field)
            if field_name in arguments and not MCPSignalSubgraphRunner._is_blank(arguments[field_name]):
                continue
            if field_name not in TECHNICAL_DEFAULTS:
                continue
            field_schema = properties.get(field_name)
            if not isinstance(field_schema, dict):
                continue
            injected_value = TECHNICAL_DEFAULTS[field_name]
            coerced, ok = MCPSignalSubgraphRunner._coerce_for_schema(value=injected_value, schema=field_schema)
            if not ok:
                continue
            arguments[field_name] = coerced
            technical_injected += 1
        return schema_injected, technical_injected

    @staticmethod
    def _coerce_for_schema(*, value: Any, schema: dict[str, Any]) -> tuple[Any, bool]:
        expected = str(schema.get("type", "")).strip().lower()
        if expected == "integer":
            if isinstance(value, bool):
                return value, False
            if isinstance(value, int):
                return value, True
            if isinstance(value, float):
                return int(value), True
            return value, False
        if expected == "number":
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value, True
            return value, False
        if expected == "string":
            return str(value), True
        if expected == "boolean":
            if isinstance(value, bool):
                return value, True
            return value, False
        return value, True

    @staticmethod
    def _find_missing_required(*, arguments: dict[str, Any], schema: dict[str, Any]) -> list[str]:
        required = schema.get("required", []) if isinstance(schema.get("required"), list) else []
        missing: list[str] = []
        for field in required:
            key = str(field)
            if key not in arguments or MCPSignalSubgraphRunner._is_blank(arguments.get(key)):
                missing.append(key)
        return missing

    @staticmethod
    def _is_blank(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        return False

    @staticmethod
    def build_call_signature(*, server: str, tool_name: str, arguments: dict[str, Any]) -> str:
        return f"{server}|{tool_name}|{MCPSignalSubgraphRunner._canonical_json(arguments)}"

    @staticmethod
    def build_raw_plan_hash(raw_plan_calls: list[dict[str, Any]]) -> str:
        canonical_calls = [
            {
                "server": str(item.get("server", "")).strip(),
                "tool_name": str(item.get("tool_name", "")).strip(),
                "arguments": item.get("arguments", {}) if isinstance(item.get("arguments"), dict) else {},
            }
            for item in raw_plan_calls
        ]
        payload = MCPSignalSubgraphRunner._canonical_json(canonical_calls)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def build_signal_hash(signal: dict[str, Any]) -> str:
        core = {
            "symbol": signal.get("symbol", ""),
            "source": signal.get("source", ""),
            "signal_type": signal.get("signal_type", ""),
            "value": signal.get("value"),
        }
        payload = MCPSignalSubgraphRunner._canonical_json(core)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _canonical_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)

    @staticmethod
    def _is_repeated_blind_plan(plan_round_stats: list[dict[str, Any]]) -> bool:
        if len(plan_round_stats) < 2:
            return False
        previous = plan_round_stats[-2]
        current = plan_round_stats[-1]
        return (
            str(previous.get("raw_plan_hash", "")) == str(current.get("raw_plan_hash", ""))
            and int(previous.get("filtered_plan_count", 0)) == 0
            and int(current.get("filtered_plan_count", 0)) == 0
        )

    @staticmethod
    def _is_plan_oscillation(raw_plan_hash_history: list[str], new_unique_signal_count: int) -> bool:
        if new_unique_signal_count != 0:
            return False
        if len(raw_plan_hash_history) < 4:
            return False
        a = raw_plan_hash_history[-4]
        b = raw_plan_hash_history[-3]
        c = raw_plan_hash_history[-2]
        d = raw_plan_hash_history[-1]
        return a == c and b == d and a != b

    def _build_recent_feedback_for_prompt(self, feedback_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not feedback_history:
            return []
        trimmed: list[dict[str, Any]] = []
        for item in feedback_history[-PLANNER_MAX_FEEDBACK_ROUNDS:]:
            if not isinstance(item, dict):
                continue
            trimmed.append(
                {
                    "round": int(item.get("round", 0)),
                    "filtered_out_calls": list(item.get("filtered_out_calls", []))[:PLANNER_MAX_FEEDBACK_ITEMS],
                    "ban_reasons": list(item.get("ban_reasons", []))[:PLANNER_MAX_FEEDBACK_ITEMS],
                    "active_constraints_summary": list(item.get("active_constraints_summary", []))[:PLANNER_MAX_FEEDBACK_ITEMS],
                }
            )
        return trimmed

    def _build_planner_prompt(
        self,
        *,
        query: str,
        symbols: list[str],
        tool_catalog: dict[str, list[dict[str, Any]]],
        max_calls: int,
        feedback: list[dict[str, Any]],
        round_no: int,
        max_rounds: int,
    ) -> str:
        symbols_text = ",".join(symbols) if symbols else "BTC,ETH"
        return (
            f"round={round_no}/{max_rounds}\n"
            f"query={query}\n"
            f"target_symbols={symbols_text}\n"
            f"max_calls={max_calls}\n\n"
            "输出 JSON 对象：\n"
            "{\n"
            '  "calls":[\n'
            "    {\n"
            '      "server":"服务名",\n'
            '      "tool_name":"工具名",\n'
            '      "arguments":{"k":"v"},\n'
            '      "reason":"不超过30字"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "约束：\n"
            "1) 仅可选择候选工具；\n"
            "2) arguments 必须是 JSON object；\n"
            "3) 禁止输出 markdown 代码块。\n\n"
            f"最近过滤反馈（只做参考，规则层仍会强约束）：\n{json.dumps(feedback, ensure_ascii=False)}\n\n"
            f"候选工具目录：\n{json.dumps(tool_catalog, ensure_ascii=False)}"
        )

    def _planner_system_prompt(self) -> str:
        return (
            "你是 MCP 调用规划器。"
            "你只负责给出候选调用，不负责规则判断。"
            "你必须只输出可被 json.loads 直接解析的 JSON 对象。"
        )

    def _parse_planner_payload(self, planner_text: str) -> dict[str, Any]:
        cleaned = planner_text.strip()
        if not cleaned:
            raise ValueError("planner output is empty")

        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()
        decoder = json.JSONDecoder()
        for index, char in enumerate(stripped):
            if char != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(stripped[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
        raise ValueError("planner output is not valid JSON object")

    def _normalize_raw_plan_calls(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        calls = payload.get("calls")
        if not isinstance(calls, list):
            raise ValueError("missing calls list")

        normalized: list[dict[str, Any]] = []
        for item in calls:
            if not isinstance(item, dict):
                continue
            server = str(item.get("server", "")).strip()
            tool_name = str(item.get("tool_name", "")).strip()
            arguments = item.get("arguments", {})
            if not isinstance(arguments, dict):
                continue
            reason = str(item.get("reason", "")).strip()
            if not server or not tool_name:
                continue
            normalized.append(
                {
                    "server": server,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "reason": reason,
                }
            )
            if len(normalized) >= PLANNER_MAX_CALLS:
                break
        return normalized

    def _summarize_tool(self, *, tool_name: str, description: str | None, schema: dict[str, Any]) -> dict[str, Any]:
        properties = schema.get("properties", {}) if isinstance(schema.get("properties"), dict) else {}
        required = schema.get("required", []) if isinstance(schema.get("required"), list) else []
        concise_properties: dict[str, Any] = {}
        for key, value in list(properties.items())[:10]:
            key_text = str(key)
            if isinstance(value, dict):
                concise_item: dict[str, Any] = {}
                if "type" in value:
                    concise_item["type"] = value["type"]
                if "default" in value:
                    concise_item["default"] = value["default"]
                if "enum" in value and isinstance(value["enum"], list):
                    concise_item["enum"] = value["enum"][:8]
                concise_properties[key_text] = concise_item
            else:
                concise_properties[key_text] = {}

        return {
            "name": tool_name,
            "description": (description or "")[:220],
            "required": [str(item) for item in required],
            "properties": concise_properties,
        }


__all__ = ["MCPSubgraphState", "MCPSignalSubgraphRunner"]
