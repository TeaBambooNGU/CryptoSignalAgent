"""LangGraph 主流程编排。

实现设计文档中的 9 个节点，并在关键节点加入重试/降级策略。
"""

from __future__ import annotations

import re
from collections import Counter
from time import perf_counter
from typing import Any
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.agents.report_agent import ReportAgent
from app.config.logging import get_logger, log_context
from app.memory.mem0_service import MemoryService
from app.models.schemas import QueryResponse, ReportGenerationInput
from app.models.state import ResearchState
from app.retrieval.research_service import ResearchService
from app.tools.mcp_client import MCPClient

logger = get_logger(__name__)

SYMBOL_PATTERN = re.compile(r"\b[A-Z]{2,10}\b")
MCP_FEEDBACK_MAX_ROUNDS = 2


class ResearchGraphRunner:
    """研报工作流执行器。"""

    def __init__(
        self,
        memory_service: MemoryService,
        mcp_client: MCPClient,
        research_service: ResearchService,
        report_agent: ReportAgent,
    ) -> None:
        self.memory_service = memory_service
        self.mcp_client = mcp_client
        self.research_service = research_service
        self.report_agent = report_agent
        self.graph = self._build_graph()

    def _build_graph(self):
        """构建 LangGraph 节点与边。"""

        builder = StateGraph(ResearchState)

        builder.add_node("load_user_memory", self.load_user_memory)
        builder.add_node("parse_intent_scope", self.parse_intent_scope)
        builder.add_node("collect_signals_via_mcp", self.collect_signals_via_mcp)
        builder.add_node("normalize_and_index", self.normalize_and_index)
        builder.add_node("retrieve_context", self.retrieve_context)
        builder.add_node("analyze_signals", self.analyze_signals)
        builder.add_node("generate_report", self.generate_report)
        builder.add_node("persist_memory", self.persist_memory)
        builder.add_node("finalize_response", self.finalize_response)

        builder.add_edge(START, "load_user_memory")
        builder.add_edge("load_user_memory", "parse_intent_scope")
        builder.add_edge("parse_intent_scope", "collect_signals_via_mcp")
        builder.add_edge("collect_signals_via_mcp", "normalize_and_index")
        builder.add_edge("normalize_and_index", "retrieve_context")
        builder.add_edge("retrieve_context", "analyze_signals")
        builder.add_edge("analyze_signals", "generate_report")
        builder.add_edge("generate_report", "persist_memory")
        builder.add_edge("persist_memory", "finalize_response")
        builder.add_edge("finalize_response", END)

        return builder.compile()

    def run(
        self,
        user_id: str,
        query: str,
        task_context: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> QueryResponse:
        """执行研报流程并返回 API 响应。"""

        task_id = str(uuid4())
        final_trace_id = trace_id or str(uuid4())
        initial_state: ResearchState = {
            "user_id": user_id,
            "query": query,
            "task_context": task_context,
            "task_id": task_id,
            "trace_id": final_trace_id,
            "errors": [],
            "retry_count": 0,
            "workflow_steps": [],
        }

        with log_context(
            trace_id=final_trace_id,
            task_id=task_id,
            user_id=user_id,
            component="graph.runner",
        ):
            logger.info("工作流开始 query_len=%s", len(query))
            output = self.graph.invoke(initial_state)
            logger.info(
                "工作流结束 citations=%s errors=%s",
                len(output.get("citations", [])),
                len(output.get("errors", [])),
            )

        return QueryResponse(
            report=output.get("final_report", "未能生成报告，请稍后重试。"),
            citations=output.get("citations", []),
            trace_id=final_trace_id,
            errors=output.get("errors", []),
            workflow_steps=output.get("workflow_steps", []),
        )

    # ===== 节点定义 =====
    def load_user_memory(self, state: ResearchState) -> ResearchState:
        """节点目标：加载长期偏好与会话上下文。

        前置条件：state 已包含 `user_id`。
        状态产出：`memory_profile`。
        """

        def _node_logic() -> ResearchState:
            user_id = state["user_id"]
            with log_context(component="graph.load_user_memory"):
                logger.info("节点开始")
                self.memory_service.save_task_context(user_id=user_id, task_context=state.get("task_context"))
                profile = self.memory_service.load_memory_profile(user_id=user_id)
                logger.info(
                    "节点完成 long_term=%s session=%s",
                    len(profile.get("long_term_memory", [])),
                    len(profile.get("session_memory", [])),
                )
            return {"memory_profile": profile}

        return self._run_tracked_node(state, node_id="load_user_memory", handler=_node_logic)

    def parse_intent_scope(self, state: ResearchState) -> ResearchState:
        """节点目标：解析查询范围与关注标的。

        前置条件：state 已包含 `query` 与 `memory_profile`。
        状态产出：`symbols`。
        """

        def _node_logic() -> ResearchState:
            query = state["query"]
            with log_context(component="graph.parse_intent_scope"):
                symbols = {token.upper() for token in SYMBOL_PATTERN.findall(query)}

                profile_watchlist = state.get("memory_profile", {}).get("watchlist", [])
                if isinstance(profile_watchlist, list):
                    symbols.update(str(item).upper() for item in profile_watchlist)

                task_context = state.get("task_context") or {}
                ctx_symbols = task_context.get("symbols") if isinstance(task_context, dict) else None
                if isinstance(ctx_symbols, list):
                    symbols.update(str(item).upper() for item in ctx_symbols)

                if not symbols:
                    symbols = {"BTC", "ETH"}

                sorted_symbols = sorted(symbols)
                logger.info("节点完成 symbols=%s", sorted_symbols)
                return {"symbols": sorted_symbols}

        return self._run_tracked_node(state, node_id="parse_intent_scope", handler=_node_logic)

    def collect_signals_via_mcp(self, state: ResearchState) -> ResearchState:
        """节点目标：通过 MCP 拉取信号。

        前置条件：state 已包含 `task_id`、`query`、`symbols`。
        状态产出：`raw_signals`、`errors`。
        """

        def _node_logic() -> ResearchState:
            with log_context(component="graph.collect_signals"):
                historical_corrections = self._load_tool_corrections_for_planner(state=state)
                combined_signals = []
                combined_errors: list[str] = []
                combined_failures: list[dict[str, Any]] = []
                correction_events: list[dict[str, Any]] = []
                feedback_for_retry: dict[str, list[dict[str, Any]]] = {}
                previous_feedback: dict[str, list[dict[str, Any]]] = {}

                for round_index in range(MCP_FEEDBACK_MAX_ROUNDS):
                    result = self._collect_with_retry(
                        task_id=state["task_id"],
                        query=state["query"],
                        symbols=state.get("symbols", []),
                        planning_context={
                            "historical_corrections": historical_corrections,
                            "server_failures": feedback_for_retry,
                        },
                    )
                    combined_signals = self._merge_raw_signals(combined_signals, result.signals)
                    combined_errors.extend(result.errors)
                    combined_failures.extend(result.failures)

                    deterministic_feedback = self._group_deterministic_failures(result.failures)
                    if deterministic_feedback and round_index + 1 < MCP_FEEDBACK_MAX_ROUNDS:
                        previous_feedback = deterministic_feedback
                        feedback_for_retry = deterministic_feedback
                        logger.info(
                            "触发 MCP 纠错重试 round=%s deterministic_failures=%s",
                            round_index + 2,
                            sum(len(items) for items in deterministic_feedback.values()),
                        )
                        continue

                    if previous_feedback:
                        correction_events = self._build_correction_events(
                            previous_feedback=previous_feedback,
                            successes=result.successes,
                        )
                        if correction_events:
                            self._persist_tool_corrections(user_id=state["user_id"], corrections=correction_events)
                            logger.info("MCP 纠错经验写回长期记忆 count=%s", len(correction_events))
                    break

                logger.info(
                    "节点完成 raw_signals=%s errors=%s failures=%s corrections=%s",
                    len(combined_signals),
                    len(combined_errors),
                    len(combined_failures),
                    len(correction_events),
                )

            merged_errors = [*state.get("errors", []), *combined_errors]
            return {
                "raw_signals": [item.model_dump() for item in combined_signals],
                "errors": merged_errors,
                "mcp_failures": combined_failures,
                "mcp_corrections": correction_events,
            }

        return self._run_tracked_node(state, node_id="collect_signals_via_mcp", handler=_node_logic)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _collect_with_retry(
        self,
        task_id: str,
        query: str,
        symbols: list[str],
        planning_context: dict[str, Any] | None,
    ):
        """MCP 采集重试包装。

        触发条件：网络异常、超时、上游瞬时错误。
        停止条件：最多 3 次。
        """

        return self.mcp_client.collect_signals_detailed(
            task_id=task_id,
            query=query,
            symbols=symbols,
            planning_context=planning_context,
        )

    def _load_tool_corrections_for_planner(self, state: ResearchState) -> list[dict[str, Any]]:
        """从状态画像读取 MCP 纠错记忆。"""

        profile = state.get("memory_profile", {})
        corrections = profile.get("tool_corrections", []) if isinstance(profile, dict) else []
        if isinstance(corrections, list):
            return [item for item in corrections if isinstance(item, dict)]
        return []

    def _merge_raw_signals(self, existing: list[Any], incoming: list[Any]) -> list[Any]:
        """跨轮次去重合并 RawSignal，避免重复写入。"""

        merged: list[Any] = []
        seen: set[tuple[str, str, str, str]] = set()
        for signal in [*existing, *incoming]:
            key = (
                str(getattr(signal, "source", "")),
                str(getattr(signal, "symbol", "")),
                str(getattr(signal, "raw_ref", "")),
                str(getattr(signal, "published_at", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(signal)
        return merged

    def _group_deterministic_failures(self, failures: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        """按 server 聚合可用于纠错重规划的确定性失败。"""

        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in failures:
            if not isinstance(item, dict) or not item.get("deterministic"):
                continue
            server = str(item.get("server", "")).strip()
            if not server:
                continue
            grouped.setdefault(server, []).append(item)
        return grouped

    def _build_correction_events(
        self,
        *,
        previous_feedback: dict[str, list[dict[str, Any]]],
        successes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """从“上一轮失败 + 本轮成功”提取可复用纠错样本。"""

        success_by_server: dict[str, list[dict[str, Any]]] = {}
        for success in successes:
            if not isinstance(success, dict):
                continue
            server = str(success.get("server", "")).strip()
            if not server:
                continue
            success_by_server.setdefault(server, []).append(success)

        events: list[dict[str, Any]] = []
        for server, failures in previous_feedback.items():
            server_successes = success_by_server.get(server, [])
            if not failures or not server_successes:
                continue
            representative_success = next(
                (item for item in server_successes if int(item.get("rows", 0) or 0) > 0),
                server_successes[0],
            )
            for failure in failures[:3]:
                error_detail = failure.get("error_detail", {})
                events.append(
                    {
                        "server": server,
                        "failed_tool": str(failure.get("tool_name", "")),
                        "failed_arguments": failure.get("arguments", {}),
                        "error_signature": self._build_error_signature_from_detail(
                            error_detail if isinstance(error_detail, dict) else {}
                        ),
                        "fixed_tool": str(representative_success.get("tool_name", "")),
                        "fixed_arguments": representative_success.get("arguments", {}),
                    }
                )
        return events

    def _persist_tool_corrections(self, *, user_id: str, corrections: list[dict[str, Any]]) -> None:
        """将纠错经验写入长期记忆。"""

        for item in corrections:
            self.memory_service.save_tool_correction(user_id=user_id, correction=item, confidence=0.78)

    def _build_error_signature_from_detail(self, detail: dict[str, Any]) -> str:
        """构建轻量错误签名，便于记忆检索。"""

        status = detail.get("status_code", "unknown")
        message = detail.get("error_message") or detail.get("message") or detail.get("response_body") or ""
        raw = f"status={status} message={message}"
        return raw.replace("\n", " ")[:220]

    def normalize_and_index(self, state: ResearchState) -> ResearchState:
        """节点目标：标准化信号并入库。

        前置条件：state 已包含 `raw_signals`。
        状态产出：`signals`。
        """

        def _node_logic() -> ResearchState:
            raw_signals = state.get("raw_signals", [])
            task_id = state["task_id"]

            with log_context(component="graph.normalize_and_index"):
                normalized = self.research_service.normalize_signals(
                    task_id=task_id,
                    raw_signals=[self._raw_signal_from_dict(item) for item in raw_signals],
                )
                if normalized:
                    inserted = self.research_service.ingest_signals(normalized)
                    logger.info("节点完成 normalized=%s inserted_chunks=%s", len(normalized), inserted)
                else:
                    logger.warning("节点降级 raw_signals=0")
                    errors = [*state.get("errors", []), "MCP 未返回有效信号，使用历史检索降级生成报告"]
                    return {"signals": [], "errors": errors}

            return {"signals": normalized}

        return self._run_tracked_node(state, node_id="normalize_and_index", handler=_node_logic)

    def retrieve_context(self, state: ResearchState) -> ResearchState:
        """节点目标：检索历史证据与最新上下文。

        前置条件：state 已包含 `query` 与 `symbols`。
        状态产出：`retrieved_docs`。
        """

        def _node_logic() -> ResearchState:
            query = state["query"]
            symbols = state.get("symbols", [])

            with log_context(component="graph.retrieve_context"):
                docs = self._retrieve_with_retry(query=query, symbols=symbols)
                fallback_used = False
                if not docs and symbols:
                    # 检索不足时放宽条件重试一次。
                    docs = self._retrieve_with_retry(query=query, symbols=[])
                    fallback_used = True
                    if not docs:
                        logger.warning("节点完成 docs=0 fallback=%s", fallback_used)
                        return {"retrieved_docs": [], "errors": [*state.get("errors", []), "检索命中不足"]}

                logger.info("节点完成 docs=%s fallback=%s", len(docs), fallback_used)
                return {"retrieved_docs": docs}

        return self._run_tracked_node(state, node_id="retrieve_context", handler=_node_logic)

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _retrieve_with_retry(self, query: str, symbols: list[str]):
        """检索重试包装。"""

        return self.research_service.retrieve(query=query, symbols=symbols, top_k=8)

    def analyze_signals(self, state: ResearchState) -> ResearchState:
        """节点目标：评估信号强弱与冲突。

        前置条件：state 已包含 `signals` 与 `retrieved_docs`。
        状态产出：`analysis_summary`。
        """

        def _node_logic() -> ResearchState:
            signals = state.get("signals", [])
            docs = state.get("retrieved_docs", [])

            with log_context(component="graph.analyze_signals"):
                if not signals:
                    summary = "实时信号不足，分析主要依赖历史证据与用户偏好。"
                    logger.info("节点完成 signals=0 docs=%s", len(docs))
                    return {"analysis_summary": summary}

                symbol_counter = Counter(item.symbol for item in signals)
                type_counter = Counter(item.signal_type.value for item in signals)
                avg_conf = sum(item.confidence for item in signals) / len(signals)

                summary = (
                    f"信号总量={len(signals)}；覆盖标的={dict(symbol_counter)}；"
                    f"类型分布={dict(type_counter)}；平均置信度={avg_conf:.2f}；"
                    f"检索命中={len(docs)}。"
                )
                logger.info("节点完成 signals=%s docs=%s", len(signals), len(docs))
                return {"analysis_summary": summary}

        return self._run_tracked_node(state, node_id="analyze_signals", handler=_node_logic)

    def generate_report(self, state: ResearchState) -> ResearchState:
        """节点目标：生成结构化研报。

        前置条件：state 已包含 `memory_profile`、`signals`、`retrieved_docs`。
        状态产出：`report_draft`、`final_report`、`citations`。
        """

        def _node_logic() -> ResearchState:
            payload = ReportGenerationInput(
                user_id=state["user_id"],
                query=state["query"],
                task_id=state["task_id"],
                memory_profile=state.get("memory_profile", {}),
                signals=state.get("signals", []),
                retrieved_docs=state.get("retrieved_docs", []),
            )

            with log_context(component="graph.generate_report"):
                output = self._generate_report_with_retry(payload)
                logger.info("节点完成 report_len=%s citations=%s", len(output.report), len(output.citations))
                return {
                    "report_draft": output.draft,
                    "final_report": output.report,
                    "citations": output.citations,
                }

        return self._run_tracked_node(state, node_id="generate_report", handler=_node_logic)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _generate_report_with_retry(self, payload: ReportGenerationInput):
        """报告生成重试包装。"""

        return self.report_agent.generate(payload)

    def persist_memory(self, state: ResearchState) -> ResearchState:
        """节点目标：写回长期可复用记忆。

        前置条件：state 已包含 `final_report`。
        状态产出：无新增核心字段（仅副作用写入）。

        写回规则：
        - 长期记忆：关注标的、风险偏好等可复用偏好。
        - 短期信息：任务上下文和一次性中间结论不进入长期记忆。
        """

        def _node_logic() -> ResearchState:
            with log_context(component="graph.persist_memory"):
                self.memory_service.persist_report_memory(
                    user_id=state["user_id"],
                    query=state["query"],
                    report=state.get("final_report", ""),
                )
                logger.info("节点完成")
            return {}

        return self._run_tracked_node(state, node_id="persist_memory", handler=_node_logic)

    def finalize_response(self, state: ResearchState) -> ResearchState:
        """节点目标：收敛输出结构，保证字段完整。"""

        def _node_logic() -> ResearchState:
            with log_context(component="graph.finalize_response"):
                final_report = state.get("final_report")
                if not final_report:
                    final_report = "当前数据不足，无法生成稳定结论，请稍后重试。"
                logger.info("节点完成 errors=%s", len(state.get("errors", [])))
                return {
                    "final_report": final_report,
                    "citations": state.get("citations", []),
                    "errors": state.get("errors", []),
                }

        return self._run_tracked_node(state, node_id="finalize_response", handler=_node_logic)

    def _run_tracked_node(self, state: ResearchState, node_id: str, handler) -> ResearchState:
        """执行节点并记录真实耗时与状态。"""

        start = perf_counter()
        try:
            update = handler() or {}
        except Exception:
            state["workflow_steps"] = self._append_workflow_step(state, node_id=node_id, status="error", start=start)
            raise
        return {
            **update,
            "workflow_steps": self._append_workflow_step(state, node_id=node_id, status="success", start=start),
        }

    def _append_workflow_step(
        self,
        state: ResearchState,
        node_id: str,
        status: str,
        start: float,
    ) -> list[dict[str, Any]]:
        """追加单个工作流节点执行记录。"""

        duration_ms = max(int((perf_counter() - start) * 1000), 0)
        step = {"node_id": node_id, "status": status, "duration_ms": duration_ms}
        return [*state.get("workflow_steps", []), step]

    def _raw_signal_from_dict(self, item: dict[str, Any]):
        """将字典恢复为 RawSignal。"""

        from app.models.schemas import RawSignal

        return RawSignal.model_validate(item)
