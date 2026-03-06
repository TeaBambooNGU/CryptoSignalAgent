"""LangGraph 主流程编排。

实现设计文档中的 9 个节点，并在关键节点加入重试/降级策略。
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter
from time import perf_counter
from typing import Any
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.agents.report_agent import ReportAgent
from app.config.logging import get_logger, log_context
from app.graph.mcp_subgraph import MCPSignalSubgraphRunner
from app.memory.mem0_service import MemoryService
from app.models.schemas import QueryResponse, ReportGenerationInput, WorkflowStep, ensure_citations, ensure_workflow_steps
from app.models.state import ResearchState
from app.retrieval.research_service import ResearchService

logger = get_logger(__name__)

PAIR_SYMBOL_PATTERN = re.compile(r"\b([A-Za-z]{2,10})/(?:USDT|USD|BTC|ETH|BUSD)\b")
TOKEN_SYMBOL_PATTERN = re.compile(r"(?<![A-Za-z0-9])\$?([A-Za-z]{2,10})(?![A-Za-z0-9])")
SYMBOL_STOPWORDS = {
    "ETF",
    "SEC",
    "CPI",
    "PPI",
    "FOMC",
    "FED",
    "NFP",
    "GDP",
    "PMI",
    "USD",
    "US",
    "AI",
}


class ResearchGraphRunner:
    """研报工作流执行器。"""

    def __init__(
        self,
        memory_service: MemoryService,
        mcp_subgraph: MCPSignalSubgraphRunner,
        research_service: ResearchService,
        report_agent: ReportAgent,
    ) -> None:
        self.memory_service = memory_service
        self.mcp_subgraph = mcp_subgraph
        self.research_service = research_service
        self.report_agent = report_agent
        self.graph = self._build_graph()

    def _build_graph(self):
        """构建 LangGraph 节点与边。"""

        builder = StateGraph(ResearchState)

        builder.add_node("load_user_memory", self.load_user_memory)
        builder.add_node("resolve_symbols", self.resolve_symbols)
        builder.add_node("collect_signals_via_mcp", self.collect_signals_via_mcp)
        builder.add_node("normalize_and_index", self.normalize_and_index)
        builder.add_node("retrieve_knowledge_evidence", self.retrieve_knowledge_evidence)
        builder.add_node("analyze_signals", self.analyze_signals)
        builder.add_node("generate_report", self.generate_report)
        builder.add_node("persist_memory", self.persist_memory)
        builder.add_node("finalize_response", self.finalize_response)

        builder.add_edge(START, "load_user_memory")
        builder.add_edge("load_user_memory", "resolve_symbols")
        builder.add_edge("resolve_symbols", "collect_signals_via_mcp")
        builder.add_edge("collect_signals_via_mcp", "normalize_and_index")
        builder.add_edge("normalize_and_index", "retrieve_knowledge_evidence")
        builder.add_edge("retrieve_knowledge_evidence", "analyze_signals")
        builder.add_edge("analyze_signals", "generate_report")
        builder.add_edge("generate_report", "persist_memory")
        builder.add_edge("persist_memory", "finalize_response")
        builder.add_edge("finalize_response", END)

        return builder.compile()

    async def arun(
        self,
        user_id: str,
        query: str,
        task_context: dict[str, Any] | None = None,
        trace_id: str | None = None,
        conversation_id: str | None = None,
        turn_id: str | None = None,
        request_id: str | None = None,
        conversation_version: int | None = None,
        context_anchor_turn_id: str | None = None,
    ) -> QueryResponse:
        """执行研报流程并返回 API 响应。"""

        task_id = str(uuid4())
        final_trace_id = trace_id or str(uuid4())
        final_conversation_id = conversation_id or f"default:{user_id}"
        final_turn_id = turn_id or "turn-1"
        final_request_id = request_id or str(uuid4())
        final_conversation_version = conversation_version or 1
        initial_state: ResearchState = {
            "user_id": user_id,
            "query": query,
            "task_context": task_context,
            "conversation_id": final_conversation_id,
            "turn_id": final_turn_id,
            "context_anchor_turn_id": context_anchor_turn_id,
            "request_id": final_request_id,
            "conversation_version": final_conversation_version,
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
            output = await self.graph.ainvoke(initial_state)
            logger.info(
                "工作流结束 citations=%s errors=%s",
                len(output.get("citations", [])),
                len(output.get("errors", [])),
            )

        return QueryResponse(
            report=output.get("final_report", "未能生成报告，请稍后重试。"),
            citations=ensure_citations(output.get("citations", [])),
            trace_id=final_trace_id,
            conversation_id=final_conversation_id,
            turn_id=final_turn_id,
            request_id=final_request_id,
            conversation_version=final_conversation_version,
            errors=output.get("errors", []),
            workflow_steps=ensure_workflow_steps(output.get("workflow_steps", [])),
        )

    def run(
        self,
        user_id: str,
        query: str,
        task_context: dict[str, Any] | None = None,
        trace_id: str | None = None,
        conversation_id: str | None = None,
        turn_id: str | None = None,
        request_id: str | None = None,
        conversation_version: int | None = None,
        context_anchor_turn_id: str | None = None,
    ) -> QueryResponse:
        """同步包装器（兼容旧调用方）。"""

        return asyncio.run(
            self.arun(
                user_id=user_id,
                query=query,
                task_context=task_context,
                trace_id=trace_id,
                conversation_id=conversation_id,
                turn_id=turn_id,
                request_id=request_id,
                conversation_version=conversation_version,
                context_anchor_turn_id=context_anchor_turn_id,
            )
        )

    # ===== 节点定义 =====
    def load_user_memory(self, state: ResearchState) -> ResearchState:
        """节点目标：加载长期偏好与会话上下文。

        前置条件：state 已包含 `user_id`。
        状态产出：`memory_profile`。
        """

        def _node_logic() -> ResearchState:
            user_id = state["user_id"]
            conversation_id = state.get("conversation_id") or f"default:{user_id}"
            with log_context(component="graph.load_user_memory"):
                logger.info("节点开始")
                self.memory_service.save_task_context(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    task_context=state.get("task_context"),
                )
                profile = self.memory_service.load_memory_profile(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    context_anchor_turn_id=state.get("context_anchor_turn_id"),
                )
                logger.info(
                    "节点完成 long_term=%s session=%s",
                    len(profile.get("long_term_memory", [])),
                    len(profile.get("session_memory", [])),
                )
            return {"memory_profile": profile}

        return self._run_tracked_node(state, node_id="load_user_memory", handler=_node_logic)

    def resolve_symbols(self, state: ResearchState) -> ResearchState:
        """节点目标：确定本次任务 symbols 路由参数。

        前置条件：state 已包含 `query` 与 `memory_profile`。
        状态产出：`hard_symbols`、`soft_symbols`。
        """

        def _node_logic() -> ResearchState:
            query = state["query"]
            with log_context(component="graph.resolve_symbols"):
                task_context = state.get("task_context") or {}
                context_symbols = self._extract_symbols_from_task_context(task_context)
                query_symbols = self._extract_symbols_from_query(query)
                watchlist = self._normalize_symbols(state.get("memory_profile", {}).get("watchlist", []))

                # 优先级：task_context > query；watchlist 仅用于 MCP 软提示，不扩张检索边界。
                if context_symbols:
                    hard_symbols = context_symbols
                elif query_symbols:
                    hard_symbols = query_symbols
                else:
                    hard_symbols = []

                soft_symbols = self._merge_symbols(hard_symbols, watchlist)
                logger.info("节点完成 hard_symbols=%s soft_symbols=%s", hard_symbols, soft_symbols)
                return {
                    "hard_symbols": hard_symbols,
                    "soft_symbols": soft_symbols,
                    "symbols": hard_symbols,
                }

        return self._run_tracked_node(state, node_id="resolve_symbols", handler=_node_logic)

    async def collect_signals_via_mcp(self, state: ResearchState) -> ResearchState:
        """节点目标：通过 MCP 拉取信号。

        前置条件：state 已包含 `task_id`、`query`。
        状态产出：`raw_signals`、`errors`。
        """

        async def _node_logic() -> ResearchState:
            with log_context(component="graph.collect_signals"):
                hard_symbols = state.get("hard_symbols")
                soft_symbols = state.get("soft_symbols")
                request_symbols = hard_symbols if isinstance(hard_symbols, list) else state.get("symbols", [])
                hint_symbols = soft_symbols if isinstance(soft_symbols, list) else request_symbols
                result = await self.mcp_subgraph.arun(
                    user_id=state["user_id"],
                    query=state["query"],
                    task_id=state["task_id"],
                    symbols=request_symbols,
                    hint_symbols=hint_symbols,
                    errors=state.get("errors", []),
                )
                logger.info(
                    "节点完成 raw_signals=%s errors=%s tools=%s termination=%s",
                    len(result.get("raw_signals", [])),
                    len(result.get("errors", [])),
                    int(result.get("mcp_tools_count", 0)),
                    result.get("mcp_termination_reason", ""),
                )
            return {
                "raw_signals": result.get("raw_signals", []),
                "errors": result.get("errors", []),
                "mcp_tools_count": int(result.get("mcp_tools_count", 0)),
                "mcp_termination_reason": result.get("mcp_termination_reason", ""),
            }

        return await self._run_tracked_node_async(
            state,
            node_id="collect_signals_via_mcp",
            handler=_node_logic,
        )

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

    def retrieve_knowledge_evidence(self, state: ResearchState) -> ResearchState:
        """节点目标：检索知识库背景证据。

        前置条件：state 已包含 `query` 与 `hard_symbols`。
        状态产出：`knowledge_docs`。
        """

        def _node_logic() -> ResearchState:
            query = state["query"]
            hard_symbols = state.get("hard_symbols")
            symbols = hard_symbols if isinstance(hard_symbols, list) else state.get("symbols", [])

            with log_context(component="graph.retrieve_knowledge_evidence"):
                docs = self._retrieve_with_retry(query=query, symbols=symbols)
                fallback_used = False
                if not docs and symbols:
                    # 检索不足时放宽条件重试一次。
                    docs = self._retrieve_with_retry(query=query, symbols=[])
                    fallback_used = True
                    if not docs:
                        logger.warning("节点完成 docs=0 fallback=%s", fallback_used)
                        return {"knowledge_docs": [], "errors": [*state.get("errors", []), "知识库检索命中不足"]}

                logger.info("节点完成 docs=%s fallback=%s", len(docs), fallback_used)
                return {"knowledge_docs": docs}

        return self._run_tracked_node(state, node_id="retrieve_knowledge_evidence", handler=_node_logic)

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _retrieve_with_retry(self, query: str, symbols: list[str]):
        """检索重试包装。"""

        return self.research_service.retrieve_knowledge(query=query, symbols=symbols, top_k=8)

    def analyze_signals(self, state: ResearchState) -> ResearchState:
        """节点目标：评估信号强弱与冲突。

        前置条件：state 已包含 `signals` 与 `knowledge_docs`。
        状态产出：`analysis_summary`。
        """

        def _node_logic() -> ResearchState:
            signals = state.get("signals", [])
            docs = state.get("knowledge_docs", [])

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

        前置条件：state 已包含 `memory_profile`、`signals`、`knowledge_docs`。
        状态产出：`report_draft`、`final_report`、`citations`。
        """

        def _node_logic() -> ResearchState:
            payload = ReportGenerationInput(
                user_id=state["user_id"],
                query=state["query"],
                task_id=state["task_id"],
                memory_profile=state.get("memory_profile", {}),
                signals=state.get("signals", []),
                knowledge_docs=state.get("knowledge_docs", []),
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
                    conversation_id=state.get("conversation_id"),
                    turn_id=state.get("turn_id"),
                    request_id=state.get("request_id"),
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

    async def _run_tracked_node_async(self, state: ResearchState, node_id: str, handler) -> ResearchState:
        """异步执行节点并记录真实耗时与状态。"""

        start = perf_counter()
        try:
            update = await handler() or {}
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
    ) -> list[WorkflowStep]:
        """追加单个工作流节点执行记录。"""

        duration_ms = max(int((perf_counter() - start) * 1000), 0)
        step = WorkflowStep(node_id=node_id, status=status, duration_ms=duration_ms)
        return [*state.get("workflow_steps", []), step]

    def _extract_symbols_from_query(self, query: str) -> list[str]:
        pair_symbols = [match.upper() for match in PAIR_SYMBOL_PATTERN.findall(query or "")]
        token_symbols = [match.upper() for match in TOKEN_SYMBOL_PATTERN.findall(query or "")]
        merged = self._merge_symbols(pair_symbols, token_symbols)
        return [symbol for symbol in merged if symbol not in SYMBOL_STOPWORDS]

    def _extract_symbols_from_task_context(self, task_context: Any) -> list[str]:
        if not isinstance(task_context, dict):
            return []
        return self._normalize_symbols(task_context.get("symbols", []))

    def _normalize_symbols(self, symbols: Any) -> list[str]:
        if not isinstance(symbols, list):
            return []
        normalized: list[str] = []
        for item in symbols:
            text = str(item).strip().upper()
            if not text or not (2 <= len(text) <= 10):
                continue
            if not re.fullmatch(r"[A-Z0-9]{2,10}", text):
                continue
            normalized.append(text)
        return self._dedupe_keep_order(normalized)

    def _merge_symbols(self, *symbol_groups: list[str]) -> list[str]:
        merged: list[str] = []
        for group in symbol_groups:
            merged.extend(group)
        return self._dedupe_keep_order(merged)

    def _dedupe_keep_order(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result

    def _raw_signal_from_dict(self, item: dict[str, Any]):
        """将字典恢复为 RawSignal。"""

        from app.models.schemas import RawSignal

        return RawSignal.model_validate(item)
