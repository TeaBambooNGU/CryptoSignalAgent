"""会话一致性服务。"""

from __future__ import annotations

from time import perf_counter
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from app.conversation.store import ConversationLockManager, SQLiteConversationTruthStore
from app.graph.workflow import ResearchGraphRunner
from app.models.schemas import ConversationAction, ConversationMessageResponse, QueryResponse


class ConversationService:
    """封装会话串行、幂等、对话路由与报告版本管理。"""

    def __init__(
        self,
        *,
        graph_runner: ResearchGraphRunner,
        truth_store: SQLiteConversationTruthStore,
        lock_manager: ConversationLockManager | None = None,
    ) -> None:
        self.graph_runner = graph_runner
        self.truth_store = truth_store
        self.lock_manager = lock_manager or ConversationLockManager()

    async def run_research_turn(
        self,
        *,
        user_id: str,
        query: str,
        task_context: dict[str, Any] | None,
        trace_id: str,
        conversation_id: str | None,
        turn_id: str | None,
        request_id: str | None,
        expected_version: int | None,
    ) -> QueryResponse:
        """在一致性约束下执行一次研报请求。"""

        resolved_conversation_id = (conversation_id or "").strip() or f"default:{user_id}"
        resolved_request_id = (request_id or "").strip() or str(uuid4())

        async with self.lock_manager.acquire(resolved_conversation_id):
            prepared = self.truth_store.prepare_turn(
                conversation_id=resolved_conversation_id,
                turn_id=turn_id,
                request_id=resolved_request_id,
                expected_version=expected_version,
            )

            if prepared.cached_response is not None:
                cached = QueryResponse.model_validate(prepared.cached_response)
                return cached.model_copy(update={"trace_id": trace_id})

            try:
                response = await self.graph_runner.arun(
                    user_id=user_id,
                    query=query,
                    task_context=task_context,
                    trace_id=trace_id,
                    conversation_id=prepared.conversation_id,
                    turn_id=prepared.turn_id,
                    request_id=prepared.request_id,
                    conversation_version=prepared.conversation_version,
                )
            except Exception as exc:
                self.truth_store.save_turn_failure(
                    request_id=prepared.request_id,
                    user_id=user_id,
                    query_text=query,
                    task_context=task_context,
                    trace_id=trace_id,
                    error_text=type(exc).__name__,
                    intent=ConversationAction.REGENERATE_REPORT.value,
                    turn_type="assistant_report",
                )
                raise

            serialized_response = response.model_dump(mode="json")
            report_payload = {
                "mode": "regenerate",
                "report": serialized_response["report"],
                "citations": serialized_response.get("citations", []),
                "workflow_steps": serialized_response.get("workflow_steps", []),
            }
            report_meta = self.truth_store.save_turn_result(
                request_id=prepared.request_id,
                user_id=user_id,
                query_text=query,
                task_context=task_context,
                assistant_message=response.report,
                response=serialized_response,
                intent=ConversationAction.REGENERATE_REPORT.value,
                turn_type="assistant_report",
                report_payload=report_payload,
            )
            self._refresh_context_summary(conversation_id=prepared.conversation_id)
            if report_meta is None:
                return response
            return response.model_copy(
                update={
                    "report": str(report_meta.get("report", response.report)),
                    "citations": report_meta.get("citations", response.citations),
                    "workflow_steps": report_meta.get("workflow_steps", response.workflow_steps),
                }
            )

    async def send_message(
        self,
        *,
        user_id: str,
        message: str,
        conversation_id: str | None,
        trace_id: str,
        task_context: dict[str, Any] | None,
        action: ConversationAction,
        target_report_id: str | None,
        from_turn_id: str | None,
        request_id: str | None,
        expected_version: int | None,
    ) -> ConversationMessageResponse:
        """统一会话消息入口。"""

        resolved_conversation_id = (conversation_id or "").strip() or f"default:{user_id}"
        resolved_request_id = (request_id or "").strip() or str(uuid4())

        if from_turn_id:
            turn = self.truth_store.get_turn(conversation_id=resolved_conversation_id, turn_id=from_turn_id)
            if turn is None:
                raise ValueError(f"from_turn_id not found: {from_turn_id}")

        async with self.lock_manager.acquire(resolved_conversation_id):
            prepared = self.truth_store.prepare_turn(
                conversation_id=resolved_conversation_id,
                turn_id=None,
                request_id=resolved_request_id,
                expected_version=expected_version,
            )

            if prepared.cached_response is not None:
                cached = ConversationMessageResponse.model_validate(prepared.cached_response)
                return cached.model_copy(update={"trace_id": trace_id})

            resolved_action = self._resolve_action(
                action=action,
                message=message,
                conversation_id=prepared.conversation_id,
            )
            try:
                if resolved_action == ConversationAction.CHAT:
                    response = self._execute_chat(
                        user_id=user_id,
                        conversation_id=prepared.conversation_id,
                        message=message,
                        trace_id=trace_id,
                        prepared=prepared,
                        task_context=task_context,
                        parent_turn_id=from_turn_id,
                    )
                elif resolved_action == ConversationAction.REWRITE_REPORT:
                    response = self._execute_rewrite_report(
                        user_id=user_id,
                        conversation_id=prepared.conversation_id,
                        message=message,
                        trace_id=trace_id,
                        prepared=prepared,
                        task_context=task_context,
                        target_report_id=target_report_id,
                        parent_turn_id=from_turn_id,
                    )
                else:
                    response = await self._execute_regenerate_report(
                        user_id=user_id,
                        conversation_id=prepared.conversation_id,
                        message=message,
                        trace_id=trace_id,
                        prepared=prepared,
                        task_context=task_context,
                        parent_turn_id=from_turn_id,
                    )
            except Exception as exc:
                self.truth_store.save_turn_failure(
                    request_id=prepared.request_id,
                    user_id=user_id,
                    query_text=message,
                    task_context=task_context,
                    trace_id=trace_id,
                    error_text=type(exc).__name__,
                    intent=resolved_action.value,
                    turn_type="assistant_chat"
                    if resolved_action == ConversationAction.CHAT
                    else "assistant_report",
                    parent_turn_id=from_turn_id,
                )
                raise

            self._refresh_context_summary(conversation_id=prepared.conversation_id)
            return response

    async def resume_research_turn(
        self,
        *,
        user_id: str,
        query: str,
        conversation_id: str,
        trace_id: str,
        task_context: dict[str, Any] | None,
        from_turn_id: str | None,
        request_id: str | None,
        expected_version: int | None,
    ) -> QueryResponse:
        """从指定会话继续执行下一轮。"""

        if from_turn_id:
            turn = self.truth_store.get_turn(conversation_id=conversation_id, turn_id=from_turn_id)
            if turn is None:
                raise ValueError(f"from_turn_id not found: {from_turn_id}")

        return await self.run_research_turn(
            user_id=user_id,
            query=query,
            task_context=task_context,
            trace_id=trace_id,
            conversation_id=conversation_id,
            turn_id=None,
            request_id=request_id,
            expected_version=expected_version,
        )

    def get_conversation_meta(self, conversation_id: str) -> dict[str, Any] | None:
        return self.truth_store.get_conversation_meta(conversation_id)

    def list_conversation_turns(
        self,
        *,
        conversation_id: str,
        limit: int = 20,
        before_version: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.truth_store.list_turns(
            conversation_id=conversation_id,
            limit=limit,
            before_version=before_version,
        )

    def get_conversation_turn(self, *, conversation_id: str, turn_id: str) -> dict[str, Any] | None:
        return self.truth_store.get_turn(conversation_id=conversation_id, turn_id=turn_id)

    def list_conversation_reports(
        self,
        *,
        conversation_id: str,
        limit: int = 20,
        before_report_version: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.truth_store.list_reports(
            conversation_id=conversation_id,
            limit=limit,
            before_report_version=before_report_version,
        )

    def get_conversation_report(self, *, report_id: str) -> dict[str, Any] | None:
        return self.truth_store.get_report(report_id=report_id)

    def _resolve_action(
        self,
        *,
        action: ConversationAction,
        message: str,
        conversation_id: str,
    ) -> ConversationAction:
        if action != ConversationAction.AUTO:
            return action

        lowered = message.lower()
        generate_keywords = (
            "生成研报",
            "生成报告",
            "生成一版",
            "生成一份",
            "给我一版",
            "先给我一版",
            "来一版",
            "出一版",
            "出个报告",
            "产出报告",
            "generate report",
            "draft report",
        )
        rewrite_keywords = ("重写", "改写", "润色", "改成", "改为")
        regenerate_keywords = ("重跑", "重新生成", "重做", "按最新", "重新分析", "regenerate", "rerun")

        if any(keyword in lowered for keyword in generate_keywords):
            return ConversationAction.REGENERATE_REPORT
        if any(keyword in lowered for keyword in regenerate_keywords):
            return ConversationAction.REGENERATE_REPORT
        if any(keyword in lowered for keyword in rewrite_keywords):
            return ConversationAction.REWRITE_REPORT
        return ConversationAction.CHAT

    def _execute_chat(
        self,
        *,
        user_id: str,
        conversation_id: str,
        message: str,
        trace_id: str,
        prepared: Any,
        task_context: dict[str, Any] | None,
        parent_turn_id: str | None,
    ) -> ConversationMessageResponse:
        llm = self.graph_runner.report_agent.llm
        context_prompt = self._build_context_prompt(conversation_id=conversation_id, include_latest_report=True)
        start = perf_counter()
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "你是加密研究助手。你正在围绕同一份研报与用户持续对话。"
                        "请先基于现有上下文回答；若用户需求更适合重跑或重写报告，请明确建议。"
                    )
                ),
                HumanMessage(
                    content=(
                        f"用户ID: {user_id}\n"
                        f"会话ID: {conversation_id}\n"
                        f"上下文:\n{context_prompt}\n\n"
                        f"用户消息:\n{message}"
                    )
                ),
            ]
        )
        assistant_message = self._extract_text(getattr(response, "content", response))
        elapsed_ms = int((perf_counter() - start) * 1000)
        response_payload = {
            "conversation_id": conversation_id,
            "turn_id": prepared.turn_id,
            "request_id": prepared.request_id,
            "conversation_version": prepared.conversation_version,
            "action_taken": ConversationAction.CHAT.value,
            "assistant_message": assistant_message,
            "trace_id": trace_id,
            "errors": [],
            "workflow_steps": [{"node_id": "chat_reply", "status": "success", "duration_ms": max(elapsed_ms, 0)}],
            "report": None,
            "citations": [],
        }
        self.truth_store.save_turn_result(
            request_id=prepared.request_id,
            user_id=user_id,
            query_text=message,
            task_context=task_context,
            assistant_message=assistant_message,
            response=response_payload,
            intent=ConversationAction.CHAT.value,
            turn_type="assistant_chat",
            parent_turn_id=parent_turn_id,
            report_payload=None,
        )
        return ConversationMessageResponse.model_validate(response_payload)

    def _execute_rewrite_report(
        self,
        *,
        user_id: str,
        conversation_id: str,
        message: str,
        trace_id: str,
        prepared: Any,
        task_context: dict[str, Any] | None,
        target_report_id: str | None,
        parent_turn_id: str | None,
    ) -> ConversationMessageResponse:
        source_report = (
            self.truth_store.get_report(report_id=target_report_id) if target_report_id else None
        ) or self.truth_store.get_latest_report(conversation_id=conversation_id)
        if source_report is None:
            raise ValueError("no report available for rewrite")

        llm = self.graph_runner.report_agent.llm
        start = perf_counter()
        rewritten = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "你是加密研究助手。请在不虚构事实的前提下，按照用户要求改写报告。"
                        "保留核心证据与风险提示，可调整结构、语气和重点。"
                    )
                ),
                HumanMessage(
                    content=(
                        f"用户ID: {user_id}\n"
                        f"会话ID: {conversation_id}\n"
                        f"原报告:\n{source_report['report']}\n\n"
                        f"改写要求:\n{message}"
                    )
                ),
            ]
        )
        rewritten_text = self._extract_text(getattr(rewritten, "content", rewritten))
        duration_ms = int((perf_counter() - start) * 1000)
        workflow_steps = [{"node_id": "rewrite_report", "status": "success", "duration_ms": max(duration_ms, 0)}]
        report_payload = {
            "mode": "rewrite",
            "based_on_report_id": source_report["report_id"],
            "report": rewritten_text,
            "citations": source_report.get("citations", []),
            "workflow_steps": workflow_steps,
        }
        response_payload = {
            "conversation_id": conversation_id,
            "turn_id": prepared.turn_id,
            "request_id": prepared.request_id,
            "conversation_version": prepared.conversation_version,
            "action_taken": ConversationAction.REWRITE_REPORT.value,
            "assistant_message": rewritten_text,
            "trace_id": trace_id,
            "errors": [],
            "workflow_steps": workflow_steps,
            "report": None,
            "citations": source_report.get("citations", []),
        }
        saved_report = self.truth_store.save_turn_result(
            request_id=prepared.request_id,
            user_id=user_id,
            query_text=message,
            task_context=task_context,
            assistant_message=rewritten_text,
            response=response_payload,
            intent=ConversationAction.REWRITE_REPORT.value,
            turn_type="assistant_report",
            parent_turn_id=parent_turn_id,
            report_payload=report_payload,
        )
        response_payload["report"] = saved_report
        self.truth_store.update_idempotency_response(
            request_id=prepared.request_id,
            response=response_payload,
        )
        return ConversationMessageResponse.model_validate(
            response_payload
        )

    async def _execute_regenerate_report(
        self,
        *,
        user_id: str,
        conversation_id: str,
        message: str,
        trace_id: str,
        prepared: Any,
        task_context: dict[str, Any] | None,
        parent_turn_id: str | None,
    ) -> ConversationMessageResponse:
        response = await self.graph_runner.arun(
            user_id=user_id,
            query=message,
            task_context=task_context,
            trace_id=trace_id,
            conversation_id=prepared.conversation_id,
            turn_id=prepared.turn_id,
            request_id=prepared.request_id,
            conversation_version=prepared.conversation_version,
        )
        serialized_response = response.model_dump(mode="json")
        report_payload = {
            "mode": "regenerate",
            "report": serialized_response["report"],
            "citations": serialized_response.get("citations", []),
            "workflow_steps": serialized_response.get("workflow_steps", []),
        }
        response_payload = {
            "conversation_id": response.conversation_id,
            "turn_id": response.turn_id,
            "request_id": response.request_id,
            "conversation_version": response.conversation_version,
            "action_taken": ConversationAction.REGENERATE_REPORT.value,
            "assistant_message": response.report,
            "trace_id": trace_id,
            "errors": serialized_response.get("errors", []),
            "workflow_steps": serialized_response.get("workflow_steps", []),
            "report": None,
            "citations": serialized_response.get("citations", []),
        }
        saved_report = self.truth_store.save_turn_result(
            request_id=prepared.request_id,
            user_id=user_id,
            query_text=message,
            task_context=task_context,
            assistant_message=response.report,
            response=response_payload,
            intent=ConversationAction.REGENERATE_REPORT.value,
            turn_type="assistant_report",
            parent_turn_id=parent_turn_id,
            report_payload=report_payload,
        )
        response_payload["report"] = saved_report
        self.truth_store.update_idempotency_response(
            request_id=prepared.request_id,
            response=response_payload,
        )
        return ConversationMessageResponse.model_validate(
            response_payload
        )

    def _refresh_context_summary(self, *, conversation_id: str, recent_window: int = 8) -> None:
        """为长会话维护摘要。"""

        meta = self.truth_store.get_conversation_meta(conversation_id)
        if not meta:
            return
        latest_version = int(meta.get("latest_version", 0) or 0)
        if latest_version <= recent_window:
            return
        through_version = latest_version - recent_window
        summary_row = self.truth_store.get_context_summary(conversation_id=conversation_id)
        if summary_row and int(summary_row.get("through_version", 0)) >= through_version:
            return
        turns = self.truth_store.list_turns_up_to_version(
            conversation_id=conversation_id,
            through_version=through_version,
            limit=200,
        )
        if not turns:
            return
        lines: list[str] = []
        for row in turns[-24:]:
            role_text = f"v{row.get('version')}[{row.get('intent', '')}]"
            query = str(row.get("query", "")).strip().replace("\n", " ")
            answer = str(row.get("assistant_message", "")).strip().replace("\n", " ")
            if len(answer) > 120:
                answer = answer[:120] + "..."
            lines.append(f"- {role_text} Q:{query} A:{answer}")
        summary_text = "\n".join(lines)
        self.truth_store.upsert_context_summary(
            conversation_id=conversation_id,
            summary_text=summary_text,
            through_version=through_version,
        )

    def _build_context_prompt(self, *, conversation_id: str, include_latest_report: bool) -> str:
        summary = self.truth_store.get_context_summary(conversation_id=conversation_id)
        recent_turns = self.truth_store.list_turns(conversation_id=conversation_id, limit=8)
        recent_turns = list(reversed(recent_turns))
        sections: list[str] = []
        if summary:
            sections.append(
                f"[历史摘要(截至v{summary['through_version']})]\n{summary['summary_text']}"
            )
        if include_latest_report:
            latest_report = self.truth_store.get_latest_report(conversation_id=conversation_id)
            if latest_report:
                sections.append(
                    f"[最新报告 v{latest_report['report_version']}]\n{latest_report['report']}"
                )
        if recent_turns:
            lines = []
            for turn in recent_turns:
                lines.append(
                    f"- v{turn['version']}[{turn.get('intent', '')}] Q:{turn.get('query', '')} "
                    f"A:{turn.get('assistant_message', '')}"
                )
            sections.append("[最近轮次]\n" + "\n".join(lines))
        return "\n\n".join(sections) if sections else "暂无历史上下文"

    @staticmethod
    def _extract_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
                        continue
                chunks.append(str(item))
            return "".join(chunks).strip()
        return str(content).strip()
