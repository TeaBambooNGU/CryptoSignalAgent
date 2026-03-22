"""会话上下文压缩与全文摘要管理器。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.config.settings import Settings
from app.conversation.token_counter import TokenCounter

logger = logging.getLogger(__name__)


class ConversationContextManager:
    """管理上下文预算、压缩归档与全文摘要。"""

    def __init__(
        self,
        *,
        settings: Settings,
        truth_store: Any,
        compression_llm: BaseChatModel | Any | None,
        full_summary_llm: BaseChatModel | Any,
    ) -> None:
        self.settings = settings
        self.truth_store = truth_store
        self.compression_llm = compression_llm
        self.full_summary_llm = full_summary_llm
        self.token_counter = TokenCounter()
        self.archive_dir = Path(settings.context_archive_dir)
        if not self.archive_dir.is_absolute():
            self.archive_dir = Path.cwd() / self.archive_dir

    def ensure_context_budget(
        self,
        *,
        conversation_id: str,
        anchor_turn_id: str | None,
        include_latest_report: bool,
        pending_user_message: str,
    ) -> dict[str, Any]:
        """按预算规则触发一次压缩，并在必要时同步做全文摘要。"""

        state = self._get_context_state(conversation_id=conversation_id, anchor_turn_id=anchor_turn_id)
        token_estimate = self.estimate_prompt_tokens(
            conversation_id=conversation_id,
            anchor_turn_id=anchor_turn_id,
            include_latest_report=include_latest_report,
            pending_user_message=pending_user_message,
        )
        logger.info(
            "context.budget.evaluated conversation_id=%s anchor_turn_id=%s token_estimate=%s trigger_tokens=%s hard_limit_tokens=%s include_latest_report=%s",
            conversation_id,
            anchor_turn_id or "__mainline__",
            token_estimate,
            self.settings.context_compression_trigger_tokens,
            self.settings.context_compression_hard_limit_tokens,
            include_latest_report,
        )

        did_compress = False
        if (
            token_estimate >= self.settings.context_compression_trigger_tokens
            and self.compression_llm is not None
        ):
            logger.info(
                "context.compress.triggered conversation_id=%s anchor_turn_id=%s token_estimate=%s latest_materialized_version=%s compression_round_count=%s",
                conversation_id,
                anchor_turn_id or "__mainline__",
                token_estimate,
                int(state.get("latest_materialized_version", 0) or 0),
                int(state.get("compression_round_count", 0) or 0),
            )
            did_compress, state = self._compress_once(
                conversation_id=conversation_id,
                anchor_turn_id=anchor_turn_id,
                state=state,
            )
            if did_compress and state["compressions_since_full_summary"] >= self.settings.context_full_summary_every_n_compressions:
                logger.info(
                    "context.full_summary.triggered conversation_id=%s anchor_turn_id=%s compression_round_count=%s source_version_upto=%s",
                    conversation_id,
                    anchor_turn_id or "__mainline__",
                    int(state.get("compression_round_count", 0) or 0),
                    int(state.get("latest_materialized_version", 0) or 0),
                )
                state = self._refresh_full_summary(
                    conversation_id=conversation_id,
                    anchor_turn_id=anchor_turn_id,
                    state=state,
                )
            token_estimate = self.estimate_prompt_tokens(
                conversation_id=conversation_id,
                anchor_turn_id=anchor_turn_id,
                include_latest_report=include_latest_report,
                pending_user_message=pending_user_message,
            )

        self.truth_store.upsert_context_state(
            conversation_id=conversation_id,
            anchor_turn_id=anchor_turn_id,
            latest_materialized_version=state["latest_materialized_version"],
            compression_round_count=state["compression_round_count"],
            compressions_since_full_summary=state["compressions_since_full_summary"],
            last_full_summary_round=state["last_full_summary_round"],
            last_full_summary_version=state["last_full_summary_version"],
            latest_prompt_token_estimate=token_estimate,
        )
        return {
            "did_compress": did_compress,
            "token_estimate": token_estimate,
            "hard_limit_exceeded": token_estimate > self.settings.context_compression_hard_limit_tokens,
        }

    def build_context_prompt(
        self,
        *,
        conversation_id: str,
        anchor_turn_id: str | None,
        include_latest_report: bool,
    ) -> str:
        sections = self._build_materialized_sections(
            conversation_id=conversation_id,
            anchor_turn_id=anchor_turn_id,
            include_latest_report=include_latest_report,
        )
        return "\n\n".join(section for section in sections if section) if sections else "暂无历史上下文"

    def build_summary_snapshot(
        self,
        *,
        conversation_id: str | None,
        anchor_turn_id: str | None,
    ) -> dict[str, Any] | None:
        if not conversation_id:
            return None
        full_summary = self.truth_store.get_full_context_summary(
            conversation_id=conversation_id,
            anchor_turn_id=anchor_turn_id,
        )
        after_round = int(full_summary.get("source_round_upto", 0) or 0) if full_summary else None
        compressions = self.truth_store.list_context_compressions(
            conversation_id=conversation_id,
            anchor_turn_id=anchor_turn_id,
            after_round_no=after_round,
        )
        parts: list[str] = []
        through_version = 0
        updated_at = 0
        if full_summary:
            parts.append(f"[全文摘要]\n{full_summary['summary_text']}")
            through_version = max(through_version, int(full_summary.get("source_version_upto", 0) or 0))
            updated_at = max(updated_at, int(full_summary.get("updated_at", 0) or 0))
        if compressions:
            parts.extend(
                f"[压缩轮次 #{item['round_no']} v{item['source_start_version']}-v{item['source_end_version']}]\n{item['compressed_text']}"
                for item in compressions
            )
            through_version = max(through_version, int(compressions[-1].get("source_end_version", 0) or 0))
            updated_at = max(updated_at, int(compressions[-1].get("updated_at", 0) or 0))
        if not parts:
            return None
        return {
            "through_version": through_version,
            "summary_text": "\n\n".join(parts),
            "updated_at": updated_at,
        }

    def estimate_prompt_tokens(
        self,
        *,
        conversation_id: str,
        anchor_turn_id: str | None,
        include_latest_report: bool,
        pending_user_message: str,
    ) -> int:
        prompt_text = self.build_context_prompt(
            conversation_id=conversation_id,
            anchor_turn_id=anchor_turn_id,
            include_latest_report=include_latest_report,
        )
        return self.token_counter.count_sections([prompt_text, pending_user_message])

    def _get_context_state(
        self,
        *,
        conversation_id: str,
        anchor_turn_id: str | None,
    ) -> dict[str, Any]:
        row = self.truth_store.get_context_state_for_anchor(
            conversation_id=conversation_id,
            anchor_turn_id=anchor_turn_id,
        )
        if row is not None:
            return row
        return {
            "conversation_id": conversation_id,
            "anchor_turn_id": anchor_turn_id,
            "latest_materialized_version": 0,
            "compression_round_count": 0,
            "compressions_since_full_summary": 0,
            "last_full_summary_round": 0,
            "last_full_summary_version": 0,
            "latest_prompt_token_estimate": 0,
            "updated_at": 0,
        }

    def _build_materialized_sections(
        self,
        *,
        conversation_id: str,
        anchor_turn_id: str | None,
        include_latest_report: bool,
    ) -> list[str]:
        sections: list[str] = []
        full_summary = self.truth_store.get_full_context_summary(
            conversation_id=conversation_id,
            anchor_turn_id=anchor_turn_id,
        )
        after_round = int(full_summary.get("source_round_upto", 0) or 0) if full_summary else None
        if full_summary:
            sections.append(f"[全文摘要]\n{full_summary['summary_text']}")
        compressions = self.truth_store.list_context_compressions(
            conversation_id=conversation_id,
            anchor_turn_id=anchor_turn_id,
            after_round_no=after_round,
        )
        if compressions:
            sections.append(
                "[增量压缩片段]\n"
                + "\n\n".join(
                    f"## 压缩轮次 #{item['round_no']} (v{item['source_start_version']}-v{item['source_end_version']})\n{item['compressed_text']}"
                    for item in compressions
                )
            )

        state = self._get_context_state(conversation_id=conversation_id, anchor_turn_id=anchor_turn_id)
        lineage_turns = self._list_target_turns(conversation_id=conversation_id, anchor_turn_id=anchor_turn_id)
        remaining_turns = [
            turn
            for turn in lineage_turns
            if int(turn.get("version", 0) or 0) > int(state.get("latest_materialized_version", 0) or 0)
        ]
        if remaining_turns:
            sections.append("[未压缩原始轮次]\n" + self._format_turns_for_prompt(remaining_turns))

        if include_latest_report:
            latest_report = (
                self.truth_store.get_latest_report_on_lineage(
                    conversation_id=conversation_id,
                    leaf_turn_id=anchor_turn_id,
                )
                if anchor_turn_id
                else self.truth_store.get_latest_report(conversation_id=conversation_id)
            )
            if latest_report:
                sections.append(f"[最新报告 v{latest_report['report_version']}]\n{latest_report['report']}")
        return sections

    def _compress_once(
        self,
        *,
        conversation_id: str,
        anchor_turn_id: str | None,
        state: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        lineage_turns = self._list_target_turns(conversation_id=conversation_id, anchor_turn_id=anchor_turn_id)
        if not lineage_turns:
            return False, state
        if len(lineage_turns) <= self.settings.context_recent_turn_window:
            return False, state

        compressible_turns = lineage_turns[: -self.settings.context_recent_turn_window]
        source_turns = [
            turn
            for turn in compressible_turns
            if int(turn.get("version", 0) or 0) > int(state.get("latest_materialized_version", 0) or 0)
        ]
        if not source_turns:
            return False, state

        round_no = int(state.get("compression_round_count", 0) or 0) + 1
        source_start_version = int(source_turns[0].get("version", 0) or 0)
        source_end_version = int(source_turns[-1].get("version", 0) or 0)
        raw_text = self._format_turns_for_archive(source_turns)
        raw_token_estimate = self.token_counter.count_text(raw_text)
        raw_file_path = self._write_archive_file(
            conversation_id=conversation_id,
            anchor_turn_id=anchor_turn_id,
            round_no=round_no,
            start_version=source_start_version,
            end_version=source_end_version,
            raw_text=raw_text,
        )
        compressed_text = self._invoke_compression_llm(
            conversation_id=conversation_id,
            anchor_turn_id=anchor_turn_id,
            start_version=source_start_version,
            end_version=source_end_version,
            raw_text=raw_text,
        )
        compressed_token_estimate = self.token_counter.count_text(compressed_text)
        self.truth_store.insert_context_compression(
            compression_id=f"ctxcmp-{uuid4()}",
            conversation_id=conversation_id,
            anchor_turn_id=anchor_turn_id,
            round_no=round_no,
            source_start_version=source_start_version,
            source_end_version=source_end_version,
            source_turn_ids=[str(item.get("turn_id", "")) for item in source_turns],
            raw_file_path=raw_file_path,
            raw_token_estimate=raw_token_estimate,
            compressed_text=compressed_text,
            compressed_token_estimate=compressed_token_estimate,
            model_name=self.settings.context_compression_model,
        )
        logger.info(
            "context.compress.completed conversation_id=%s anchor_turn_id=%s round_no=%s source_start_version=%s source_end_version=%s raw_token_estimate=%s compressed_token_estimate=%s raw_file_path=%s model_name=%s",
            conversation_id,
            anchor_turn_id or "__mainline__",
            round_no,
            source_start_version,
            source_end_version,
            raw_token_estimate,
            compressed_token_estimate,
            raw_file_path,
            self.settings.context_compression_model,
        )
        return True, {
            **state,
            "latest_materialized_version": source_end_version,
            "compression_round_count": round_no,
            "compressions_since_full_summary": int(state.get("compressions_since_full_summary", 0) or 0) + 1,
        }

    def _refresh_full_summary(
        self,
        *,
        conversation_id: str,
        anchor_turn_id: str | None,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        compressions = self.truth_store.list_context_compressions(
            conversation_id=conversation_id,
            anchor_turn_id=anchor_turn_id,
        )
        if not compressions:
            return state

        raw_manifest: list[dict[str, Any]] = []
        source_turn_ids: list[str] = []
        raw_parts: list[str] = []
        max_version = 0
        for item in compressions:
            raw_file_path = Path(str(item.get("raw_file_path", "")))
            if not raw_file_path.exists():
                logger.warning("全文摘要跳过缺失 raw file path=%s", raw_file_path)
                continue
            raw_text = raw_file_path.read_text(encoding="utf-8")
            raw_parts.append(raw_text)
            source_turn_ids.extend(str(turn_id) for turn_id in item.get("source_turn_ids", []))
            max_version = max(max_version, int(item.get("source_end_version", 0) or 0))
            raw_manifest.append(
                {
                    "round_no": int(item.get("round_no", 0) or 0),
                    "raw_file_path": str(raw_file_path),
                    "source_start_version": int(item.get("source_start_version", 0) or 0),
                    "source_end_version": int(item.get("source_end_version", 0) or 0),
                }
            )
        if not raw_parts:
            return state

        joined_raw_text = "\n\n".join(raw_parts)
        response = self.full_summary_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "你是会话全文摘要器。"
                        "请基于完整原始对话文本生成可长期复用的中文摘要。"
                        "必须保留：用户目标、稳定偏好、关键结论、证据、未决问题、后续约束。"
                        "禁止虚构未出现的事实。"
                    )
                ),
                HumanMessage(
                    content=(
                        f"会话ID: {conversation_id}\n"
                        f"Anchor: {anchor_turn_id or '(mainline)'}\n"
                        f"压缩轮次总数: {len(compressions)}\n\n"
                        f"原始文本:\n\n{joined_raw_text}"
                    )
                ),
            ]
        )
        summary_text = self._extract_text(getattr(response, "content", response))
        self.truth_store.upsert_full_context_summary(
            conversation_id=conversation_id,
            anchor_turn_id=anchor_turn_id,
            summary_text=summary_text,
            source_round_upto=int(state.get("compression_round_count", 0) or 0),
            source_version_upto=max_version,
            source_turn_ids=source_turn_ids,
            raw_manifest=raw_manifest,
            model_name=self.settings.llm_model,
        )
        logger.info(
            "context.full_summary.completed conversation_id=%s anchor_turn_id=%s source_round_upto=%s source_version_upto=%s raw_file_count=%s model_name=%s summary_length=%s",
            conversation_id,
            anchor_turn_id or "__mainline__",
            int(state.get("compression_round_count", 0) or 0),
            max_version,
            len(raw_manifest),
            self.settings.llm_model,
            len(summary_text),
        )
        return {
            **state,
            "compressions_since_full_summary": 0,
            "last_full_summary_round": int(state.get("compression_round_count", 0) or 0),
            "last_full_summary_version": max_version,
        }

    def _invoke_compression_llm(
        self,
        *,
        conversation_id: str,
        anchor_turn_id: str | None,
        start_version: int,
        end_version: int,
        raw_text: str,
    ) -> str:
        response = self.compression_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "你是会话上下文压缩器。"
                        "请将原始多轮对话压缩成后续推理可直接复用的中文上下文。"
                        "必须保留：用户目标与关注标的、稳定偏好、报告版本演进、关键事实、未决事项。"
                        "删除寒暄、重复表达和低价值细节。"
                    )
                ),
                HumanMessage(
                    content=(
                        f"会话ID: {conversation_id}\n"
                        f"Anchor: {anchor_turn_id or '(mainline)'}\n"
                        f"版本范围: v{start_version}-v{end_version}\n\n"
                        f"原始文本:\n\n{raw_text}"
                    )
                ),
            ]
        )
        return self._extract_text(getattr(response, "content", response))

    def _write_archive_file(
        self,
        *,
        conversation_id: str,
        anchor_turn_id: str | None,
        round_no: int,
        start_version: int,
        end_version: int,
        raw_text: str,
    ) -> str:
        anchor_key = self._anchor_path_key(anchor_turn_id)
        conversation_dir = self.archive_dir / conversation_id / anchor_key
        conversation_dir.mkdir(parents=True, exist_ok=True)
        file_path = conversation_dir / f"round_{round_no:04d}_v{start_version}_v{end_version}.md"
        file_path.write_text(raw_text, encoding="utf-8")
        return str(file_path)

    def _list_target_turns(
        self,
        *,
        conversation_id: str,
        anchor_turn_id: str | None,
    ) -> list[dict[str, Any]]:
        if anchor_turn_id:
            lineage = self.truth_store.list_turn_lineage(
                conversation_id=conversation_id,
                leaf_turn_id=anchor_turn_id,
                limit=500,
            )
            return list(reversed(lineage))
        meta = self.truth_store.get_conversation_meta(conversation_id)
        limit = min(max(int(meta.get("turn_count", 0) or 0), 1), 500) if meta else 500
        return list(reversed(self.truth_store.list_turns(conversation_id=conversation_id, limit=limit)))

    def _anchor_path_key(self, anchor_turn_id: str | None) -> str:
        return str(anchor_turn_id).strip() if anchor_turn_id else "__mainline__"

    def _format_turns_for_archive(self, turns: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        for turn in turns:
            block = [
                f"## Turn v{turn.get('version')} [{turn.get('intent', '')}]",
                f"turn_id: {turn.get('turn_id', '')}",
                f"user_query:\n{turn.get('query', '')}",
            ]
            assistant_message = str(turn.get("assistant_message", "") or "")
            report_text = str(turn.get("report", "") or "")
            if assistant_message:
                block.append(f"assistant_message:\n{assistant_message}")
            if report_text and report_text != assistant_message:
                block.append(f"report:\n{report_text}")
            blocks.append("\n\n".join(block))
        return "\n\n---\n\n".join(blocks)

    def _format_turns_for_prompt(self, turns: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for turn in turns:
            query = str(turn.get("query", "")).strip().replace("\n", " ")
            answer = str(turn.get("assistant_message", "") or turn.get("report", "")).strip().replace("\n", " ")
            lines.append(f"- v{turn.get('version')}[{turn.get('intent', '')}] Q:{query} A:{answer}")
        return "\n".join(lines)

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
                elif isinstance(item, str):
                    chunks.append(item)
            return "\n".join(part.strip() for part in chunks if part and part.strip()).strip()
        return str(content).strip()
