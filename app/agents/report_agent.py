"""研报生成代理。

负责将“用户问题 + 记忆画像 + 信号 + 知识证据”组织成结构化 Prompt，
再调用可替换 LLM 客户端生成最终报告草稿。
"""

from __future__ import annotations

from collections import Counter
import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.config.settings import Settings
from app.models.schemas import Citation, NormalizedSignal, ReportGenerationInput, ReportGenerationOutput, RetrievedChunk


class ReportAgent:
    """报告生成服务。"""

    def __init__(self, settings: Settings, llm: BaseChatModel) -> None:
        self.settings = settings
        self.llm = llm
        self.signal_detail_limit = max(1, int(settings.report_signal_detail_limit))
        self.signal_value_max_chars = max(1, int(settings.report_signal_value_max_chars))

    def generate(self, payload: ReportGenerationInput) -> ReportGenerationOutput:
        """生成结构化研报。"""

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(payload, evidence_docs=payload.knowledge_docs)

        response = self.llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        report_text = self._extract_text(getattr(response, "content", response))
        citations = self._build_citations(payload.knowledge_docs)
        final_report = self._append_disclaimer(report_text)

        return ReportGenerationOutput(
            report=final_report,
            draft=report_text,
            citations=citations,
        )

    def _build_system_prompt(self) -> str:
        """构造系统提示词，固定输出结构。"""

        return (
            "你是一名加密市场研究员。请基于给定证据生成中文结构化研报，"
            "必须包含：执行摘要、关键信号、分歧与风险、未来24小时观察点。"
            "若证据不足，明确写出不确定性与数据缺口。"
        )

    def _build_user_prompt(self, payload: ReportGenerationInput, *, evidence_docs: list[RetrievedChunk]) -> str:
        """将状态数据编排为用户提示词。"""

        signal_summary = self._summarize_signals(payload.signals)
        signal_details = self._build_signal_details(
            payload.signals,
            query=payload.query,
            memory_profile=payload.memory_profile,
        )
        docs_summary = self._summarize_docs(evidence_docs)
        reinforcement_signal = self._build_reinforcement_signal()
        query_reinforcement = self._build_query_reinforcement(payload.query)

        return (
            f"{reinforcement_signal}\n\n"
            f"用户ID: {payload.user_id}\n"
            f"任务ID: {payload.task_id}\n"
            f"用户问题: {payload.query}\n\n"
            f"用户记忆画像:\n{payload.memory_profile}\n\n"
            f"信号摘要:\n{signal_summary}\n\n"
            f"实时信号明细:\n{signal_details}\n\n"
            f"知识证据摘要:\n{docs_summary}\n\n"
            "输出要求:\n"
            "1) 先给出结论和置信度；\n"
            "2) 每个结论关联至少一条证据；\n"
            "3) 明确风险与反例；\n"
            "4) 使用中文，风格偏专业研报。\n\n"
            f"{reinforcement_signal}\n"
            f"{query_reinforcement}"
        )

    @staticmethod
    def _build_reinforcement_signal() -> str:
        """构造首尾复用的强化信号，避免长上下文下指令衰减。"""

        return (
            "【强化信号】请严格遵守：先给结论和置信度；每个结论绑定证据；"
            "明确风险与反例；保持中文专业研报风格。"
        )

    @staticmethod
    def _build_query_reinforcement(query: str) -> str:
        """将用户问题放在提示词最后一行，避免主诉求被长上下文稀释。"""

        normalized_query = " ".join(str(query or "").split())
        return f"【用户问题强化】{normalized_query}" if normalized_query else "【用户问题强化】(空)"

    def _summarize_signals(self, signals: list[NormalizedSignal]) -> str:
        """汇总信号统计信息。"""

        if not signals:
            return "暂无实时信号。"

        by_symbol = Counter(signal.symbol for signal in signals)
        by_type = Counter(signal.signal_type.value for signal in signals)

        top_lines = [
            f"- 信号总数: {len(signals)}",
            f"- 覆盖标的: {dict(by_symbol)}",
            f"- 类型分布: {dict(by_type)}",
        ]
        return "\n".join(top_lines)

    def _build_signal_details(
        self,
        signals: list[NormalizedSignal],
        *,
        query: str,
        memory_profile: dict[str, Any],
    ) -> str:
        """构造供 LLM 直接消费的实时信号明细。"""

        if not signals:
            return "暂无实时信号明细。"

        selected_signals = self._select_signals_for_prompt(
            signals,
            query=query,
            memory_profile=memory_profile,
        )
        focus_symbols = self._extract_focus_symbols(query=query, signals=signals, memory_profile=memory_profile)

        lines: list[str] = []
        if len(selected_signals) < len(signals):
            focus_text = ", ".join(sorted(focus_symbols)) if focus_symbols else "无显式焦点标的"
            lines.append(
                f"- 共 {len(signals)} 条，超过上限 {self.signal_detail_limit}；"
                f"已按“焦点标的优先，其余按置信度降序”保留 {len(selected_signals)} 条；"
                f"焦点标的: {focus_text}"
            )
        else:
            lines.append(f"- 共 {len(signals)} 条，全部已展示")

        for index, signal in enumerate(selected_signals, start=1):
            timestamp = signal.timestamp.isoformat() if signal.timestamp else ""
            raw_ref = str(signal.raw_ref or "")
            value_text = self._format_signal_payload(signal.value, max_chars=self.signal_value_max_chars)
            lines.append(
                f"{index}. symbol={signal.symbol} | type={signal.signal_type.value} | "
                f"confidence={signal.confidence:.2f} | source={signal.source} | timestamp={timestamp}"
            )
            lines.append(f"   raw_ref={raw_ref or '(空)'}")
            lines.append(f"   value={value_text}")

        return "\n".join(lines)

    def _select_signals_for_prompt(
        self,
        signals: list[NormalizedSignal],
        *,
        query: str,
        memory_profile: dict[str, Any],
    ) -> list[NormalizedSignal]:
        """选择写入 Prompt 的信号列表。"""

        if len(signals) <= self.signal_detail_limit:
            return list(signals)

        focus_symbols = self._extract_focus_symbols(query=query, signals=signals, memory_profile=memory_profile)
        ranked = sorted(
            enumerate(signals),
            key=lambda item: self._signal_rank_key(item[1], index=item[0], focus_symbols=focus_symbols),
            reverse=True,
        )
        return [signal for _, signal in ranked[: self.signal_detail_limit]]

    def _signal_rank_key(
        self,
        signal: NormalizedSignal,
        *,
        index: int,
        focus_symbols: set[str],
    ) -> tuple[int, float, int, int, int]:
        """计算信号截断排序键。"""

        normalized_symbol = str(signal.symbol or "").upper()
        return (
            1 if normalized_symbol in focus_symbols and normalized_symbol != "UNKNOWN" else 0,
            float(signal.confidence),
            1 if normalized_symbol != "UNKNOWN" else 0,
            1 if bool(signal.raw_ref) else 0,
            -index,
        )

    def _extract_focus_symbols(
        self,
        *,
        query: str,
        signals: list[NormalizedSignal],
        memory_profile: dict[str, Any],
    ) -> set[str]:
        """识别用户当前最关注的标的。"""

        candidates = [
            str(signal.symbol or "").upper()
            for signal in signals
            if str(signal.symbol or "").upper() and str(signal.symbol or "").upper() != "UNKNOWN"
        ]
        if not candidates:
            return set()

        query_text = str(query or "").upper()
        focus_from_query = {symbol for symbol in candidates if symbol in query_text}
        if focus_from_query:
            return focus_from_query

        focus_from_memory = {
            symbol
            for symbol in self._extract_symbols_from_memory_profile(memory_profile)
            if symbol in set(candidates)
        }
        return focus_from_memory

    def _extract_symbols_from_memory_profile(self, memory_profile: dict[str, Any]) -> set[str]:
        """从用户记忆中提取可能的标的偏好。"""

        symbols: set[str] = set()
        if not isinstance(memory_profile, dict):
            return symbols

        for key in ("watchlist", "symbols"):
            value = memory_profile.get(key)
            if isinstance(value, list):
                symbols.update(str(item).upper() for item in value if str(item).strip())

        session_memory = memory_profile.get("session_memory")
        if isinstance(session_memory, list):
            for item in session_memory:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if not isinstance(content, dict):
                    continue
                value = content.get("symbols")
                if isinstance(value, list):
                    symbols.update(str(symbol).upper() for symbol in value if str(symbol).strip())
        return symbols

    def _summarize_docs(self, docs: list[RetrievedChunk]) -> str:
        """汇总检索证据。"""

        if not docs:
            return "暂无可引用历史证据。"

        lines = []
        for index, doc in enumerate(docs, start=1):
            snippet = doc.text.replace("\n", " ").strip()
            lines.append(f"{index}. [{doc.source}/{doc.symbol}] score={doc.score:.3f} {snippet}")
        return "\n".join(lines)

    def _build_citations(self, docs: list[RetrievedChunk]) -> list[Citation]:
        """生成引用对象列表。"""

        citations: list[Citation] = []
        for doc in docs[:10]:
            snippet = doc.text.replace("\n", " ").strip()
            if len(snippet) > 120:
                snippet = snippet[:120] + "..."
            citations.append(
                Citation(
                    source=doc.source,
                    raw_ref=str(doc.metadata.get("raw_ref", doc.doc_id)),
                    snippet=snippet,
                    published_at=doc.published_at,
                )
            )
        return citations

    def _append_disclaimer(self, report: str) -> str:
        """附加合规免责声明。"""

        cleaned = report.strip() if report else ""
        if not cleaned:
            cleaned = "当前证据不足，建议补充数据后再执行结论判断。"
        return f"{cleaned}\n\n{self.settings.report_disclaimer}"

    def _format_signal_payload(self, value: Any, *, max_chars: int) -> str:
        """格式化信号 value，保留可读的原始内容。"""

        if isinstance(value, str):
            text = " ".join(value.split())
        elif isinstance(value, (dict, list, tuple)):
            try:
                text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
            except TypeError:
                text = str(value)
        else:
            text = str(value)

        if len(text) > max_chars:
            return text[:max_chars] + "...(truncated)"
        return text

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
