"""研报生成代理。

负责将“用户问题 + 记忆画像 + 信号 + 检索证据”组织成结构化 Prompt，
再调用可替换 LLM 客户端生成最终报告草稿。
"""

from __future__ import annotations

from collections import Counter
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

    def generate(self, payload: ReportGenerationInput) -> ReportGenerationOutput:
        """生成结构化研报。"""

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(payload)

        response = self.llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        report_text = self._extract_text(getattr(response, "content", response))
        citations = self._build_citations(payload.retrieved_docs)
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

    def _build_user_prompt(self, payload: ReportGenerationInput) -> str:
        """将状态数据编排为用户提示词。"""

        signal_summary = self._summarize_signals(payload.signals)
        docs_summary = self._summarize_docs(payload.retrieved_docs)

        return (
            f"用户ID: {payload.user_id}\n"
            f"任务ID: {payload.task_id}\n"
            f"用户问题: {payload.query}\n\n"
            f"用户记忆画像:\n{payload.memory_profile}\n\n"
            f"信号摘要:\n{signal_summary}\n\n"
            f"证据摘要:\n{docs_summary}\n\n"
            "输出要求:\n"
            "1) 先给出结论和置信度；\n"
            "2) 每个结论关联至少一条证据；\n"
            "3) 明确风险与反例；\n"
            "4) 使用中文，风格偏专业研报。"
        )

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

        # 仅保留前 5 条示例，避免提示词过长。
        for signal in signals[:5]:
            top_lines.append(
                f"- {signal.symbol} | {signal.signal_type.value} | conf={signal.confidence:.2f} | src={signal.source}"
            )

        return "\n".join(top_lines)

    def _summarize_docs(self, docs: list[RetrievedChunk]) -> str:
        """汇总检索证据。"""

        if not docs:
            return "暂无可引用历史证据。"

        lines = []
        for index, doc in enumerate(docs[:8], start=1):
            snippet = doc.text.replace("\n", " ").strip()
            if len(snippet) > 180:
                snippet = snippet[:180] + "..."
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
