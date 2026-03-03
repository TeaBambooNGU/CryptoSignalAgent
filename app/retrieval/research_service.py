"""研究语料入库与检索服务。

职责：
1. 将信号/文档切块并向量化写入 Milvus。
2. 按查询执行召回与重排，输出可引用证据片段。
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any

from app.config.logging import get_logger, log_context
from app.config.settings import Settings
from app.models.schemas import IngestDocument, NormalizedSignal, RawSignal, RetrievedChunk
from app.retrieval.embedding import embed_texts, text_to_embedding
from app.retrieval.milvus_store import MilvusStore

logger = get_logger(__name__)

try:
    from llama_index.core.node_parser import SentenceSplitter
except Exception:  # pragma: no cover - import 失败时降级分块
    SentenceSplitter = None  # type: ignore


class ResearchService:
    """信号标准化、入库与检索服务。"""

    def __init__(self, settings: Settings, milvus_store: MilvusStore) -> None:
        self.settings = settings
        self.milvus_store = milvus_store
        self._splitter = (
            SentenceSplitter(chunk_size=700, chunk_overlap=80) if SentenceSplitter is not None else None
        )

    def normalize_signals(self, task_id: str, raw_signals: list[RawSignal]) -> list[NormalizedSignal]:
        """将 RawSignal 统一映射为标准信号模型。"""

        normalized: list[NormalizedSignal] = []
        for raw in raw_signals:
            confidence = self._estimate_signal_confidence(raw)
            normalized.append(
                NormalizedSignal(
                    timestamp=raw.published_at,
                    symbol=raw.symbol,
                    source=raw.source,
                    signal_type=raw.signal_type,
                    value=raw.value,
                    confidence=confidence,
                    raw_ref=raw.raw_ref,
                    ingest_mode="mcp",
                    task_id=task_id,
                )
            )

        with log_context(component="retrieval.normalize"):
            logger.info("标准化信号完成 raw=%s normalized=%s", len(raw_signals), len(normalized))
        return normalized

    def ingest_signals(self, signals: list[NormalizedSignal]) -> int:
        """将标准化信号转为研究语料 chunk 并写入 Milvus。"""

        rows: list[dict[str, Any]] = []
        texts: list[str] = []
        for signal in signals:
            text = self._signal_to_text(signal)
            chunks = self._split_text(text)
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{signal.task_id}-{signal.symbol}-{idx}"
                rows.append(
                    {
                        "id": self._make_row_id(signal.task_id, signal.symbol, chunk_id),
                        "doc_id": f"signal-{signal.task_id}-{signal.symbol}",
                        "chunk_id": chunk_id,
                        "symbol": signal.symbol,
                        "source": signal.source,
                        "published_at": int(signal.timestamp.timestamp()),
                        "text": chunk,
                        "metadata": {
                            "signal_type": signal.signal_type.value,
                            "confidence": signal.confidence,
                            "raw_ref": signal.raw_ref,
                            "ingest_mode": signal.ingest_mode,
                        },
                        "task_id": signal.task_id,
                    }
                )
                texts.append(chunk)

        if not rows:
            with log_context(component="retrieval.ingest"):
                logger.warning("信号入库跳过 rows=0")
            return 0

        embeddings = embed_texts(texts=texts, settings=self.settings)
        for row, embedding in zip(rows, embeddings, strict=True):
            row["embedding"] = embedding

        inserted = self.milvus_store.upsert_research_chunks(rows)
        with log_context(component="retrieval.ingest"):
            logger.info("信号入库完成 signals=%s chunks=%s inserted=%s", len(signals), len(rows), inserted)
        return inserted

    def ingest_documents(self, task_id: str, docs: list[IngestDocument]) -> int:
        """将外部文档入库到研究集合。"""

        rows: list[dict[str, Any]] = []
        texts: list[str] = []
        for doc in docs:
            for idx, chunk in enumerate(self._split_text(doc.text)):
                chunk_id = f"{doc.doc_id}-{idx}"
                rows.append(
                    {
                        "id": self._make_row_id(task_id, doc.doc_id, chunk_id),
                        "doc_id": doc.doc_id,
                        "chunk_id": chunk_id,
                        "symbol": doc.symbol.upper(),
                        "source": doc.source,
                        "published_at": int(doc.published_at.timestamp()),
                        "text": chunk,
                        "metadata": doc.metadata,
                        "task_id": task_id,
                    }
                )
                texts.append(chunk)

        if not rows:
            with log_context(component="retrieval.ingest"):
                logger.warning("文档入库跳过 docs=%s chunks=0", len(docs))
            return 0

        embeddings = embed_texts(texts=texts, settings=self.settings)
        for row, embedding in zip(rows, embeddings, strict=True):
            row["embedding"] = embedding

        inserted = self.milvus_store.upsert_research_chunks(rows)
        with log_context(component="retrieval.ingest"):
            logger.info("文档入库完成 docs=%s chunks=%s inserted=%s", len(docs), len(rows), inserted)
        return inserted

    def retrieve(self, query: str, symbols: list[str], top_k: int = 8) -> list[RetrievedChunk]:
        """执行召回与重排。"""

        with log_context(component="retrieval.search"):
            logger.info("检索开始 symbols=%s top_k=%s", symbols or ["ALL"], top_k)

        query_vector = text_to_embedding(query, self.settings)
        rows = self.milvus_store.search_research_chunks(
            query_vector=query_vector,
            symbols=symbols,
            top_k=max(top_k * 2, 10),
        )
        reranked = self._rerank_rows(rows=rows, now=datetime.now(timezone.utc), top_k=top_k)

        result: list[RetrievedChunk] = []
        for row in reranked:
            published_at = None
            ts = row.get("published_at")
            if isinstance(ts, (int, float)):
                published_at = datetime.fromtimestamp(ts, tz=timezone.utc)
            result.append(
                RetrievedChunk(
                    chunk_id=str(row.get("chunk_id", "")),
                    doc_id=str(row.get("doc_id", "")),
                    symbol=str(row.get("symbol", "UNKNOWN")),
                    source=str(row.get("source", "unknown")),
                    text=str(row.get("text", "")),
                    score=float(row.get("final_score", row.get("score", 0.0))),
                    published_at=published_at,
                    metadata=row.get("metadata", {}),
                )
            )

        with log_context(component="retrieval.search"):
            logger.info("检索完成 raw_hits=%s returned=%s", len(rows), len(result))
        return result

    def _signal_to_text(self, signal: NormalizedSignal) -> str:
        """将结构化信号转为文本，供切块与检索。"""

        value_repr = signal.value
        if not isinstance(value_repr, str):
            value_repr = json.dumps(value_repr, ensure_ascii=False)
        return (
            f"symbol={signal.symbol}\n"
            f"source={signal.source}\n"
            f"signal_type={signal.signal_type.value}\n"
            f"timestamp={signal.timestamp.isoformat()}\n"
            f"confidence={signal.confidence:.2f}\n"
            f"raw_ref={signal.raw_ref}\n"
            f"value={value_repr}"
        )

    def _split_text(self, text: str) -> list[str]:
        """文本切块。

        优先使用 LlamaIndex SentenceSplitter；若不可用则按长度降级切分。
        """

        if not text.strip():
            return []

        if self._splitter is not None:
            nodes = self._splitter.split_text(text)
            return [node.strip() for node in nodes if node.strip()]

        chunk_size = 700
        chunks: list[str] = []
        start = 0
        while start < len(text):
            chunks.append(text[start : start + chunk_size])
            start += chunk_size
        return chunks

    def _make_row_id(self, task_id: str, doc_id: str, chunk_id: str) -> str:
        """构造幂等行 ID。"""

        raw = f"{task_id}:{doc_id}:{chunk_id}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _estimate_signal_confidence(self, signal: RawSignal) -> float:
        """估算信号置信度。"""

        source_score = {
            "binance": 0.9,
            "coindesk": 0.8,
            "glassnode": 0.85,
            "x": 0.55,
            "reddit": 0.5,
        }
        canonical_source = self._canonical_source(
            source=signal.source,
            metadata=signal.metadata if isinstance(signal.metadata, dict) else {},
        )
        base = source_score.get(canonical_source, 0.65)
        if signal.signal_type.value in {"price", "onchain"}:
            base += 0.08
        return max(0.1, min(0.99, base))

    def _rerank_rows(
        self,
        rows: list[dict[str, Any]],
        now: datetime,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """按时间衰减 + 来源可信度 + 语义分数进行重排。"""

        source_weight = {
            "binance": 1.0,
            "coindesk": 0.9,
            "glassnode": 0.92,
            "cointelegraph": 0.82,
            "x": 0.65,
            "reddit": 0.6,
        }

        rescored: list[dict[str, Any]] = []
        for row in rows:
            row_copy = row.copy()
            semantic = float(row.get("score", 0.0))

            ts = row.get("published_at")
            if isinstance(ts, (int, float)):
                age_hours = max(0.0, (now.timestamp() - float(ts)) / 3600.0)
            else:
                age_hours = 72.0
            time_decay = math.exp(-age_hours / 72.0)

            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            source = self._canonical_source(source=str(row.get("source", "unknown")), metadata=metadata)
            credibility = source_weight.get(source, 0.75)

            final_score = semantic * 0.55 + time_decay * 0.25 + credibility * 0.20
            row_copy["final_score"] = final_score
            rescored.append(row_copy)

        rescored.sort(key=lambda item: item.get("final_score", 0.0), reverse=True)
        return rescored[:top_k]

    def _canonical_source(self, source: str, metadata: dict[str, Any] | None = None) -> str:
        alias_map = {
            "binance": "binance",
            "coindesk": "coindesk",
            "glassnode": "glassnode",
            "cointelegraph": "cointelegraph",
            "reddit": "reddit",
            "x": "x",
            "twitter": "x",
            "x.com": "x",
        }
        candidates = [source.strip().lower()]
        if ":" in candidates[0]:
            candidates.append(candidates[0].split(":", 1)[1])

        if isinstance(metadata, dict):
            for key in ("provider", "source", "tool"):
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    candidates.append(value.strip().lower())

        for candidate in candidates:
            if not candidate:
                continue
            wrapped = f" {candidate} "
            for alias, canonical in alias_map.items():
                if alias in candidate or alias in wrapped:
                    return canonical

        for candidate in candidates:
            if candidate:
                return candidate
        return "unknown"
