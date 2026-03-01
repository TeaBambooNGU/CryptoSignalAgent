"""Milvus 存储层封装。

职责：
1. 初始化并维护 `research_chunks` 与 `user_memory` 两个集合。
2. 对外暴露统一的写入/检索接口。
3. 在 Milvus 不可用时按配置降级到内存存储，保证主流程可用。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config.settings import Settings
from app.retrieval.embedding import cosine_similarity

logger = logging.getLogger(__name__)


class MilvusStore:
    """Milvus 数据访问层。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.alias = "crypto_signal_agent"
        self._research_collection: Collection | None = None
        self._memory_collection: Collection | None = None
        self._connected = False

        # 降级存储：仅在允许 fallback 且 Milvus 不可用时使用。
        self._research_fallback: list[dict[str, Any]] = []
        self._memory_fallback: list[dict[str, Any]] = []

    @property
    def using_fallback(self) -> bool:
        """是否处于内存降级模式。"""

        return not self._connected

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def connect(self) -> None:
        """连接 Milvus 并准备集合。

        重试策略说明：
        - 触发条件：连接超时、网络抖动、临时服务不可达。
        - 停止条件：最多重试 3 次。
        """

        if not self.settings.milvus_enabled:
            logger.warning("MILVUS_ENABLED=false，直接使用内存降级存储")
            return

        try:
            connections.connect(
                alias=self.alias,
                uri=self.settings.milvus_uri,
                token=self.settings.milvus_token or None,
                db_name=self.settings.milvus_db_name,
            )
            self._connected = True
            self._ensure_collections()
            logger.info("Milvus 连接成功: %s", self.settings.milvus_uri)
        except Exception:
            self._connected = False
            logger.exception("Milvus 连接失败")
            if not self.settings.milvus_allow_fallback:
                raise
            logger.warning("启用内存降级模式，数据不会持久化")

    def _ensure_collections(self) -> None:
        """确保研究集合与记忆集合存在并可检索。"""

        self._research_collection = self._prepare_research_collection()
        self._memory_collection = self._prepare_memory_collection()

    def _prepare_research_collection(self) -> Collection:
        name = self.settings.milvus_research_collection
        if not utility.has_collection(name, using=self.alias):
            fields = [
                FieldSchema("id", DataType.VARCHAR, is_primary=True, max_length=128),
                FieldSchema("doc_id", DataType.VARCHAR, max_length=128),
                FieldSchema("chunk_id", DataType.VARCHAR, max_length=128),
                FieldSchema("symbol", DataType.VARCHAR, max_length=32),
                FieldSchema("source", DataType.VARCHAR, max_length=128),
                FieldSchema("published_at", DataType.INT64),
                FieldSchema("text", DataType.VARCHAR, max_length=8192),
                FieldSchema("metadata", DataType.JSON),
                FieldSchema("task_id", DataType.VARCHAR, max_length=128),
                FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=self.settings.vector_dim),
            ]
            schema = CollectionSchema(fields=fields, description="研究语料分块", enable_dynamic_field=False)
            collection = Collection(name=name, schema=schema, using=self.alias)
            collection.create_index(
                field_name="embedding",
                index_params={
                    "index_type": "HNSW",
                    "metric_type": "COSINE",
                    "params": {"M": 8, "efConstruction": 64},
                },
            )
            logger.info("创建 Milvus 集合: %s", name)
        else:
            collection = Collection(name=name, using=self.alias)

        collection.load()
        return collection

    def _prepare_memory_collection(self) -> Collection:
        name = self.settings.milvus_memory_collection
        if not utility.has_collection(name, using=self.alias):
            fields = [
                FieldSchema("id", DataType.VARCHAR, is_primary=True, max_length=128),
                FieldSchema("user_id", DataType.VARCHAR, max_length=128),
                FieldSchema("memory_type", DataType.VARCHAR, max_length=64),
                FieldSchema("content", DataType.VARCHAR, max_length=4096),
                FieldSchema("confidence", DataType.FLOAT),
                FieldSchema("updated_at", DataType.INT64),
                FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=self.settings.vector_dim),
            ]
            schema = CollectionSchema(fields=fields, description="用户记忆", enable_dynamic_field=False)
            collection = Collection(name=name, schema=schema, using=self.alias)
            collection.create_index(
                field_name="embedding",
                index_params={
                    "index_type": "HNSW",
                    "metric_type": "COSINE",
                    "params": {"M": 8, "efConstruction": 64},
                },
            )
            logger.info("创建 Milvus 集合: %s", name)
        else:
            collection = Collection(name=name, using=self.alias)

        collection.load()
        return collection

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def upsert_research_chunks(self, rows: list[dict[str, Any]]) -> int:
        """写入研究语料分块。

        幂等策略：
        - 调用方需提供稳定 `id`（通常由 doc_id + chunk_id 组成）。
        - 重复写入相同 id 时由上层先删后插或跳过，本层保持简单插入语义。
        """

        if not rows:
            return 0

        if self.using_fallback:
            self._research_fallback.extend(rows)
            return len(rows)

        if self._research_collection is None:
            raise RuntimeError("research collection 未初始化")

        normalized_rows = []
        for row in rows:
            normalized_rows.append(
                {
                    "id": str(row["id"]),
                    "doc_id": str(row["doc_id"]),
                    "chunk_id": str(row["chunk_id"]),
                    "symbol": str(row["symbol"]),
                    "source": str(row["source"]),
                    "published_at": int(row["published_at"]),
                    "text": str(row["text"])[:8192],
                    "metadata": row.get("metadata", {}),
                    "task_id": str(row["task_id"]),
                    "embedding": row["embedding"],
                }
            )

        ids = [item["id"] for item in normalized_rows]
        expr_values = ",".join(json.dumps(item) for item in ids)
        delete_expr = f"id in [{expr_values}]"
        self._research_collection.delete(delete_expr)
        self._research_collection.insert(normalized_rows)
        self._research_collection.flush()
        return len(normalized_rows)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def search_research_chunks(
        self,
        query_vector: list[float],
        top_k: int = 8,
        symbols: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """检索研究语料分块。"""

        if not query_vector:
            return []

        if self.using_fallback:
            candidates = self._research_fallback
            if symbols:
                symbols_set = set(symbols)
                candidates = [item for item in candidates if item.get("symbol") in symbols_set]
            ranked = sorted(
                candidates,
                key=lambda item: cosine_similarity(item.get("embedding", []), query_vector),
                reverse=True,
            )
            return ranked[:top_k]

        if self._research_collection is None:
            raise RuntimeError("research collection 未初始化")

        expr = ""
        if symbols:
            symbol_values = ",".join(json.dumps(symbol) for symbol in symbols)
            expr = f"symbol in [{symbol_values}]"

        result = self._research_collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            expr=expr or None,
            output_fields=["id", "doc_id", "chunk_id", "symbol", "source", "published_at", "text", "metadata", "task_id"],
        )

        records: list[dict[str, Any]] = []
        for hit in result[0]:
            entity = hit.entity
            records.append(
                {
                    "id": entity.get("id"),
                    "doc_id": entity.get("doc_id"),
                    "chunk_id": entity.get("chunk_id"),
                    "symbol": entity.get("symbol"),
                    "source": entity.get("source"),
                    "published_at": entity.get("published_at"),
                    "text": entity.get("text"),
                    "metadata": entity.get("metadata") or {},
                    "task_id": entity.get("task_id"),
                    "score": float(hit.distance),
                }
            )
        return records

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def upsert_user_memory(self, rows: list[dict[str, Any]]) -> int:
        """写入用户记忆。"""

        if not rows:
            return 0

        if self.using_fallback:
            for row in rows:
                row_copy = row.copy()
                row_copy["updated_at"] = int(row_copy.get("updated_at", datetime.now(timezone.utc).timestamp()))
                self._memory_fallback.append(row_copy)
            return len(rows)

        if self._memory_collection is None:
            raise RuntimeError("memory collection 未初始化")

        normalized_rows = []
        for row in rows:
            normalized_rows.append(
                {
                    "id": str(row["id"]),
                    "user_id": str(row["user_id"]),
                    "memory_type": str(row["memory_type"]),
                    "content": str(row["content"])[:4096],
                    "confidence": float(row.get("confidence", 0.5)),
                    "updated_at": int(row.get("updated_at", datetime.now(timezone.utc).timestamp())),
                    "embedding": row["embedding"],
                }
            )

        ids = [item["id"] for item in normalized_rows]
        expr_values = ",".join(json.dumps(item) for item in ids)
        delete_expr = f"id in [{expr_values}]"
        self._memory_collection.delete(delete_expr)
        self._memory_collection.insert(normalized_rows)
        self._memory_collection.flush()
        return len(normalized_rows)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def query_user_memory(self, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """按用户查询记忆列表。"""

        if self.using_fallback:
            rows = [item for item in self._memory_fallback if item.get("user_id") == user_id]
            rows.sort(key=lambda item: item.get("updated_at", 0), reverse=True)
            return rows[:limit]

        if self._memory_collection is None:
            raise RuntimeError("memory collection 未初始化")

        user_literal = json.dumps(user_id)
        expr = f"user_id == {user_literal}"
        rows = self._memory_collection.query(
            expr=expr,
            output_fields=["id", "user_id", "memory_type", "content", "confidence", "updated_at"],
            limit=limit,
        )
        rows.sort(key=lambda item: item.get("updated_at", 0), reverse=True)
        return rows

    def close(self) -> None:
        """关闭连接。"""

        if self._connected:
            connections.disconnect(self.alias)
            self._connected = False
