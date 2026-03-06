"""Milvus 存储层封装。

职责：
1. 初始化并维护 `signal_chunks`、`knowledge_chunks` 与 `user_memory` 三个集合。
2. 对外暴露实时信号、知识库文档、用户记忆的写入/检索接口。
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
        self._signal_collection: Collection | None = None
        self._knowledge_collection: Collection | None = None
        self._memory_collection: Collection | None = None
        self._connected = False

        # 降级存储：仅在允许 fallback 且 Milvus 不可用时使用。
        self._signal_fallback: list[dict[str, Any]] = []
        self._knowledge_fallback: list[dict[str, Any]] = []
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
        """确保信号集合、知识集合与记忆集合存在并可检索。"""

        self._signal_collection = self._prepare_signal_collection()
        self._knowledge_collection = self._prepare_knowledge_collection()
        self._memory_collection = self._prepare_memory_collection()

    def _prepare_chunk_collection(self, *, name: str, description: str) -> Collection:
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
            schema = CollectionSchema(fields=fields, description=description, enable_dynamic_field=False)
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

    def _prepare_signal_collection(self) -> Collection:
        return self._prepare_chunk_collection(
            name=self.settings.milvus_signal_collection,
            description="实时信号分块",
        )

    def _prepare_knowledge_collection(self) -> Collection:
        return self._prepare_chunk_collection(
            name=self.settings.milvus_knowledge_collection,
            description="知识库文档分块",
        )

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
    def _upsert_chunk_rows(
        self,
        *,
        rows: list[dict[str, Any]],
        collection: Collection | None,
        fallback_store: list[dict[str, Any]],
        collection_name: str,
    ) -> int:
        """写入 chunk 行。

        幂等策略：
        - 调用方需提供稳定 `id`（通常由 doc_id + chunk_id 组成）。
        - 重复写入相同 id 时由上层先删后插或跳过，本层保持简单插入语义。
        """

        if not rows:
            return 0

        if self.using_fallback:
            fallback_store.extend(rows)
            return len(rows)

        if collection is None:
            raise RuntimeError(f"{collection_name} collection 未初始化")

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
        collection.delete(delete_expr)
        collection.insert(normalized_rows)
        collection.flush()
        return len(normalized_rows)

    def upsert_signal_chunks(self, rows: list[dict[str, Any]]) -> int:
        """写入实时信号分块。"""

        return self._upsert_chunk_rows(
            rows=rows,
            collection=self._signal_collection,
            fallback_store=self._signal_fallback,
            collection_name="signal",
        )

    def upsert_knowledge_chunks(self, rows: list[dict[str, Any]]) -> int:
        """写入知识库文档分块。"""

        return self._upsert_chunk_rows(
            rows=rows,
            collection=self._knowledge_collection,
            fallback_store=self._knowledge_fallback,
            collection_name="knowledge",
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _search_chunk_rows(
        self,
        *,
        query_vector: list[float],
        top_k: int = 8,
        symbols: list[str] | None = None,
        collection: Collection | None,
        fallback_store: list[dict[str, Any]],
        collection_name: str,
    ) -> list[dict[str, Any]]:
        """检索 chunk 行。"""

        if not query_vector:
            return []

        if self.using_fallback:
            candidates = fallback_store
            if symbols:
                symbols_set = set(symbols)
                candidates = [item for item in candidates if item.get("symbol") in symbols_set]
            ranked = sorted(
                candidates,
                key=lambda item: cosine_similarity(item.get("embedding", []), query_vector),
                reverse=True,
            )
            return ranked[:top_k]

        if collection is None:
            raise RuntimeError(f"{collection_name} collection 未初始化")

        expr = ""
        if symbols:
            symbol_values = ",".join(json.dumps(symbol) for symbol in symbols)
            expr = f"symbol in [{symbol_values}]"

        result = collection.search(
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

    def search_signal_chunks(
        self,
        query_vector: list[float],
        top_k: int = 8,
        symbols: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """检索实时信号分块。"""

        return self._search_chunk_rows(
            query_vector=query_vector,
            top_k=top_k,
            symbols=symbols,
            collection=self._signal_collection,
            fallback_store=self._signal_fallback,
            collection_name="signal",
        )

    def search_knowledge_chunks(
        self,
        query_vector: list[float],
        top_k: int = 8,
        symbols: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """检索知识库文档分块。"""

        return self._search_chunk_rows(
            query_vector=query_vector,
            top_k=top_k,
            symbols=symbols,
            collection=self._knowledge_collection,
            fallback_store=self._knowledge_fallback,
            collection_name="knowledge",
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def delete_knowledge_chunks_by_doc_id(self, doc_id: str) -> int:
        """按文档 ID 删除知识库 chunk。"""

        normalized_doc_id = str(doc_id).strip()
        if not normalized_doc_id:
            return 0

        if self.using_fallback:
            original_size = len(self._knowledge_fallback)
            self._knowledge_fallback = [
                row for row in self._knowledge_fallback if str(row.get("doc_id", "")) != normalized_doc_id
            ]
            return original_size - len(self._knowledge_fallback)

        if self._knowledge_collection is None:
            raise RuntimeError("knowledge collection 未初始化")

        delete_expr = f"doc_id == {json.dumps(normalized_doc_id)}"
        self._knowledge_collection.delete(delete_expr)
        self._knowledge_collection.flush()
        return 1

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def drop_legacy_research_collection(self, name: str = "research_chunks") -> bool:
        """删除历史遗留的统一 research collection。"""

        if self.using_fallback:
            return False
        if not utility.has_collection(name, using=self.alias):
            return False
        utility.drop_collection(name, using=self.alias)
        logger.info("已删除历史遗留 collection: %s", name)
        return True

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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def delete_user_memory_by_ids(self, ids: list[str]) -> int:
        """按主键批量删除用户记忆。"""

        normalized_ids = [str(item).strip() for item in ids if str(item).strip()]
        if not normalized_ids:
            return 0

        if self.using_fallback:
            original_size = len(self._memory_fallback)
            delete_set = set(normalized_ids)
            self._memory_fallback = [row for row in self._memory_fallback if str(row.get("id", "")) not in delete_set]
            return original_size - len(self._memory_fallback)

        if self._memory_collection is None:
            raise RuntimeError("memory collection 未初始化")

        expr_values = ",".join(json.dumps(item) for item in normalized_ids)
        delete_expr = f"id in [{expr_values}]"
        self._memory_collection.delete(delete_expr)
        self._memory_collection.flush()
        return len(normalized_ids)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def list_all_user_memory(
        self,
        *,
        limit: int = 50000,
        batch_size: int = 2000,
        include_embedding: bool = False,
    ) -> list[dict[str, Any]]:
        """扫描用户记忆集合（用于离线修复脚本）。"""

        final_limit = max(int(limit), 0)
        if final_limit <= 0:
            return []

        if self.using_fallback:
            rows = [dict(row) for row in self._memory_fallback]
            rows.sort(key=lambda item: item.get("updated_at", 0), reverse=True)
            return rows[:final_limit]

        if self._memory_collection is None:
            raise RuntimeError("memory collection 未初始化")

        output_fields = ["id", "user_id", "memory_type", "content", "confidence", "updated_at"]
        if include_embedding:
            output_fields.append("embedding")

        rows: list[dict[str, Any]] = []
        iterator_factory = getattr(self._memory_collection, "query_iterator", None)
        if callable(iterator_factory):
            iterator = None
            try:
                iterator = iterator_factory(
                    batch_size=max(1, int(batch_size)),
                    expr="id != ''",
                    output_fields=output_fields,
                )
                while len(rows) < final_limit:
                    batch = iterator.next()
                    if not batch:
                        break
                    rows.extend(dict(item) for item in batch)
            except Exception:
                logger.exception("query_iterator 扫描 user_memory 失败，回退单次查询")
                rows = []
            finally:
                close_fn = getattr(iterator, "close", None) if iterator is not None else None
                if callable(close_fn):
                    close_fn()

        if not rows:
            rows = self._memory_collection.query(
                expr="id != ''",
                output_fields=output_fields,
                limit=final_limit,
            )

        rows.sort(key=lambda item: item.get("updated_at", 0), reverse=True)
        return rows[:final_limit]

    def close(self) -> None:
        """关闭连接。"""

        if self._connected:
            connections.disconnect(self.alias)
            self._connected = False
