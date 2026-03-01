"""嵌入工具模块。

说明：
- 默认使用智谱 Embedding（`langchain_community.embeddings.ZhipuAIEmbeddings`）。
- 为保证离线联调可用，当依赖缺失或 `ZHIPUAI_API_KEY` 未配置时自动降级为哈希向量。
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from functools import lru_cache
from typing import Any, cast

from app.config.settings import Settings

logger = logging.getLogger(__name__)

try:
    from langchain_community.embeddings import ZhipuAIEmbeddings
except Exception:  # pragma: no cover - 未安装依赖时自动降级
    ZhipuAIEmbeddings = None  # type: ignore[misc,assignment]


class BatchedZhipuAIEmbeddings(  # type: ignore[misc,valid-type]
    ZhipuAIEmbeddings if ZhipuAIEmbeddings is not None else object
):
    """智谱 embedding 批处理封装。

    智谱接口单次最多 64 条输入，因此做分批保护。
    """

    batch_size: int = 64

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if ZhipuAIEmbeddings is None:
            raise RuntimeError("langchain_community 未安装，无法调用智谱 embedding")

        all_embeddings: list[list[float]] = []
        safe_batch_size = max(1, min(self.batch_size, 64))
        for index in range(0, len(texts), safe_batch_size):
            batch = texts[index : index + safe_batch_size]
            all_embeddings.extend(cast(list[list[float]], super().embed_documents(batch)))
        return all_embeddings


class _HashEmbeddingAdapter:
    """哈希向量降级实现。"""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_hash_text_to_embedding(text, self.dim) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return _hash_text_to_embedding(text, self.dim)


@lru_cache(maxsize=8)
def _build_backend(
    provider: str,
    model: str,
    batch_size: int,
    vector_dim: int,
    has_api_key: bool,
) -> Any:
    """构建并缓存 embedding 后端实例。"""

    provider_normalized = provider.strip().lower()
    if provider_normalized == "zhipu":
        if ZhipuAIEmbeddings is None:
            logger.warning("未安装 langchain_community，Embedding 自动降级为哈希向量")
            return _HashEmbeddingAdapter(dim=vector_dim)
        if not has_api_key:
            logger.warning("未配置 ZHIPUAI_API_KEY，Embedding 自动降级为哈希向量")
            return _HashEmbeddingAdapter(dim=vector_dim)
        return BatchedZhipuAIEmbeddings(model=model, batch_size=min(batch_size, 64))
    logger.warning("未知 EMBEDDING_PROVIDER=%s，Embedding 自动降级为哈希向量", provider)
    return _HashEmbeddingAdapter(dim=vector_dim)


def embed_texts(texts: list[str], settings: Settings) -> list[list[float]]:
    """批量向量化文本。"""

    if not texts:
        return []

    has_api_key = bool(os.getenv("ZHIPUAI_API_KEY"))
    backend = _build_backend(
        provider=settings.embedding_provider,
        model=settings.zhipu_embedding_model,
        batch_size=settings.zhipu_embedding_batch_size,
        vector_dim=settings.vector_dim,
        has_api_key=has_api_key,
    )
    vectors = cast(list[list[float]], backend.embed_documents(texts))
    _validate_vector_dim(vectors=vectors, expected_dim=settings.vector_dim)
    return vectors


def text_to_embedding(text: str, settings: Settings) -> list[float]:
    """单条文本向量化。"""

    vectors = embed_texts([text], settings=settings)
    if not vectors:
        return [0.0] * settings.vector_dim
    return vectors[0]


def _hash_text_to_embedding(text: str, dim: int) -> list[float]:
    """将文本映射为固定维度向量。

    该实现为确定性哈希向量，用于开发阶段和无密钥环境。
    """

    if dim <= 0:
        raise ValueError("dim 必须大于 0")

    values = [0.0] * dim
    words = text.split()
    if not words:
        return values

    for word in words:
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        for idx in range(dim):
            byte = digest[idx % len(digest)]
            values[idx] += (byte / 255.0) - 0.5

    norm = math.sqrt(sum(item * item for item in values))
    if norm == 0:
        return values
    return [item / norm for item in values]


def _validate_vector_dim(vectors: list[list[float]], expected_dim: int) -> None:
    """校验向量维度与 Milvus 配置一致。"""

    if not vectors:
        return
    actual_dim = len(vectors[0])
    if actual_dim != expected_dim:
        raise ValueError(
            "Embedding 向量维度不匹配："
            f"实际={actual_dim}, VECTOR_DIM={expected_dim}。"
            "请将 VECTOR_DIM 调整为智谱模型输出维度。"
        )


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """计算余弦相似度。"""

    if len(vec1) != len(vec2) or not vec1:
        return 0.0

    dot = sum(a * b for a, b in zip(vec1, vec2, strict=True))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)
