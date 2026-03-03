"""短期会话记忆存储抽象。"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class SessionMemoryStore(Protocol):
    """短期会话记忆读写接口。"""

    def append(self, *, conversation_id: str, item: dict[str, Any]) -> None:
        ...

    def get(self, *, conversation_id: str, limit: int = 50) -> list[dict[str, Any]]:
        ...

    def close(self) -> None:
        ...


class InMemorySessionMemoryStore:
    """进程内短期会话记忆存储。"""

    def __init__(self, max_items: int = 50) -> None:
        self.max_items = max_items
        self._data: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def append(self, *, conversation_id: str, item: dict[str, Any]) -> None:
        with self._lock:
            bucket = self._data.setdefault(conversation_id, [])
            bucket.append(item)
            if len(bucket) > self.max_items:
                del bucket[:-self.max_items]

    def get(self, *, conversation_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = list(self._data.get(conversation_id, []))
        if limit <= 0:
            return []
        return rows[-limit:]

    def close(self) -> None:
        return None


class RedisSessionMemoryStore:
    """Redis 短期会话记忆存储。"""

    def __init__(self, *, redis_url: str, ttl_seconds: int, max_items: int) -> None:
        try:
            import redis  # type: ignore
        except Exception as exc:  # pragma: no cover - 环境缺失时走回退逻辑
            raise RuntimeError("redis package is required for RedisSessionMemoryStore") from exc

        self._client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.ttl_seconds = max(ttl_seconds, 60)
        self.max_items = max(max_items, 1)

    def _key(self, conversation_id: str) -> str:
        return f"crypto_signal_agent:session:{conversation_id}"

    def append(self, *, conversation_id: str, item: dict[str, Any]) -> None:
        payload = json.dumps(item, ensure_ascii=False)
        key = self._key(conversation_id)
        pipe = self._client.pipeline()
        pipe.rpush(key, payload)
        pipe.ltrim(key, -self.max_items, -1)
        pipe.expire(key, self.ttl_seconds)
        pipe.execute()

    def get(self, *, conversation_id: str, limit: int = 50) -> list[dict[str, Any]]:
        key = self._key(conversation_id)
        size = max(limit, 1)
        rows = self._client.lrange(key, -size, -1)
        parsed: list[dict[str, Any]] = []
        for item in rows:
            try:
                payload = json.loads(item)
            except Exception:
                continue
            if isinstance(payload, dict):
                parsed.append(payload)
        return parsed

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            logger.exception("redis client close failed")


def build_session_memory_store(
    *,
    backend: str,
    redis_url: str,
    ttl_seconds: int,
    max_items: int,
) -> SessionMemoryStore:
    """构建短期会话记忆存储。"""

    normalized = backend.strip().lower()
    if normalized == "redis":
        try:
            return RedisSessionMemoryStore(
                redis_url=redis_url,
                ttl_seconds=ttl_seconds,
                max_items=max_items,
            )
        except Exception:
            logger.exception("Redis 会话存储初始化失败，回退内存模式")
    return InMemorySessionMemoryStore(max_items=max_items)

