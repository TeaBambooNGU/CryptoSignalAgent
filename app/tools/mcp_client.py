"""MCP 数据采集客户端。

V1 约束：只采集可通过 MCP Tool 获取的数据。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config.settings import Settings
from app.models.schemas import RawSignal, SignalType

logger = logging.getLogger(__name__)


class MCPClient:
    """MCP 采集适配层。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError)),
        reraise=True,
    )
    def _fetch_endpoint(self, endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        """请求单个 MCP 端点并返回列表数据。"""

        with httpx.Client(timeout=15.0) as client:
            response = client.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
            if isinstance(data, dict):
                payload = data.get("data")
                if isinstance(payload, list):
                    return [item for item in payload if isinstance(item, dict)]
            return []

    def collect_signals(
        self,
        task_id: str,
        query: str,
        symbols: list[str],
    ) -> tuple[list[RawSignal], list[str]]:
        """通过 MCP 端点采集原始信号。

        返回：
        - raw_signals: 采集到的原始信号。
        - errors: 可降级错误列表，不中断后续节点。
        """

        errors: list[str] = []
        if not self.settings.mcp_tool_endpoints:
            errors.append("未配置 MCP_TOOL_ENDPOINTS，跳过实时采集")
            return [], errors

        params = {"query": query, "symbols": ",".join(symbols)}
        rows: list[dict[str, Any]] = []
        for endpoint in self.settings.mcp_tool_endpoints:
            try:
                rows.extend(self._fetch_endpoint(endpoint=endpoint, params=params))
            except Exception as exc:
                logger.exception("MCP 端点请求失败: %s", endpoint)
                errors.append(f"MCP 端点失败: {endpoint} ({type(exc).__name__})")

        return self._normalize_raw_rows(task_id=task_id, rows=rows), errors

    def _normalize_raw_rows(self, task_id: str, rows: list[dict[str, Any]]) -> list[RawSignal]:
        """将 MCP 响应映射为统一 RawSignal 结构。"""

        normalized: list[RawSignal] = []
        for item in rows:
            symbol = str(item.get("symbol", "UNKNOWN")).upper()
            source = str(item.get("source", "mcp"))
            raw_type = str(item.get("signal_type", "news")).lower()
            if raw_type not in {member.value for member in SignalType}:
                raw_type = SignalType.NEWS.value

            published_at = datetime.now(timezone.utc)
            raw_published = item.get("published_at")
            if isinstance(raw_published, str):
                try:
                    published_at = datetime.fromisoformat(raw_published.replace("Z", "+00:00"))
                except Exception:
                    published_at = datetime.now(timezone.utc)

            normalized.append(
                RawSignal(
                    symbol=symbol,
                    source=source,
                    signal_type=SignalType(raw_type),
                    value=item.get("value", item),
                    raw_ref=str(item.get("raw_ref", item.get("url", source))),
                    published_at=published_at,
                    metadata={"task_id": task_id, "raw": item},
                )
            )

        return normalized
