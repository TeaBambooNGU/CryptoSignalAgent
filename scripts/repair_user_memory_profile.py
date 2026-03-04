"""修复 Milvus 用户长期偏好为“单条画像”结构。

规则：
- watchlist: 并集去重（保序，后写入新增追加）。
- risk_preference: 最新有效值覆盖。
- reading_habit: 最新有效值覆盖。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config.settings import Settings
from app.memory.mem0_service import MemoryService
from app.models.schemas import MemoryType
from app.retrieval.milvus_store import MilvusStore


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="修复 user_memory 中 preference 历史冗余数据")
    parser.add_argument("--limit", type=int, default=50000, help="扫描上限（默认 50000）")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不写入")
    return parser.parse_args()


def _coerce_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _coerce_embedding(embedding: Any, *, dim: int) -> list[float]:
    if isinstance(embedding, list) and len(embedding) == dim:
        return [float(item) for item in embedding]
    return [0.0] * dim


def _merge_preference_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    ordered = sorted(rows, key=lambda item: int(item.get("updated_at", 0) or 0))
    for row in ordered:
        parsed = _coerce_dict(row.get("content"))
        normalized = MemoryService._normalize_preference_payload(parsed)
        if not normalized:
            continue
        merged = MemoryService._merge_preference_payload(merged, normalized)
    return merged


def main() -> int:
    args = _parse_args()
    settings = Settings.from_env()
    store = MilvusStore(settings)
    store.connect()
    try:
        if store.using_fallback:
            print("Milvus 未连接成功，当前是 fallback 模式，终止修复。")
            return 1

        rows = store.list_all_user_memory(limit=max(args.limit, 1), include_embedding=True)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            user_id = str(row.get("user_id", "")).strip()
            if not user_id:
                continue
            grouped[user_id].append(row)

        users_total = 0
        users_changed = 0
        upsert_count = 0
        delete_count = 0

        for user_id, user_rows in grouped.items():
            preference_rows = [
                row for row in user_rows if str(row.get("memory_type", "")) == MemoryType.PREFERENCE.value
            ]
            if not preference_rows:
                continue

            users_total += 1
            merged_profile = _merge_preference_rows(preference_rows)
            if not merged_profile:
                continue

            canonical_id = MemoryService.build_preference_profile_id(user_id)
            latest_row = max(preference_rows, key=lambda item: int(item.get("updated_at", 0) or 0))
            canonical_content = json.dumps(merged_profile, ensure_ascii=False, sort_keys=True)
            canonical_confidence = max(float(row.get("confidence", 0.0) or 0.0) for row in preference_rows)
            canonical_updated_at = int(latest_row.get("updated_at", 0) or 0)
            canonical_embedding = _coerce_embedding(latest_row.get("embedding"), dim=settings.vector_dim)
            canonical_record = {
                "id": canonical_id,
                "user_id": user_id,
                "memory_type": MemoryType.PREFERENCE.value,
                "content": canonical_content,
                "confidence": canonical_confidence,
                "updated_at": canonical_updated_at,
                "embedding": canonical_embedding,
            }

            stale_ids = [
                str(row.get("id", "")).strip()
                for row in preference_rows
                if str(row.get("id", "")).strip() and str(row.get("id", "")).strip() != canonical_id
            ]
            existing_canonical = next(
                (row for row in preference_rows if str(row.get("id", "")).strip() == canonical_id),
                None,
            )
            existing_content = str(existing_canonical.get("content", "")) if existing_canonical else ""
            needs_upsert = existing_content != canonical_content

            if not stale_ids and not needs_upsert:
                continue

            users_changed += 1
            if args.dry_run:
                upsert_count += 1 if needs_upsert else 0
                delete_count += len(stale_ids)
                continue

            if needs_upsert:
                store.upsert_user_memory([canonical_record])
                upsert_count += 1
            if stale_ids:
                delete_count += store.delete_user_memory_by_ids(stale_ids)

        print(
            "修复完成 "
            f"users_total={users_total} users_changed={users_changed} "
            f"upsert={upsert_count} deleted={delete_count} dry_run={args.dry_run}"
        )
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
