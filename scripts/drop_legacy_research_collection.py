"""删除历史遗留的 `research_chunks` collection。

用途：
- 在完成双向量库切换后，清理旧统一库数据。
- 可重复执行；若集合不存在则直接退出。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import settings
from app.retrieval.milvus_store import MilvusStore


def main() -> None:
    store = MilvusStore(settings)
    store.connect()
    dropped = store.drop_legacy_research_collection()
    if dropped:
        print("已删除历史遗留 collection: research_chunks")
    else:
        print("未发现历史遗留 collection: research_chunks")
    store.close()


if __name__ == "__main__":
    main()
