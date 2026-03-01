"""Milvus 初始化脚本。

用途：
- 本地启动后一次性创建所需集合与索引。
- 可重复执行，已存在时自动跳过创建。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许使用 `uv run python scripts/init_milvus.py` 直接运行。
# 直接运行脚本时，Python 默认只把 `scripts/` 加入 sys.path，
# 因此这里显式补充项目根目录，确保可导入 `app` 包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import settings
from app.retrieval.milvus_store import MilvusStore


def main() -> None:
    store = MilvusStore(settings)
    store.connect()
    mode = "fallback" if store.using_fallback else "milvus"
    print(f"Milvus 初始化完成，当前模式: {mode}")
    store.close()


if __name__ == "__main__":
    main()
