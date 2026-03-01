"""日志配置模块。

统一定义日志格式，确保服务、图节点与重试错误输出一致，方便排障与追踪。
"""

from __future__ import annotations

import logging


def setup_logging(level: str = "INFO") -> None:
    """初始化全局日志配置。

    参数:
    - level: 日志等级，支持 DEBUG/INFO/WARNING/ERROR。
    """

    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
