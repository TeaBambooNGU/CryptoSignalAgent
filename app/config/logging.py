"""日志配置与链路上下文工具。

统一定义日志格式，并通过 contextvars 传播 trace/task/user 维度，
方便跨模块追踪与排障。
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import logging.config
from collections.abc import Iterator

TRACE_ID_KEY = "trace_id"
TASK_ID_KEY = "task_id"
USER_ID_KEY = "user_id"
COMPONENT_KEY = "component"
_EMPTY_VALUE = "-"

_trace_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(TRACE_ID_KEY, default=_EMPTY_VALUE)
_task_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(TASK_ID_KEY, default=_EMPTY_VALUE)
_user_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(USER_ID_KEY, default=_EMPTY_VALUE)
_component_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(COMPONENT_KEY, default=_EMPTY_VALUE)

_LOGGING_INITIALIZED = False


class _ContextFilter(logging.Filter):
    """将当前上下文注入到每条日志记录。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _trace_id_ctx.get()  # type: ignore[attr-defined]
        record.task_id = _task_id_ctx.get()  # type: ignore[attr-defined]
        record.user_id = _user_id_ctx.get()  # type: ignore[attr-defined]

        component = _component_ctx.get()
        if component == _EMPTY_VALUE:
            component = record.name
        record.component = component  # type: ignore[attr-defined]
        return True


def setup_logging(level: str = "INFO") -> None:
    """初始化全局日志配置。"""

    global _LOGGING_INITIALIZED

    log_level = level.upper()
    if _LOGGING_INITIALIZED:
        logging.getLogger().setLevel(log_level)
        return

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {
                "context": {"()": _ContextFilter},
            },
            "formatters": {
                "standard": {
                    "format": (
                        "%(asctime)s | %(levelname)s | %(component)s | "
                        "trace_id=%(trace_id)s task_id=%(task_id)s user_id=%(user_id)s | %(message)s"
                    )
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": log_level,
                    "formatter": "standard",
                    "filters": ["context"],
                }
            },
            "root": {
                "level": log_level,
                "handlers": ["console"],
            },
        }
    )
    _LOGGING_INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """获取模块 logger，供业务模块统一导入。"""

    return logging.getLogger(name)


def get_current_trace_id() -> str:
    """获取当前上下文 trace_id。"""

    return _trace_id_ctx.get()


def set_log_context(
    *,
    trace_id: str | None = None,
    task_id: str | None = None,
    user_id: str | None = None,
    component: str | None = None,
) -> dict[str, contextvars.Token[str]]:
    """设置日志上下文并返回可回滚 token。"""

    tokens: dict[str, contextvars.Token[str]] = {}
    if trace_id is not None:
        tokens[TRACE_ID_KEY] = _trace_id_ctx.set(trace_id)
    if task_id is not None:
        tokens[TASK_ID_KEY] = _task_id_ctx.set(task_id)
    if user_id is not None:
        tokens[USER_ID_KEY] = _user_id_ctx.set(user_id)
    if component is not None:
        tokens[COMPONENT_KEY] = _component_ctx.set(component)
    return tokens


def reset_log_context(tokens: dict[str, contextvars.Token[str]]) -> None:
    """回滚 set_log_context 产生的 token。"""

    for key, token in tokens.items():
        if key == TRACE_ID_KEY:
            _trace_id_ctx.reset(token)
        elif key == TASK_ID_KEY:
            _task_id_ctx.reset(token)
        elif key == USER_ID_KEY:
            _user_id_ctx.reset(token)
        elif key == COMPONENT_KEY:
            _component_ctx.reset(token)


@contextlib.contextmanager
def log_context(
    *,
    trace_id: str | None = None,
    task_id: str | None = None,
    user_id: str | None = None,
    component: str | None = None,
) -> Iterator[None]:
    """上下文管理器版本的日志上下文设置。"""

    tokens = set_log_context(trace_id=trace_id, task_id=task_id, user_id=user_id, component=component)
    try:
        yield
    finally:
        reset_log_context(tokens)


def clear_log_context() -> None:
    """主动清空上下文字段。"""

    _trace_id_ctx.set(_EMPTY_VALUE)
    _task_id_ctx.set(_EMPTY_VALUE)
    _user_id_ctx.set(_EMPTY_VALUE)
    _component_ctx.set(_EMPTY_VALUE)
