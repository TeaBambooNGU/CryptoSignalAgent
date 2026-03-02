"""日志配置与链路上下文工具。

统一定义日志格式，并通过 contextvars 传播 trace/task/user 维度，
方便跨模块追踪与排障。
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import logging.config
import os
import time
from collections.abc import Iterator
from datetime import date, datetime, timezone, timedelta
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

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


class SizeAndTimeRotatingFileHandler(TimedRotatingFileHandler):
    """按天轮转，并在单文件超大小时继续切分。"""

    def __init__(
        self,
        filename: str,
        *,
        max_bytes: int,
        backup_days: int,
        encoding: str = "utf-8",
        utc: bool = False,
        delay: bool = False,
    ) -> None:
        super().__init__(
            filename=filename,
            when="midnight",
            interval=1,
            backupCount=0,
            encoding=encoding,
            delay=delay,
            utc=utc,
        )
        self.max_bytes = max(1, int(max_bytes))
        self.backup_days = max(1, int(backup_days))
        self.suffix = "%Y-%m-%d"
        self._rollover_reason: str | None = None

    def shouldRollover(self, record: logging.LogRecord) -> int:  # noqa: N802
        if super().shouldRollover(record):
            self._rollover_reason = "time"
            return 1

        if self.stream is None:
            self.stream = self._open()

        message = f"{self.format(record)}\n"
        message_size = len(message.encode(self.encoding or "utf-8", errors="replace"))

        if self.stream.tell() + message_size >= self.max_bytes:
            self._rollover_reason = "size"
            return 1
        return 0

    def doRollover(self) -> None:  # noqa: N802
        if self.stream:
            self.stream.close()
            self.stream = None

        now_ts = int(time.time())
        suffix_ts = now_ts if self._rollover_reason == "size" else int(self.rolloverAt - self.interval)
        time_tuple = time.gmtime(suffix_ts) if self.utc else time.localtime(suffix_ts)
        date_suffix = time.strftime(self.suffix, time_tuple)

        dated_filename = f"{self.baseFilename}.{date_suffix}"
        target_filename = self._next_indexed_filename(dated_filename)

        if os.path.exists(self.baseFilename):
            os.replace(self.baseFilename, target_filename)

        if not self.delay:
            self.stream = self._open()

        if now_ts >= self.rolloverAt:
            next_rollover_at = self.computeRollover(now_ts)
            while next_rollover_at <= now_ts:
                next_rollover_at += self.interval
            self.rolloverAt = next_rollover_at

        self._cleanup_expired_files()
        self._rollover_reason = None

    def _next_indexed_filename(self, dated_filename: str) -> str:
        if not os.path.exists(dated_filename):
            return dated_filename

        index = 1
        while True:
            candidate = f"{dated_filename}.{index}"
            if not os.path.exists(candidate):
                return candidate
            index += 1

    def _cleanup_expired_files(self) -> None:
        base_path = Path(self.baseFilename)
        base_name = base_path.name
        prefix = f"{base_name}."

        today = datetime.now(timezone.utc).date() if self.utc else date.today()
        cutoff = today - timedelta(days=self.backup_days - 1)

        for path in base_path.parent.glob(f"{base_name}.*"):
            suffix = path.name[len(prefix) :]
            day_token = suffix.split(".", 1)[0]
            try:
                day_value = datetime.strptime(day_token, "%Y-%m-%d").date()
            except ValueError:
                continue

            if day_value < cutoff:
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue


def setup_logging(
    level: str = "INFO",
    *,
    log_to_file: bool = False,
    log_file_path: str = "logs/app.log",
    log_file_max_mb: int = 10,
    log_file_backup_days: int = 5,
) -> None:
    """初始化全局日志配置。"""

    global _LOGGING_INITIALIZED

    log_level = level.upper()
    if _LOGGING_INITIALIZED:
        logging.getLogger().setLevel(log_level)
        return

    handlers: dict[str, dict[str, object]] = {
        "console": {
            "class": "logging.StreamHandler",
            "level": log_level,
            "formatter": "standard",
            "filters": ["context"],
        }
    }
    root_handlers = ["console"]

    file_config_error = ""
    resolved_file_path = ""
    if log_to_file:
        try:
            raw_path = log_file_path.strip()
            if not raw_path:
                raise ValueError("LOG_FILE_PATH 不能为空")
            log_path = Path(raw_path).expanduser()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            resolved_file_path = str(log_path)
            max_bytes = max(1, int(log_file_max_mb)) * 1024 * 1024
            handlers["file"] = {
                "()": SizeAndTimeRotatingFileHandler,
                "level": log_level,
                "formatter": "standard",
                "filters": ["context"],
                "filename": resolved_file_path,
                "max_bytes": max_bytes,
                "backup_days": max(1, int(log_file_backup_days)),
                "encoding": "utf-8",
            }
            root_handlers.append("file")
        except Exception as exc:  # pragma: no cover
            file_config_error = f"{type(exc).__name__}: {exc}"

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
            "handlers": handlers,
            "root": {
                "level": log_level,
                "handlers": root_handlers,
            },
        }
    )
    _LOGGING_INITIALIZED = True

    logger = logging.getLogger(__name__)
    if file_config_error:
        logger.warning("日志文件输出启用失败，已降级为控制台输出: %s", file_config_error)
    elif log_to_file:
        logger.info(
            "日志文件输出已启用: %s (max_mb=%s backup_days=%s)",
            resolved_file_path,
            max(1, int(log_file_max_mb)),
            max(1, int(log_file_backup_days)),
        )


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
