"""应用配置模块。

负责统一读取环境变量并提供强类型配置对象，避免在业务代码中散落 `os.getenv`。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


@dataclass(slots=True)
class Settings:
    """全局配置对象。

    字段说明：
    - 所有密钥均从环境变量读取，不允许硬编码。
    - `llm_provider` 用于支持可替换 LLM 客户端，当前默认 `minimax`。
    """

    app_name: str = "Crypto Signal Agent"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_env: str = "dev"
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file_path: str = "logs/app.log"
    log_file_max_mb: int = 10
    log_file_backup_days: int = 5

    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "crypto-signal-agent"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    llm_provider: str = "minimax"
    llm_model: str = "MiniMax-M2.5"
    llm_temperature: float = 0.2
    llm_timeout_seconds: int = 60

    minimax_api_key: str = ""
    minimax_api_host: str = "https://api.minimax.chat"

    openai_api_key: str = ""
    openai_base_url: str = ""

    embedding_provider: str = "zhipu"
    zhipu_embedding_model: str = "embedding-3"
    zhipu_embedding_batch_size: int = 64

    milvus_enabled: bool = True
    milvus_allow_fallback: bool = True
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_token: str = ""
    milvus_db_name: str = "default"
    milvus_research_collection: str = "research_chunks"
    milvus_memory_collection: str = "user_memory"
    vector_dim: int = 384

    mem0_enabled: bool = False
    mem0_mode: str = "platform"
    mem0_oss_collection: str = "mem0_memory"
    mem0_api_key: str = ""
    mem0_org_id: str = ""
    mem0_project_id: str = ""

    mcp_servers: dict[str, dict[str, Any]] = field(default_factory=dict)
    mcp_max_rounds: int = 4

    report_disclaimer: str = "免责声明：本报告仅用于研究与信息交流，不构成任何投资建议。"

    @classmethod
    def from_env(cls, env_file: str = ".env") -> "Settings":
        """从 `.env` 与系统环境变量加载配置。"""

        defaults = cls()
        env_path = Path(env_file)
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)

        def _as_bool(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            if raw is None:
                return default
            return raw.strip().lower() in {"1", "true", "yes", "on"}

        def _as_int(name: str, default: int) -> int:
            raw = os.getenv(name)
            if raw is None:
                return default
            try:
                return int(raw)
            except ValueError:
                return default

        def _as_float(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None:
                return default
            try:
                return float(raw)
            except ValueError:
                return default

        env_pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

        def _resolve_env_placeholders(value: Any) -> Any:
            if isinstance(value, str):
                return env_pattern.sub(lambda match: os.getenv(match.group(1), ""), value)
            if isinstance(value, list):
                return [_resolve_env_placeholders(item) for item in value]
            if isinstance(value, dict):
                return {
                    str(key): _resolve_env_placeholders(item)
                    for key, item in value.items()
                }
            return value

        def _as_mcp_servers(default_config_path: str = ".mcp.json") -> dict[str, dict[str, Any]]:
            config_path_raw = os.getenv("MCP_CONFIG_PATH", default_config_path).strip()
            if not config_path_raw:
                return {}

            config_path = Path(config_path_raw).expanduser()
            if not config_path.is_absolute():
                config_path = (env_path.parent / config_path).resolve()

            if not config_path.exists():
                return {}

            try:
                parsed: Any = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                return {}

            servers_block: Any = parsed
            if isinstance(parsed, dict) and isinstance(parsed.get("mcpServers"), dict):
                servers_block = parsed["mcpServers"]

            if not isinstance(servers_block, dict):
                return {}

            resolved = _resolve_env_placeholders(servers_block)
            if not isinstance(resolved, dict):
                return {}

            servers: dict[str, dict[str, Any]] = {}
            for raw_name, raw_spec in resolved.items():
                name = str(raw_name).strip()
                if not name or not isinstance(raw_spec, dict):
                    continue
                servers[name] = {str(key): value for key, value in raw_spec.items()}
            return servers

        def _normalize_minimax_host(default: str) -> str:
            raw_host = os.getenv("MINIMAX_API_HOST", "").strip()
            if raw_host:
                return raw_host.rstrip("/")
            legacy_base = os.getenv("MINIMAX_BASE_URL", "").strip().rstrip("/")
            if not legacy_base:
                return default
            if legacy_base.endswith("/v1"):
                return legacy_base[: -len("/v1")]
            return legacy_base

        return cls(
            app_name=os.getenv("APP_NAME", defaults.app_name),
            app_host=os.getenv("APP_HOST", defaults.app_host),
            app_port=_as_int("APP_PORT", defaults.app_port),
            app_env=os.getenv("APP_ENV", defaults.app_env),
            log_level=os.getenv("LOG_LEVEL", defaults.log_level),
            log_to_file=_as_bool("LOG_TO_FILE", defaults.log_to_file),
            log_file_path=os.getenv("LOG_FILE_PATH", defaults.log_file_path),
            log_file_max_mb=_as_int("LOG_FILE_MAX_MB", defaults.log_file_max_mb),
            log_file_backup_days=_as_int("LOG_FILE_BACKUP_DAYS", defaults.log_file_backup_days),
            langsmith_tracing=_as_bool("LANGSMITH_TRACING", defaults.langsmith_tracing),
            langsmith_api_key=os.getenv("LANGSMITH_API_KEY", ""),
            langsmith_project=os.getenv("LANGSMITH_PROJECT", defaults.langsmith_project),
            langsmith_endpoint=os.getenv("LANGSMITH_ENDPOINT", defaults.langsmith_endpoint),
            llm_provider=os.getenv("LLM_PROVIDER", defaults.llm_provider),
            llm_model=os.getenv("LLM_MODEL", defaults.llm_model),
            llm_temperature=_as_float("LLM_TEMPERATURE", defaults.llm_temperature),
            llm_timeout_seconds=_as_int("LLM_TIMEOUT_SECONDS", defaults.llm_timeout_seconds),
            minimax_api_key=os.getenv("MINIMAX_API_KEY", ""),
            minimax_api_host=_normalize_minimax_host(defaults.minimax_api_host),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_base_url=os.getenv("OPENAI_BASE_URL", ""),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", defaults.embedding_provider),
            zhipu_embedding_model=os.getenv("ZHIPU_EMBEDDING_MODEL", defaults.zhipu_embedding_model),
            zhipu_embedding_batch_size=_as_int(
                "ZHIPU_EMBEDDING_BATCH_SIZE",
                defaults.zhipu_embedding_batch_size,
            ),
            milvus_enabled=_as_bool("MILVUS_ENABLED", defaults.milvus_enabled),
            milvus_allow_fallback=_as_bool("MILVUS_ALLOW_FALLBACK", defaults.milvus_allow_fallback),
            milvus_uri=os.getenv("MILVUS_URI", defaults.milvus_uri),
            milvus_token=os.getenv("MILVUS_TOKEN", ""),
            milvus_db_name=os.getenv("MILVUS_DB_NAME", defaults.milvus_db_name),
            milvus_research_collection=os.getenv(
                "MILVUS_RESEARCH_COLLECTION",
                defaults.milvus_research_collection,
            ),
            milvus_memory_collection=os.getenv("MILVUS_MEMORY_COLLECTION", defaults.milvus_memory_collection),
            vector_dim=_as_int("VECTOR_DIM", defaults.vector_dim),
            mem0_enabled=_as_bool("MEM0_ENABLED", defaults.mem0_enabled),
            mem0_mode=os.getenv("MEM0_MODE", defaults.mem0_mode),
            mem0_oss_collection=os.getenv("MEM0_OSS_COLLECTION", defaults.mem0_oss_collection),
            mem0_api_key=os.getenv("MEM0_API_KEY", ""),
            mem0_org_id=os.getenv("MEM0_ORG_ID", ""),
            mem0_project_id=os.getenv("MEM0_PROJECT_ID", ""),
            mcp_servers=_as_mcp_servers(),
            mcp_max_rounds=_as_int("MCP_MAX_ROUNDS", defaults.mcp_max_rounds),
            report_disclaimer=os.getenv("REPORT_DISCLAIMER", defaults.report_disclaimer),
        )


settings = Settings.from_env()
