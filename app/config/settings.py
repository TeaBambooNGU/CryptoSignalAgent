"""应用配置模块。

负责统一读取环境变量并提供强类型配置对象，避免在业务代码中散落 `os.getenv`。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
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

    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "crypto-signal-agent"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    llm_provider: str = "minimax"
    llm_model: str = "MiniMax-M2.5"
    llm_temperature: float = 0.2
    llm_timeout_seconds: int = 60

    minimax_api_key: str = ""
    minimax_base_url: str = "https://api.minimax.chat/v1"

    openai_compatible_api_key: str = ""
    openai_compatible_base_url: str = ""

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
    mem0_api_key: str = ""
    mem0_org_id: str = ""
    mem0_project_id: str = ""

    mcp_tool_endpoints: tuple[str, ...] = ()

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

        def _as_tuple(name: str) -> tuple[str, ...]:
            raw = os.getenv(name, "")
            if not raw:
                return ()
            # 优先支持 JSON 数组，便于多环境配置。
            try:
                parsed: Any = json.loads(raw)
                if isinstance(parsed, list):
                    return tuple(str(item).strip() for item in parsed if str(item).strip())
            except json.JSONDecodeError:
                pass
            return tuple(part.strip() for part in raw.split(",") if part.strip())

        return cls(
            app_name=os.getenv("APP_NAME", defaults.app_name),
            app_host=os.getenv("APP_HOST", defaults.app_host),
            app_port=_as_int("APP_PORT", defaults.app_port),
            app_env=os.getenv("APP_ENV", defaults.app_env),
            log_level=os.getenv("LOG_LEVEL", defaults.log_level),
            langsmith_tracing=_as_bool("LANGSMITH_TRACING", defaults.langsmith_tracing),
            langsmith_api_key=os.getenv("LANGSMITH_API_KEY", ""),
            langsmith_project=os.getenv("LANGSMITH_PROJECT", defaults.langsmith_project),
            langsmith_endpoint=os.getenv("LANGSMITH_ENDPOINT", defaults.langsmith_endpoint),
            llm_provider=os.getenv("LLM_PROVIDER", defaults.llm_provider),
            llm_model=os.getenv("LLM_MODEL", defaults.llm_model),
            llm_temperature=_as_float("LLM_TEMPERATURE", defaults.llm_temperature),
            llm_timeout_seconds=_as_int("LLM_TIMEOUT_SECONDS", defaults.llm_timeout_seconds),
            minimax_api_key=os.getenv("MINIMAX_API_KEY", ""),
            minimax_base_url=os.getenv("MINIMAX_BASE_URL", defaults.minimax_base_url),
            openai_compatible_api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY", ""),
            openai_compatible_base_url=os.getenv("OPENAI_COMPATIBLE_BASE_URL", ""),
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
            mem0_api_key=os.getenv("MEM0_API_KEY", ""),
            mem0_org_id=os.getenv("MEM0_ORG_ID", ""),
            mem0_project_id=os.getenv("MEM0_PROJECT_ID", ""),
            mcp_tool_endpoints=_as_tuple("MCP_TOOL_ENDPOINTS"),
            report_disclaimer=os.getenv("REPORT_DISCLAIMER", defaults.report_disclaimer),
        )


settings = Settings.from_env()
