"""应用运行时容器。

将配置、基础设施、服务与工作流统一装配，供 API 层依赖注入使用。
"""

from __future__ import annotations

from app.agents.llm import create_llm_client
from app.agents.report_agent import ReportAgent
from app.config.logging import get_logger, log_context
from app.config.settings import Settings
from app.graph.workflow import ResearchGraphRunner
from app.memory.mem0_service import MemoryService
from app.observability.langsmith import configure_langsmith
from app.retrieval.milvus_store import MilvusStore
from app.retrieval.research_service import ResearchService
from app.tools.mcp_client import MCPClient

logger = get_logger(__name__)


class AppRuntime:
    """运行时依赖容器。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        with log_context(component="runtime.bootstrap"):
            logger.info("运行时初始化开始")
            configure_langsmith(settings)

            self.milvus_store = MilvusStore(settings)
            self.milvus_store.connect()
            logger.info("MilvusStore 初始化完成")

            self.llm_client = create_llm_client(settings)
            self.memory_service = MemoryService(settings=settings, milvus_store=self.milvus_store)
            self.mcp_client = MCPClient(settings=settings, llm_client=self.llm_client)
            self.research_service = ResearchService(settings=settings, milvus_store=self.milvus_store)
            logger.info("核心服务初始化完成")

            self.report_agent = ReportAgent(settings=settings, llm_client=self.llm_client)

            self.graph_runner = ResearchGraphRunner(
                memory_service=self.memory_service,
                mcp_client=self.mcp_client,
                research_service=self.research_service,
                report_agent=self.report_agent,
            )
            logger.info("运行时初始化完成")

    def close(self) -> None:
        """释放运行时资源。"""

        with log_context(component="runtime.shutdown"):
            self.milvus_store.close()
            logger.info("运行时资源已释放")
