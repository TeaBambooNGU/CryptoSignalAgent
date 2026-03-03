# Crypto Signal Agent

基于 `LangChain + LangGraph + Mem0 + LlamaIndex + Milvus` 的加密市场信号研报 Agent。

## 核心能力

- 通过 **MCP 数据源**采集市场信号（基于官方 `langchain-mcp-adapters`）
- MCP 采用 `create_agent` 自主工具调用机制（async-only）
- 对信号做统一标准化并入库 Milvus
- 基于用户长期/短期记忆生成个性化研报，并支持围绕报告持续对话
- 使用 LangGraph 编排完整研究流程
- LLM 客户端可配置替换，当前默认接入 MiniMax M2.5
- 使用 tenacity 对关键节点提供重试

## MCP 架构（当前实现）

- 主流程节点 `collect_signals_via_mcp` 调用 `MCPSignalSubgraphRunner`
- 通过 `MultiServerMCPClient` 发现工具，`create_agent(...).ainvoke(...)` 执行工具调用
- 最终统一回收 `raw_signals / errors / mcp_tools_count / mcp_termination_reason`

## 工程结构

```text
/Users/teabamboo/Documents/AIplusLLM/CryptoSignalAgent
  app/
    api/
    agents/
    config/
    graph/
    memory/
    retrieval/
    tools/
    models/
    observability/
  scripts/
  tests/
  docs/
```

## 环境变量

建议创建 `.env`：

```env
APP_ENV=dev
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO
LOG_TO_FILE=false
LOG_FILE_PATH=logs/app.log
LOG_FILE_MAX_MB=10
LOG_FILE_BACKUP_DAYS=5

# LangSmith
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=crypto-signal-agent
LANGSMITH_ENDPOINT=https://api.smith.langchain.com

# 可替换 LLM 客户端
# 可选: minimax / openai
LLM_PROVIDER=minimax
LLM_MODEL=MiniMax-M2.5
LLM_TEMPERATURE=0.2
LLM_TIMEOUT_SECONDS=60

# MiniMax（OpenAI-compatible，使用 ChatOpenAI）
MINIMAX_API_KEY=
MINIMAX_API_HOST=https://api.minimax.chat

# OpenAI（可选，当 LLM_PROVIDER=openai 时必填 OPENAI_API_KEY）
OPENAI_API_KEY=
OPENAI_BASE_URL=

# Embedding（默认智谱）
EMBEDDING_PROVIDER=zhipu
ZHIPU_EMBEDDING_MODEL=embedding-3
ZHIPU_EMBEDDING_BATCH_SIZE=64
ZHIPUAI_API_KEY=

# Milvus
MILVUS_ENABLED=true
MILVUS_ALLOW_FALLBACK=true
MILVUS_URI=http://127.0.0.1:19530
MILVUS_TOKEN=
MILVUS_DB_NAME=default
MILVUS_RESEARCH_COLLECTION=research_chunks
MILVUS_MEMORY_COLLECTION=user_memory
VECTOR_DIM=384

# Mem0
MEM0_ENABLED=false
# 可选: platform / oss
MEM0_MODE=platform
# 仅在 MEM0_MODE=oss 时生效（建议独立 collection）
MEM0_OSS_COLLECTION=mem0_memory
# 可选：覆盖 OSS 模式下智谱 Embedding 的 OpenAI-compatible Base URL
ZHIPU_OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
# 仅在 MEM0_MODE=platform 时生效
MEM0_API_KEY=
MEM0_ORG_ID=
MEM0_PROJECT_ID=

# 会话一致性与短期记忆存储
CONVERSATION_STORE_PATH=data/conversation_state.db
# 可选: memory / redis
SESSION_STORE_BACKEND=memory
# 示例（带密码）：redis://:your_redis_password@127.0.0.1:6379/0
# 若启用 ACL 用户名，可用：redis://your_username:your_redis_password@127.0.0.1:6379/0
REDIS_URL=redis://:your_redis_password@127.0.0.1:6379/0
SESSION_MEMORY_TTL_SECONDS=86400
SESSION_MEMORY_MAX_ITEMS=50

# MCP（Claude Code 风格配置）
# 默认读取项目根目录 .mcp.json
MCP_CONFIG_PATH=.mcp.json
# 可选：供 .mcp.json 中 ${CRYPTOPANIC_AUTH_TOKEN} 占位符使用
CRYPTOPANIC_AUTH_TOKEN=
MCP_MAX_ROUNDS=4
```

## 启动方式

1. 初始化 Milvus（可选但推荐）：

```bash
uv run python scripts/init_milvus.py
```

2. 启动服务：

```bash
uv run python main.py
```

3. 访问文档：

- `http://127.0.0.1:8000/docs`

## API 列表

- 会话主入口（推荐）
  - `POST /v1/conversation/{conversation_id}/message`
- 会话查询与回放
  - `GET /v1/conversation/{conversation_id}`
  - `GET /v1/conversation/{conversation_id}/turns`
  - `GET /v1/conversation/{conversation_id}/turns/{turn_id}`
  - `GET /v1/conversation/{conversation_id}/reports`
  - `GET /v1/conversation/{conversation_id}/reports/{report_id}`
  - `POST /v1/conversation/{conversation_id}/resume`
- 兼容入口
  - `POST /v1/research/query`（语义等价于 `action=regenerate_report`）
- 其他接口
  - `POST /v1/user/preferences`
  - `GET /v1/user/profile/{user_id}`
  - `POST /v1/research/ingest`

## 前端控制台（极简科技风）

前端工程位于 `frontend/`，技术栈为 `React + Vite + TypeScript + TanStack Query + Zustand`。

1. 安装依赖：

```bash
cd frontend
npm install
```

2. 启动前端：

```bash
npm run dev
```

3. 默认通过 Vite 代理请求后端 `http://127.0.0.1:8000` 的 `/v1/*` 接口。  
   若需要自定义后端地址，设置环境变量：

```bash
VITE_API_BASE=http://127.0.0.1:8000
```

4. UI 联调（浏览器页面 + 终端日志）：

```bash
# 无头模式：自动抓日志与截图（推荐先用）
npm run ui:debug

# 有头模式：会真实打开 Chromium 页面
npm run ui:debug:live
```

截图输出到 `frontend/artifacts/`，终端会打印：
- 浏览器 Console（`[browser-console:*]`）
- 页面运行错误（`[browser-pageerror]`）
- 失败请求/4xx/5xx（`[browser-requestfailed]` / `[browser-http-*]`）

## 标准 MCP 配置示例（Claude Code 风格）

```json
{
  "mcpServers": {
    "coingecko": {
      "type": "http",
      "url": "https://mcp.api.coingecko.com/mcp"
    },
    "defillama": {
      "type": "http",
      "url": "https://mcpllama.com/mcp"
    },
    "cryptonews": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "crypto-news-mcp"],
      "cwd": "/Users/teabamboo/Documents/AIplusLLM/cryptorNewsMCP",
      "env": {
        "CRYPTOPANIC_AUTH_TOKEN": "${CRYPTOPANIC_AUTH_TOKEN}"
      }
    }
  }
}
```

`MCP_CONFIG_PATH` 默认是 `.mcp.json`，你可以从 `.mcp.json.example` 复制一份开始配置。

- `coingecko`：行情/币种/趋势数据
- `defillama`：链上 TVL/协议维度数据
- `crypto-news-mcp`：CryptoPanic 新闻/投研信号（stdio，本地 `uv run crypto-news-mcp`）

对接 `crypto-news-mcp` 前请先在对应目录完成一次初始化：

```bash
cd /Users/teabamboo/Documents/AIplusLLM/cryptorNewsMCP
uv sync
```

可用性验证：

```bash
uv run python scripts/verify_mcp_servers.py
```

## 用户如何拿到研报（会话模式）

推荐直接使用会话入口：`POST /v1/conversation/{conversation_id}/message`。

1. 先生成首版研报（`regenerate_report`）

```bash
curl -X POST http://127.0.0.1:8000/v1/conversation/conv-u001-1/message \
  -H "Content-Type: application/json" \
  -d '{
    "user_id":"u001",
    "message":"请给我 BTC 和 ETH 的 24 小时风险信号研报",
    "action":"regenerate_report",
    "task_context":{"symbols":["BTC","ETH"]},
    "request_id":"req-u001-turn1"
  }'
```

2. 基于该研报继续对话（`chat`）

```bash
curl -X POST http://127.0.0.1:8000/v1/conversation/conv-u001-1/message \
  -H "Content-Type: application/json" \
  -d '{
    "user_id":"u001",
    "message":"请解释这版报告里最关键的下行风险",
    "action":"chat",
    "expected_version":1,
    "request_id":"req-u001-turn2"
  }'
```

3. 在对话中重写报告（`rewrite_report`）

```bash
curl -X POST http://127.0.0.1:8000/v1/conversation/conv-u001-1/message \
  -H "Content-Type: application/json" \
  -d '{
    "user_id":"u001",
    "message":"改写成更保守、面向低风险投资者的版本",
    "action":"rewrite_report",
    "target_report_id":"rpt-xxxxxx",
    "expected_version":2,
    "request_id":"req-u001-turn3"
  }'
```

4. 查看报告版本历史

```bash
curl "http://127.0.0.1:8000/v1/conversation/conv-u001-1/reports?limit=20"
```

> 若你仍在使用 `POST /v1/research/query`，它会走兼容路径，行为等价于 `regenerate_report`。

## 测试

```bash
uv run python -m unittest tests/test_api.py
```

## 关键说明

- V1 严格只处理 MCP 可获取的数据源。
- 报告默认附带风险免责声明：仅供研究，不构成投资建议。
- 当 Milvus 不可用时，系统可按配置降级到内存存储。
- 当 LLM 不可用时，`/v1/research/query` 与 `/v1/conversation/{conversation_id}/message` 都会返回 500（硬失败）。
- 使用智谱 embedding 时，请确保 `VECTOR_DIM` 与模型输出维度一致。
- 记忆写入采用 outbox projector 异步投影（Milvus/Mem0 最终一致）；会话真相状态先落地本地 SQLite。
- 若要启用 `SESSION_STORE_BACKEND=redis`，请先安装 Python Redis 客户端（例如 `uv add redis`）。
