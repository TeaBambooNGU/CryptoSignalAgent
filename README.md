# Crypto Signal Agent

> 对话优先的加密研究 Agent：整合 MCP 多源信号、记忆系统与可追溯报告版本。

![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white&style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135%2B-009688?logo=fastapi&logoColor=white&style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-Enabled-1C3C3C?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0%2B-1f6feb?style=flat-square)
![MCP](https://img.shields.io/badge/MCP-Integrated-6f42c1?style=flat-square)
![Milvus](https://img.shields.io/badge/Milvus-2.x-00A1EA?style=flat-square)
![Mem0](https://img.shields.io/badge/Mem0-Optional-FF6B6B?style=flat-square)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=000&style=flat-square)
![Vite](https://img.shields.io/badge/Vite-5-646CFF?logo=vite&logoColor=white&style=flat-square)
![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white&style=flat-square)

基于 `LangChain + LangGraph + MCP + Milvus + Mem0` 构建，面向“信号采集 -> 研报生成 -> 持续对话 -> 分支回放”的完整链路。
当前采用双向量库设计：`signal_chunks` 负责实时信号归档与复盘，`knowledge_chunks` 负责长期知识证据检索与引用。
演示视频：https://www.bilibili.com/video/BV1SUPvz3Egj
## 目录

- [Crypto Signal Agent](#crypto-signal-agent)
  - [目录](#目录)
  - [为什么选择它](#为什么选择它)
  - [核心能力](#核心能力)
  - [系统架构](#系统架构)
  - [快速开始](#快速开始)
    - [1) 环境准备](#1-环境准备)
    - [2) 安装依赖](#2-安装依赖)
    - [3) 配置环境变量](#3-配置环境变量)
    - [4) 初始化 Milvus（可选但推荐）](#4-初始化-milvus可选但推荐)
    - [5) 验证 MCP 可用性（推荐）](#5-验证-mcp-可用性推荐)
    - [6) 启动后端](#6-启动后端)
    - [7) 启动前端](#7-启动前端)
  - [配置说明](#配置说明)
  - [MCP 配置示例](#mcp-配置示例)
  - [API 快速体验](#api-快速体验)
    - [1) 生成首版研报](#1-生成首版研报)
    - [2) 基于当前会话继续追问](#2-基于当前会话继续追问)
    - [3) 使用 auto 自动判定动作](#3-使用-auto-自动判定动作)
    - [4) 上传知识库文档](#4-上传知识库文档)
    - [5) 查看知识库文档列表](#5-查看知识库文档列表)
    - [6) 查看报告版本历史](#6-查看报告版本历史)
  - [前端控制台](#前端控制台)
  - [开发与测试](#开发与测试)
  - [项目结构](#项目结构)
  - [常见问题](#常见问题)
  - [许可证](#许可证)
  - [风险声明](#风险声明)

## 为什么选择它

- **对话优先**：统一入口支持 `auto/chat/rewrite_report/regenerate_report`，其中 `auto` 优先由 DeepSeek 小模型判定动作。
- **可追溯**：会话与报告版本可回放，支持从历史 `turn` 分支恢复。
- **多源信号**：通过标准 MCP 协议接入行情、链上、新闻等数据。
- **工程化可观测**：请求级 `X-Trace-Id`、节点级耗时、重试与降级策略。

## 核心能力

- MCP 多服务工具发现与调用（`langchain-mcp-adapters`）
- 通过每个MCP对应一个子agent的方式,解决多MCP导致agent上下文爆炸的问题，同时多agent并行收集数据，优化了性能
- 每个子agent支持 MCP 工具失败后的自动重试与错误回传，具备基础自修复能力
- LangGraph 编排研究流程，输出 `workflow_steps`
- Milvus 双向量库：`signal_chunks` 用于实时信号归档与复盘，`knowledge_chunks` 用于知识证据检索
- 知识库上传与管理：支持纯文本创建、文件上传、列表查询、详情查看与删除
- 长期偏好自动抽取（使用小模型）并聚合为单条画像
- `action=auto` 优先使用 DeepSeek 小模型分类为 `chat / rewrite_report / regenerate_report`
- DeepSeek 动作分类失败或未配置时，自动回退到规则判断
- 会话一致性保障：`request_id` 幂等 + `expected_version` CAS
- 异步 outbox 投影：会话真相库强一致、外部记忆最终一致

## 系统架构

```mermaid
flowchart LR
    U[User / Frontend] --> API[FastAPI API]
    API --> CS[Conversation Service]
    API --> KAPI[Knowledge APIs]
    CS --> G[LangGraph Workflow]
    G --> MCP[MCP Servers]
    G --> SIG[(Milvus signal_chunks)]
    G --> KB[(Milvus knowledge_chunks)]
    G --> LLM[LLM Provider]
    KAPI --> KB
    CS --> TRUTH[(SQLite Truth Store)]
    CS --> OUTBOX[Outbox Projector]
    OUTBOX --> MEM[Milvus / Mem0 Memory]
```

- `signal_chunks`：保存实时/准实时信号，主要用于审计、回放与历史复盘。
- `knowledge_chunks`：保存白皮书、历史研报、治理提案等长期资料，供报告生成时引用。

详细架构说明见：[`docs/architecture.md`](docs/architecture.md)

## 快速开始

### 1) 环境准备

- Python `3.12+`
- [uv](https://docs.astral.sh/uv/)
- Node.js `18+`（前端需要）
- 可选：Milvus、Redis

### 2) 安装依赖

```bash
# 后端
uv sync

# 前端
npm --prefix frontend install
```

### 3) 配置环境变量

```bash
cp .env.example .env
```

最少请配置：

- `LLM_PROVIDER` + 对应密钥（默认 MiniMax）
- `DEEPSEEK_API_KEY`（用于长期偏好抽取与 `auto` 动作分类）
- `MCP_CONFIG_PATH` 指向 `.mcp.json`
- 若启用向量库：`MILVUS_URI`
- 建议同时确认集合配置：`MILVUS_SIGNAL_COLLECTION`、`MILVUS_KNOWLEDGE_COLLECTION`

### 4) 初始化 Milvus（可选但推荐）

```bash
uv run python scripts/init_milvus.py
```

如从旧单库版本升级，可额外清理历史遗留集合：

```bash
uv run python scripts/drop_legacy_research_collection.py
```

### 5) 验证 MCP 可用性（推荐）

```bash
uv run python scripts/verify_mcp_servers.py
```

### 6) 启动后端

```bash
uv run python main.py
```

启动后访问：<http://127.0.0.1:8000/docs>

### 7) 启动前端

```bash
npm --prefix frontend run dev
```

默认地址：<http://127.0.0.1:5173>

## 配置说明

完整变量请参考 [`.env.example`](.env.example)。

常用项：

| 分类 | 变量 | 说明 |
|---|---|---|
| 服务 | `APP_HOST` / `APP_PORT` | API 服务监听地址 |
| LLM | `LLM_PROVIDER` / `LLM_MODEL` | 模型供应商与模型名 |
| Embedding | `EMBEDDING_PROVIDER` / `ZHIPU_EMBEDDING_MODEL` | 向量化配置 |
| Milvus | `MILVUS_URI` / `MILVUS_SIGNAL_COLLECTION` / `MILVUS_KNOWLEDGE_COLLECTION` / `VECTOR_DIM` | 双向量库连接、集合名与维度 |
| 记忆抽取 | `MEMORY_EXTRACTOR_MODEL` / `MEMORY_EXTRACTOR_TIMEOUT_SECONDS` / `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` | 长期偏好自动抽取模型配置 |
| 动作分类 | `CONVERSATION_ACTION_MODEL` / `CONVERSATION_ACTION_TIMEOUT_SECONDS` / `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` | `action=auto` 的 DeepSeek 动作分类配置 |
| 会话 | `CONVERSATION_STORE_PATH` | SQLite 真相库存储路径 |
| Session | `SESSION_STORE_BACKEND` / `REDIS_URL` | 短期会话记忆存储 |
| MCP | `MCP_CONFIG_PATH` / `MCP_MAX_ROUNDS` | MCP 配置与调用预算 |
| Report | `REPORT_SIGNAL_DETAIL_LIMIT` / `REPORT_SIGNAL_VALUE_MAX_CHARS` | 研报 Prompt 中实时信号明细条数上限与单条 value 截断上限 |

## MCP 配置示例

项目默认读取根目录 `.mcp.json`：

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
      "cwd": "/path/to/cryptoNewsMCP",
      "env": {
        "CRYPTOPANIC_AUTH_TOKEN": "${CRYPTOPANIC_AUTH_TOKEN}"
      }
    }
  }
}
```

## API 快速体验

知识库相关接口已统一到 `/v1/knowledge/*`，旧 `/v1/research/ingest` 已移除。

### 1) 生成首版研报

```bash
curl -X POST http://127.0.0.1:8000/v1/conversation/conv-u001/message \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "u001",
    "message": "请给我 BTC 和 ETH 的 24 小时风险信号研报",
    "action": "regenerate_report",
    "task_context": {"symbols": ["BTC", "ETH"]},
    "request_id": "req-u001-turn1"
  }'
```

### 2) 基于当前会话继续追问

```bash
curl -X POST http://127.0.0.1:8000/v1/conversation/conv-u001/message \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "u001",
    "message": "请解释这版报告里最关键的下行风险",
    "action": "chat",
    "expected_version": 1,
    "request_id": "req-u001-turn2"
  }'
```

### 3) 使用 auto 自动判定动作

```bash
curl -X POST http://127.0.0.1:8000/v1/conversation/conv-u001/message \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "u001",
    "message": "把刚才那版 BTC 报告改成更保守的版本",
    "action": "auto",
    "expected_version": 2,
    "request_id": "req-u001-turn3"
  }'
```

说明：

- `action=auto` 会优先调用 DeepSeek 小模型判断当前请求更适合 `chat`、`rewrite_report` 还是 `regenerate_report`
- 若 `DEEPSEEK_API_KEY` 未配置，或分类模型调用失败，会自动回退到现有规则判断

### 4) 上传知识库文档

```bash
curl -X POST http://127.0.0.1:8000/v1/knowledge/upload \
  -F "user_id=u001" \
  -F "title=BTC ETF 周报" \
  -F "source=internal_research" \
  -F "doc_type=research_report" \
  -F "symbols=BTC,ETH" \
  -F "tags=etf,macro" \
  -F "kb_id=default" \
  -F "language=zh" \
  -F "file=@./docs/sample.pdf"
```

### 5) 查看知识库文档列表

```bash
curl "http://127.0.0.1:8000/v1/knowledge/documents?limit=20&kb_id=default"
```

### 6) 查看报告版本历史

```bash
curl "http://127.0.0.1:8000/v1/conversation/conv-u001/reports?limit=20"
```

更多接口请查看 OpenAPI 文档或 [`app/api/routes.py`](app/api/routes.py)。

## 前端控制台

前端位于 `frontend/`，技术栈：`React + Vite + TypeScript + TanStack Query + Zustand`。

- Message Composer：支持 `chat/rewrite/regenerate/auto`
- Dialogue：连续对话流
- Timeline + Branch Tree：按 `parent_turn_id` 回放分支
- Version Tape：报告版本选择与回放
- Knowledge 页面：位于 `/knowledge`，支持知识文档上传、列表、详情与删除

UI 联调辅助：

```bash
# 无头模式（自动截图+日志）
npm --prefix frontend run ui:debug

# 有头模式
npm --prefix frontend run ui:debug:live
```

## 开发与测试

```bash
# 后端回归
uv run python -m unittest tests.test_api tests.test_report_agent tests.test_workflow_symbols

# 前端构建验证
npm --prefix frontend run build

# 可选：修复历史长期偏好为单条画像结构（先 dry-run）
uv run python scripts/repair_user_memory_profile.py --dry-run

# 可选：MCP 原始响应巡检
uv run python scripts/inspect_mcp.py
```

## 项目结构

```text
.
├── app/                  # 后端核心代码
│   ├── api/              # HTTP 路由
│   ├── agents/           # LLM 代理与报告生成
│   ├── config/           # 配置与日志
│   ├── conversation/     # 会话真相库/幂等/CAS/outbox
│   ├── graph/            # LangGraph 工作流与 MCP 子图
│   ├── memory/           # 会话/长期记忆服务
│   ├── models/           # 数据模型
│   └── retrieval/        # 向量检索与入库
├── frontend/             # React 控制台
├── scripts/              # 初始化与诊断脚本
├── tests/                # 测试用例
├── docs/                 # 设计与架构文档
└── main.py               # 启动入口
```

## 常见问题

- **Q: 没配置 Milvus 能跑吗？**  
  A: 可以，按配置可降级为内存模式（生产仍建议启用 Milvus）。

- **Q: LLM 不可用时会怎样？**  
  A: `/v1/conversation/{conversation_id}/message` 会返回 500（硬失败）；知识库上传与检索也会受 Embedding / 向量库能力影响。

- **Q: `from_turn_id` 的作用？**  
  A: 作为分支锚点，后续上下文仅基于该分支链路构建。

- **Q: `action=auto` 现在怎么判定？**  
  A: 优先由 DeepSeek 小模型判定 `chat / rewrite_report / regenerate_report`；若 DeepSeek 未配置或分类失败，则回退到规则判断。

## 许可证

本项目采用 [MIT License](LICENSE)。

## 风险声明

本项目输出仅用于研究与信息参考，不构成任何投资建议。
