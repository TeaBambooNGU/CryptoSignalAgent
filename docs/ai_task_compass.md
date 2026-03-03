# AI任务罗盘：LangGraph + MCP 工具接入避坑指南

## 适用场景
- 将 MCP Server 工具接入 LangGraph / LangChain Agent。
- 出现 `NotImplementedError("StructuredTool does not support sync invocation.")`。
- 需要从“自定义子图/规则引擎”迁移到 `create_agent`。

## 一句话结论（根因）
- 根因不是 MCPSubgraph 这个命名本身，而是**执行链路误走了同步工具调用**：
  - MCP adapter 返回的部分工具是 async-only 的 `StructuredTool`；
  - 走 `invoke()` / 同步 ToolNode 时会触发 `StructuredTool does not support sync invocation.`。

## 正确架构（必须全链路异步）
1. 工具发现：`tools = await client.get_tools()`
2. Agent 调用：`result = await agent.ainvoke(input)`
3. 图执行：`output = await graph.ainvoke(state)`
4. API 入口：`async def` + `await graph_runner.arun(...)`

> 核心原则：**MCP tools 一律按 async-only 能力设计，禁止在主路径使用同步 invoke。**

## 本次落地的标准改造模板
- `mcp_subgraph.py`
  - 提供 `arun(...)` 作为主入口；
  - 强制 `agent.ainvoke(...)`；
  - `run(...)` 仅做兼容包装（`asyncio.run(...)`）。
- `workflow.py`
  - 提供 `arun(...)` 主入口；
  - 使用 `await self.graph.ainvoke(...)`；
  - MCP 节点改为 async 节点。
- `routes.py`
  - `/research/query` 改为 `async def`；
  - 调用 `await runtime.graph_runner.arun(...)`。
- LLM 客户端
  - 删除自定义 `BaseLLMClient` 包装；
  - 直接使用 LangChain 原生 `BaseChatModel/ChatOpenAI`。

## AI最容易犯的错误（下次禁止）
- 只改 `create_agent`，但仍在某处用同步 `invoke`。
- Graph 已异步，但路由/节点仍是同步函数，导致隐式走 sync path。
- 用“捕获异常后重试”掩盖架构问题，而不修正调用语义。
- 保留过多历史抽象层（如多余 LLM wrapper），让调用链不可见。

## 最小排查清单（5分钟）
- 搜索关键字：`invoke(`、`ainvoke(`、`async def`。
- 确认 MCP 相关链路均为：`get_tools -> agent.ainvoke -> graph.ainvoke`。
- 检查是否仍有老接口：`graph_runner.run` / `mcp_subgraph.run` 被主路径调用。
- 若报同类错，直接查看本地：
  - `.venv/lib/python*/site-packages/langchain_core/tools/structured.py`
  - `_run` 是否抛出 `StructuredTool does not support sync invocation.`

## 回归测试基线
- 单测必须覆盖：
  - async agent path 正常；
  - sync-not-supported 场景仍可通过 async path 运行；
  - API `/v1/research/query` 异步调用链可用。
- 推荐命令：
  - `uv run python -m unittest discover tests`

## Definition of Done（DoD）
- [ ] 主路径不再依赖同步工具调用。
- [ ] API / Graph / MCP Runner 三层全部支持并使用 async 主入口。
- [ ] 删除冗余抽象（尤其 LLM wrapper）后测试仍全绿。
- [ ] 架构文档同步到 async-only 事实实现。

## 参考（官方）
- LangChain MCP Adapters README（`await client.get_tools()` + `await agent.ainvoke(...)`）
  - https://github.com/langchain-ai/langchain-mcp-adapters/blob/main/README.md
- LangGraph prebuilt README（图调用可走 `ainvoke`）
  - https://github.com/langchain-ai/langgraph/blob/main/libs/prebuilt/README.md
