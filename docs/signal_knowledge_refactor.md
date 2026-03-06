# 信号库与知识库拆分重构方案（双向量库）

## 1. 背景与问题

当前研报主链路中，实时信号与研究文档共用一套检索语义：

1. `collect_signals_via_mcp` 拉取实时/准实时信号。
2. `normalize_and_index` 将标准化后的信号写入统一研究集合。
3. `retrieve_context` 再从同一集合中召回“上下文”。

这会导致三个核心问题：

1. **语义混淆**：实时信号既是“本轮事实输入”，又被当作“背景证据”召回。
2. **证据重复**：刚入库的信号可能立刻被检索回来，形成“自己引用自己”的链路。
3. **产品能力缺口**：系统虽具备文档入库函数，但缺少面向用户的“上传知识库”能力，无法稳定沉淀白皮书、历史研报、治理提案、方法论文档等长期资料。

因此，需要将“实时信号层”和“知识证据层”彻底拆开，明确各自职责，并同步调整工作流节点命名与上传入口设计。

## 2. 重构目标

本次重构目标如下：

1. 拆分为两个独立向量库：`signal_chunks` 与 `knowledge_chunks`。
2. 将当前 `retrieve_context` 重命名为 `retrieve_knowledge_evidence`。
3. 明确研报生成输入为三路：
   - 当前实时信号
   - 知识库证据
   - 用户记忆画像
4. 新增“上传知识库”产品能力，使用户可上传并管理背景文档。
5. 保持现有 MCP 实时拉取主链路可用，不因重构影响已有出报能力。

## 3. 设计原则

### 3.1 职责单一

- `signal_chunks` 只承载实时/准实时市场信号。
- `knowledge_chunks` 只承载背景研究资料与可复用证据。
- `user_memory` 只承载用户偏好与会话上下文。

### 3.2 检索语义清晰

- 报告默认引用知识库证据，不默认引用本轮实时信号入库结果。
- 实时信号直接来自工作流状态，不通过“先入库再召回”的方式参与本轮结论生成。

### 3.3 兼容优先

- 保留现有实时信号入库能力，用于审计、回放、历史分析。
- 首阶段不强依赖全量迁移，先新增双库能力，再切换检索入口。

### 3.4 最小风险演进

- 先补存储与接口，再切工作流，再做迁移和前端增强。

## 4. 重构后的总体架构

### 4.1 三层数据职责

#### A. 实时信号层（`signal_chunks`）

- 来源：MCP tools、行情源、链上信号、快讯源。
- 作用：作为本轮研报的实时事实输入。
- 使用方式：由 `collect_signals_via_mcp` 输出并直接参与分析。
- 保留价值：用于回放、追溯、统计分析、历史信号复盘。

#### B. 知识证据层（`knowledge_chunks`）

- 来源：用户上传文档、内部历史研报、白皮书、治理提案、方法论文档、专题研究报告。
- 作用：为研报提供背景依据、历史参考、结构性解释与可引用证据。
- 使用方式：由 `retrieve_knowledge_evidence` 召回后进入报告生成 Prompt。

#### C. 用户记忆层（`user_memory`）

- 来源：用户偏好、关注标的、阅读习惯、会话摘要、最近轮次。
- 作用：用于个性化输出，不作为主要证据来源。

### 4.2 新工作流命名

建议将现有 9 节点流程调整为以下语义：

1. `load_user_memory`
2. `resolve_symbols`
3. `collect_signals_via_mcp`
4. `normalize_signals`
5. `index_signals`
6. `retrieve_knowledge_evidence`
7. `analyze_signals`
8. `generate_report`
9. `persist_memory`
10. `finalize_response`

说明：

- 若希望保持节点数量不变，可暂时保留 `normalize_and_index` 为一个节点，只将 `retrieve_context` 改名并改职责。
- 若希望后续更利于演进，推荐将“标准化”和“信号入库”拆开，便于未来将实时信号异步写库。

## 5. 向量库与数据模型设计

## 5.1 `signal_chunks`

用于存储实时或准实时结构化信号的切块。

建议字段：

- `id`
- `task_id`
- `doc_id`
- `chunk_id`
- `symbol`
- `source`
- `published_at`
- `text`
- `embedding`
- `metadata`

建议 `metadata`：

- `signal_type`
- `confidence`
- `raw_ref`
- `ingest_mode`
- `trace_id`（可选）

主要用途：

- 历史回放
- 审计追踪
- 未来的“历史信号对比”能力

默认不用于 `retrieve_knowledge_evidence`。

## 5.2 `knowledge_chunks`

用于存储知识库文档切块。

建议字段：

- `id`
- `kb_id`
- `doc_id`
- `chunk_id`
- `symbol`
- `source`
- `published_at`
- `text`
- `embedding`
- `metadata`

建议 `metadata`：

- `doc_type`
- `title`
- `author`
- `tags`
- `language`
- `file_name`
- `checksum`
- `uploaded_by`
- `is_active`

典型 `doc_type`：

- `whitepaper`
- `research_report`
- `governance_proposal`
- `methodology`
- `internal_memo`
- `event_note`

## 5.3 文档主表（推荐新增）

为了支撑上传、列表、删除与重建索引，建议新增文档元数据表，例如：`knowledge_document`。

建议字段：

- `doc_id`
- `kb_id`
- `title`
- `source`
- `doc_type`
- `symbols_json`
- `tags_json`
- `file_name`
- `content_type`
- `checksum`
- `status`（`processing | ready | failed | deleted`）
- `chunk_count`
- `uploaded_by`
- `created_at`
- `updated_at`

## 6. 检索与生成策略

### 6.1 本轮实时分析

本轮实时分析应直接依赖：

- `signals`
- `memory_profile`

而不是从向量库中重新召回本轮信号。

### 6.2 知识证据召回

`retrieve_knowledge_evidence` 仅检索 `knowledge_chunks`：

- 查询条件：`query + hard_symbols`
- 兜底策略：若按 `symbols` 检索不足，再放宽到全局知识库
- 返回对象：`knowledge_docs`

### 6.3 报告生成输入

报告生成阶段明确消费以下三路输入：

1. `signals`：本轮实时事实
2. `knowledge_docs`：背景知识证据
3. `memory_profile`：用户偏好与会话记忆

建议报告 Prompt 语义也随之调整：

- `信号摘要`：当前市场事实
- `知识证据摘要`：背景依据与历史资料
- `用户记忆画像`：个性化偏好

### 6.4 后续可选增强

如果后续确实需要“历史信号对比”，建议新增独立可选节点：

- `retrieve_signal_history`

其职责是：

- 基于当前 symbol 检索历史信号
- 用于趋势对比、相似事件回放
- 不与知识证据召回混用

## 7. 上传知识库能力设计

## 7.1 产品目标

支持用户将长期有效的背景文档上传到知识库，并在后续研报生成时作为可引用证据参与召回。

## 7.2 首版支持格式

建议首版支持：

- `.md`
- `.txt`
- `.pdf`
- `.docx`

## 7.3 API 设计

建议新增接口：

1. `POST /v1/knowledge/upload`
   - 文件上传并自动解析、切块、写入 `knowledge_chunks`
2. `POST /v1/knowledge/documents`
   - 直接提交文本内容入库
3. `GET /v1/knowledge/documents`
   - 获取文档列表与状态
4. `GET /v1/knowledge/documents/{doc_id}`
   - 获取单文档详情
5. `DELETE /v1/knowledge/documents/{doc_id}`
   - 软删除文档并从召回中剔除

## 7.4 上传元信息

建议上传时支持以下字段：

- `title`
- `source`
- `doc_type`
- `symbols`
- `tags`
- `published_at`
- `kb_id`（预留多知识库）
- `language`

## 7.5 前端能力

建议新增 `Knowledge Base` 页面，支持：

- 上传文件
- 文本直贴入库
- 编辑元信息
- 查看解析状态
- 查看 chunk 数量
- 删除/停用文档

## 8. 迁移策略

## Phase 0：文档与设计先行

- 输出本重构方案文档
- 统一命名：`retrieve_context` → `retrieve_knowledge_evidence`
- 明确双向量库边界与验收标准

## Phase 1：存储层拆库

- 在 Milvus 中新增 `signal_chunks` collection
- 在 Milvus 中新增 `knowledge_chunks` collection
- 扩展存储层接口：
  - `upsert_signal_chunks`
  - `search_signal_chunks`
  - `upsert_knowledge_chunks`
  - `search_knowledge_chunks`

目标：完成物理层与接口层分离，但暂不切主链路。

## Phase 2：服务层拆职责

- `ingest_signals()` 改为仅写入 `signal_chunks`
- `ingest_documents()` 改为仅写入 `knowledge_chunks`
- 新增 `retrieve_knowledge_evidence()` 或等价服务方法

目标：实时信号与知识文档在服务层彻底分开。

## Phase 3：工作流重命名与切换

- `retrieve_context` 改名为 `retrieve_knowledge_evidence`
- 工作流中改为只检索 `knowledge_chunks`
- `generate_report` 改为消费 `knowledge_docs`
- `citations` 默认来自知识库文档

目标：主链路完成语义切换。

## Phase 4：知识库上传与管理

- 新增上传接口
- 新增文档元数据表
- 新增列表、详情、删除能力
- 前端增加知识库管理页

目标：补齐产品层可操作能力。

## Phase 5：历史数据迁移（可选）

如果旧统一研究库中混有可复用文档，可编写迁移脚本：

- 识别历史记录类型
- 文档迁移到 `knowledge_chunks`
- 实时信号迁移到 `signal_chunks`
- 校验迁移数量与召回效果

该阶段可选，不阻塞首版重构上线。

## 9. 详细开发步骤（待确认后实施）

以下步骤为后续开发执行清单，确认后按顺序实施：

1. 修改 Milvus 存储封装，支持双 collection 初始化与读写。
2. 调整 `ResearchService`，拆分信号入库与知识文档入库职责。
3. 重构主工作流节点命名与状态字段：`retrieved_docs` → `knowledge_docs`。
4. 调整 `ReportAgent` Prompt，将“证据摘要”明确为“知识证据摘要”。
5. 保留并兼容现有 `/research/ingest`，同时新增 `/knowledge/*` 接口。
6. 增加知识库文档元数据持久化能力。
7. 增加知识库上传、列表、删除的后端测试。
8. 增加前端知识库管理页与上传交互。
9. 更新 `docs/architecture.md`，同步新的工作流与数据边界。
10. 补充迁移脚本或迁移说明（如需要）。

## 10. 风险与注意事项

1. **命名切换风险**：`retrieve_context` 改名后，文档、日志、测试、前端展示文案都要同步。
2. **旧接口兼容风险**：现有 `/research/ingest` 仍可能被外部调用，首阶段不应直接废弃。
3. **引用来源变化**：切换到知识库召回后，`citations` 数量和来源会变化，需校验输出是否更稳定。
4. **迁移复杂度**：如果历史统一集合中数据类型混杂严重，自动迁移可能需要额外规则。
5. **上传解析风险**：PDF/DOCX 的文本解析质量会影响召回效果，需要为失败状态提供可见反馈。

## 11. 验收标准

完成重构后，应满足以下标准：

1. 本轮实时信号不再通过知识证据召回链路重复进入报告。
2. 上传的知识库文档可被后续研报稳定召回并作为 citation 来源。
3. `retrieve_knowledge_evidence` 仅从 `knowledge_chunks` 检索。
4. 删除或停用知识库文档后，该文档不再参与召回。
5. 原有 MCP 实时信号采集与出报流程可继续正常工作。
6. 架构文档、接口文档、工作流命名与代码实现保持一致。

## 12. 推荐实施策略

建议采用**稳妥型实施顺序**：

1. 先补双库与接口
2. 再切工作流检索来源
3. 再补知识库上传产品能力
4. 最后补迁移与历史整理

原因：

- 改动路径更短
- 对现有出报链路影响更可控
- 便于逐步验证召回质量和报告效果

---

本文件用于指导后续“信号库 / 知识库”拆分重构。待方案确认后，再进入代码实施阶段。
