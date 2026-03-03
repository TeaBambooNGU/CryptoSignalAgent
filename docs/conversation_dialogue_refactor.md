# 会话对话化重构方案（3 阶段）

## 1. 背景与问题

当前 `POST /v1/research/query` 本质是单次任务接口，虽然已有 `conversation_id/turn_id/request_id/version`，但交互形态仍是“问一次、回一次研报”。

这带来三个直接问题：

1. 恢复会话价值有限：恢复后缺少持续对话入口。
2. 报告不可版本化：用户只能重新 query，无法围绕已生成报告继续编辑、改写。
3. 长会话上下文退化：多轮后缺少摘要压缩，提示上下文成本不断上升。

## 2. 重构目标

将系统升级为“对话优先 + 报告可重写 + 可持续恢复”的会话系统：

1. 用户可围绕报告持续对话（chat）。
2. 用户可在会话中触发“重写报告”（rewrite）与“重跑报告”（regenerate）。
3. 报告具备版本资产能力，可按 `report_id/report_version` 回放。
4. 长会话可通过摘要压缩保障上下文质量与性能。

## 3. 总体方案

### 3.1 统一入口

新增统一入口：

- `POST /v1/conversation/{conversation_id}/message`

由 `action` 决定执行模式：

- `chat`：基于历史与最新报告进行对话回复。
- `rewrite_report`：基于已有报告文本重写，通常不拉新外部数据。
- `regenerate_report`：调用现有 graph 重新拉取/分析并生成新报告。
- `auto`：自动路由（根据用户消息与上下文判定）。

### 3.2 双层持久化

- `conversation_turn`：每轮交互落库（用户消息 + 助手回复 + 意图 + 可选 report_id）。
- `conversation_report`：报告版本资产表（版本号、来源 turn、模式、内容、引用、workflow）。

### 3.3 上下文构建

对话/改写时上下文来源：

1. 会话摘要（长期压缩）
2. 最近 N 轮 turn（短窗口）
3. 最新报告或目标报告
4. 任务上下文与用户偏好

## 4. 数据模型设计

## 4.1 conversation_turn（扩展）

在现有字段基础上新增：

- `assistant_message_text`：助手回复正文（chat/rewrite/regenerate 统一存储）
- `intent`：`chat | rewrite_report | regenerate_report`
- `turn_type`：`assistant_chat | assistant_report`
- `parent_turn_id`：关联起点 turn（可选）
- `report_id`：本轮产出报告 ID（可选）

保留：`query_text/response_report/citations/errors/workflow_steps`，兼容现有报告链路。

### 4.2 conversation_report（新增）

- `report_id`（PK）
- `conversation_id`
- `report_version`（会话内递增）
- `created_by_turn_id`
- `based_on_report_id`（rewrite 链）
- `mode`（`rewrite|regenerate`）
- `report_text`
- `citations_json`
- `workflow_steps_json`
- `status`
- `created_at/updated_at`

约束：

- `UNIQUE(conversation_id, report_version)`

### 4.3 conversation_context_summary（新增）

- `conversation_id`（PK）
- `summary_text`
- `through_version`（摘要覆盖到的版本）
- `updated_at`

## 5. API 设计

### 5.1 新增

1. `POST /v1/conversation/{conversation_id}/message`
2. `GET /v1/conversation/{conversation_id}/reports`
3. `GET /v1/conversation/{conversation_id}/reports/{report_id}`

### 5.2 兼容

- 继续保留 `POST /v1/research/query`
- 语义上等价于 `action=regenerate_report`

## 6. 三阶段实施范围

## Phase 1：对话闭环

- 新增 `message` 入口
- 支持 `chat` 与 `regenerate_report`
- 每轮 turn 完整落库并可恢复

## Phase 2：报告版本化

- 新增 `conversation_report`
- 支持 `rewrite_report`
- 增加 report 查询 API

## Phase 3：上下文压缩

- 新增 `conversation_context_summary`
- 按版本推进摘要（摘要 + 最近窗口）
- 对话与报告执行统一使用压缩上下文

## 7. 关键一致性规则

1. 同会话单写者锁 + CAS version。
2. request_id 幂等：重试直接命中缓存结果。
3. turn 与 idempotency 状态在同事务提交。
4. 报告版本创建与 turn 绑定提交（避免“有 turn 无报告”）。

## 8. 验收标准

1. 单会话连续 20 轮（聊天/改写/重跑）可重启恢复。
2. 指定 `report_id` 能获取完整报告版本数据。
3. 同 `request_id` 重试返回同 `turn_id/version/action/result`。
4. 长会话摘要可稳定推进，且不影响最近轮次可追溯性。

